# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import json
import os
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from pprint import pp

import regex as re
import torch
from datasets import load_dataset
from pydantic_settings import BaseSettings
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


class Args(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    model: str = "google/gemma-4-12B-it"
    output_dir: str = "./outputs"
    evals: list[str] = ["wikitext", "arc_easy"]

    # Experiments
    baseline: bool = False
    quant: bool = False
    my_method_kv: bool = True

    # Quant hyperparameters
    bit_width: int = 8

    # Quant QAT hyperparameters
    finetune_dataset: str = "wikitext"

    # My method hyperparameters
    monte_carlo_steps: int = 1
    sigma: float = 0.001
    bsz: int = 2


# ---------------------------------------------------------------------------
#  Gradient checkpointing helper (reduces OOM on large models like Gemma 4 12B)
# ---------------------------------------------------------------------------


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing to trade compute for memory.

    Puts the model into training mode so that checkpointing wraps the forward
    passes.  Call *after* selective ``requires_grad_()`` on the parameters
    that need to seed the autograd graph.
    """
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        grad_ckpt_kwargs: dict = {}
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=grad_ckpt_kwargs
            )
        except (TypeError, ValueError):
            # Older HF versions don't accept the kwargs argument.
            model.gradient_checkpointing_enable()
        print("[gradient_checkpointing] enabled")
    else:
        print(
            "[gradient_checkpointing] WARNING – model does not support "
            "gradient_checkpointing_enable(); will try to proceed without it"
        )


# ---------------------------------------------------------------------------
#  Bit-width annealing (shared by weight and KV-cache methods)
# ---------------------------------------------------------------------------


def _anneal_bitwidths(
    sensitivities: torch.Tensor,
    sizes: torch.Tensor,
    target_bit_width: float,
) -> torch.Tensor:
    """Map sensitivity values to bit-widths under a total-size constraint.

    The algorithm searches for the largest "temperature" that keeps all
    allocated bit-widths in the valid range [1, 16], then decreases the
    temperature until the constraint is satisfied.

    Args:
        sensitivities: 1-D tensor of shape ``(N,)`` – one per layer/group.
        sizes: 1-D tensor of shape ``(N,)`` – number of elements per group.
        target_bit_width: desired *average* bit-width.

    Returns:
        1-D tensor of shape ``(N,)`` with bit-widths in ``[1, 16]``.
    """
    bitwidths = torch.zeros_like(sensitivities) + 4

    def is_valid(bw: torch.Tensor) -> bool:
        return bool(bw.min() >= 1 and bw.max() <= 16)

    start_temp = 1.0
    for _ in range(100):
        if not is_valid(bitwidths):
            break
        weights = (sensitivities * start_temp).softmax(-1)
        bitwidths = weights * target_bit_width * sizes.sum() / (weights * sizes).sum()
        start_temp *= 2.0

    for temperature in range(100, -1, -1):
        weights = (sensitivities * start_temp * (temperature / 100)).softmax(-1)
        bitwidths = weights * target_bit_width * sizes.sum() / (weights * sizes).sum()
        if is_valid(bitwidths):
            break

    print("Final Temperature:", temperature)

    assert bitwidths.min() >= 1 and bitwidths.max() <= 16, "Should be impossible"
    return bitwidths


# ---------------------------------------------------------------------------
#  KV-cache noise sensitivity (new method)
# ---------------------------------------------------------------------------


def find_kv_projection_modules(
    model: nn.Module,
) -> OrderedDict[str, dict[str, str | bool]]:
    """Discover per-layer ``k_proj`` / ``v_proj`` modules and weight names.

    Returns an ``OrderedDict`` mapping layer-index strings (e.g. ``"0"``,
    ``"1"``, …) to dicts::

        {
            "k_proj": "model.layers.N.self_attn.k_proj",
            "v_proj": "model.layers.N.self_attn.v_proj",
            "k_eq_v": False,
        }

    When a layer has ``k_eq_v=True`` (e.g. K=V fused attention in
    Gemma 4 full-attention layers), the ``v_proj`` value is the same as
    ``k_proj`` — there is no separate ``v_proj`` module.
    """
    by_layer: OrderedDict[str, dict[str, str | bool]] = OrderedDict()

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        proj: str | None = None
        if name.endswith(".k_proj") or name.endswith(".key_proj"):
            proj = "k_proj"
        elif name.endswith(".v_proj") or name.endswith(".value_proj"):
            proj = "v_proj"
        else:
            continue

        m = re.search(r"\.layers\.(\d+)\.", name)
        if m is None:
            m = re.search(r"\.h\.(\d+)\.", name)
        if m is None:
            m = re.search(r"\.block\.(\d+)\.", name)
        if m is None:
            m = re.search(r"\.encoder\.(\d+)\.", name)
        if m is None:
            m = re.search(r"(\d+)", name.split(".")[-2])
        if m is None:
            continue

        layer_idx = m.group(1)
        if layer_idx not in by_layer:
            by_layer[layer_idx] = {"k_proj": None, "v_proj": None, "k_eq_v": False}

        if proj == "k_proj":
            by_layer[layer_idx]["k_proj"] = name
        else:
            by_layer[layer_idx]["v_proj"] = name

    # K=V fused layers: point v_proj to the same module as k_proj
    for info in by_layer.values():
        if info["k_proj"] is not None and info["v_proj"] is None:
            info["v_proj"] = info["k_proj"]
            info["k_eq_v"] = True

    return OrderedDict({k: v for k, v in by_layer.items() if v["k_proj"] is not None})


# ---------------------------------------------------------------------------
#  KV-cache noise injection via forward hooks
# ---------------------------------------------------------------------------


@contextmanager
def perturb_kv_activations(
    model: nn.Module, noise_std: float, kv_module_names: set[str]
):
    """Add Gaussian noise directly to K/V activation values.

    Registers forward hooks on every ``k_proj`` and ``v_proj`` module
    listed in *kv_module_names*.  Each hook adds i.i.d. noise to the
    projection output (i.e. the K / V values that would be stored in the
    KV cache), simulating quantization noise.

    Safe to use with gradient checkpointing: PyTorch's ``checkpoint``
    preserves the RNG state by default (``preserve_rng_state=True``),
    so the noise is identical during forward and backward replay.
    """
    hooks = []

    def _make_hook(std: float):
        return lambda mod, inp, out: out + torch.randn_like(out) * std

    for name, module in model.named_modules():
        if name in kv_module_names:
            hooks.append(module.register_forward_hook(_make_hook(noise_std)))

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


# ---------------------------------------------------------------------------
#  K/V value gradient capture via backward hooks
# ---------------------------------------------------------------------------


class KVGradCapture:
    """Captures ``∂L/∂K`` and ``∂L/∂V`` for every attention layer.

    Registers forward hooks on each ``k_proj`` / ``v_proj`` module.
    Inside each forward hook it registers a **backward** hook on the
    module's output tensor.  During ``loss.backward()`` those backward
    hooks receive ``∂L/∂K`` and ``∂L/∂V``, which are accumulated into
    ``self.grads``.

    Usage::

        capture = KVGradCapture(model, kv_layers)
        model.zero_grad()
        loss.backward()
        # capture.grads[(layer_idx, 'k')] now holds ∂L/∂K for that layer
    """

    def __init__(
        self,
        model: nn.Module,
        kv_layers: OrderedDict[str, dict[str, str | bool]],
    ):
        self.grads: dict[tuple[str, str], torch.Tensor] = {}
        self._hooks = []
        self._register(model, kv_layers)

    def _register(
        self, model: nn.Module, kv_layers: OrderedDict[str, dict[str, str | bool]]
    ) -> None:
        name_to_meta: dict[str, tuple[str, str]] = {}
        for layer_idx, info in kv_layers.items():
            name_to_meta[info["k_proj"]] = (layer_idx, "k")
            if not info.get("k_eq_v", False):
                name_to_meta[info["v_proj"]] = (layer_idx, "v")

        for name, module in model.named_modules():
            if name not in name_to_meta:
                continue
            layer_idx, proj = name_to_meta[name]

            def _make_forward_hook(lidx: str, proj: str):
                def _forward_hook(
                    _mod: nn.Module, _inp, out: torch.Tensor
                ) -> torch.Tensor:
                    if out.requires_grad:

                        def _backward_hook(grad: torch.Tensor) -> None:
                            key = (lidx, proj)
                            g = grad.detach().clone()
                            if key in self.grads:
                                self.grads[key] += g
                            else:
                                self.grads[key] = g

                        out.register_hook(_backward_hook)
                    return out

                return _forward_hook

            self._hooks.append(
                module.register_forward_hook(_make_forward_hook(layer_idx, proj))
            )

    def reset(self) -> None:
        """Clear all captured gradients."""
        self.grads.clear()

    def cleanup(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


def get_kv_cache_optimal_config(
    model, tokenizer, ds, args: Args
) -> dict[str, dict[str, float]]:
    """Determine per-layer KV-cache bit widths via noise sensitivity.

    The method directly perturbs the **K/V activation values** (the values
    that would be stored in the KV cache), not the projection weights::

        1. Register backward hooks on ``k_proj`` / ``v_proj`` outputs to
           capture ``∂L/∂K`` and ``∂L/∂V`` for every attention layer.
        2. For each batch in the dataset:
           a. Clean pass – forward + backward, capture clean K/V gradients.
           b. Noisy pass – add Gaussian noise to K/V values, forward +
              backward, capture noisy K/V gradients.
           c. Per-layer sensitivity for this batch =
              ``MSE(∂L/∂K_clean, ∂L/∂K_noisy) / ||∂L/∂K_clean||²``.
        3. Average sensitivity across all batches.
        4. Allocate bit-widths with the same annealing algorithm.

    Batch-level processing is required because K/V gradients for different
    sequence lengths have different tensor shapes and cannot be accumulated.

    Gradient checkpointing is enabled to reduce peak memory on large models
    (PyTorch preserves the RNG state across checkpoint forward/backward,
    so the noise hooks produce identical noise during replay).

    Returns
    -------
    ``dict[str, int]``
        Mapping from layer-index strings to bit-widths, suitable for
        ``CacheConfig.per_layer_kv_cache_fake_quant_bits``.
    """
    model.zero_grad()

    enable_gradient_checkpointing(model)

    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    loader = DataLoader(ds, batch_size=args.bsz, collate_fn=collate_fn)
    num_batches = len(loader)

    kv_layers = find_kv_projection_modules(model)
    print(
        f"[kv_sensitivity] Found {len(kv_layers)} layers with separate K/V projections"
    )
    if not kv_layers:
        raise RuntimeError(
            "No separate k_proj / v_proj layers found. "
            "This method requires a model with un-fused K and V projections."
        )

    # Build KV weight names for selective requires_grad.
    kv_weight_names: set[str] = set()
    for info in kv_layers.values():
        kv_weight_names.add(f"{info['k_proj']}.weight")
        kv_weight_names.add(f"{info['v_proj']}.weight")

    # Only K/V projection weights need requires_grad.
    # During checkpoint replay, every layer has at least one K/V weight
    # with grad enabled, which seeds the autograd graph through the
    # rest of the layer. All other parameters stay grad-free, saving
    # ~24 GB of gradient memory on a 12B model.
    for name, param in model.named_parameters():
        param.requires_grad_(name in kv_weight_names)

    # Collect module names used for noise injection
    kv_module_names: set[str] = set()
    for info in kv_layers.values():
        kv_module_names.add(info["k_proj"])
        kv_module_names.add(info["v_proj"])

    # Gradient capture for ∂L/∂K and ∂L/∂V (always active)
    capture = KVGradCapture(model, kv_layers)

    # Each batch has a different sequence length, so K/V gradient tensor
    # shapes vary. We compute sensitivity per batch and average, rather
    # than accumulating full gradient tensors across batches.
    noise_sensitivities: dict[str, float] = defaultdict(float)

    for mc_iter in range(args.monte_carlo_steps):
        print(f"[kv_sensitivity] MC iteration {mc_iter + 1}/{args.monte_carlo_steps}")

        progress = tqdm(loader, desc="Computing per-batch sensitivity")
        for batch in progress:
            batch = {
                k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # --- Clean pass -------------------------------------------------
            model.zero_grad(set_to_none=True)
            capture.reset()
            out = model(**batch)
            (out.loss / num_batches).backward()
            clean_grads = dict(capture.grads)

            # --- Noisy pass -------------------------------------------------
            model.zero_grad(set_to_none=True)
            capture.reset()
            with perturb_kv_activations(model, args.sigma, kv_module_names):
                out = model(**batch)
                (out.loss / num_batches).backward()
            noisy_grads = dict(capture.grads)

            # --- Per-batch sensitivity --------------------------------------
            with torch.no_grad():
                for layer_idx in kv_layers:
                    sen = 0.0
                    count = 0
                    for proj in ("k", "v"):
                        key = (layer_idx, proj)
                        if key in clean_grads and key in noisy_grads:
                            normalizer = (
                                clean_grads[key].norm(p=2).square().clamp(min=1e-12)
                            )
                            sen += (
                                nn.functional.mse_loss(
                                    clean_grads[key], noisy_grads[key]
                                )
                                / normalizer
                            )
                            count += 1
                    if count > 0:
                        noise_sensitivities[layer_idx] += (
                            sen / count / args.monte_carlo_steps
                        ).item()

            # Clean up after each batch to keep peak memory low
            del clean_grads, noisy_grads, out
            gc.collect()
            torch.accelerator.empty_cache()

    capture.cleanup()

    # -------------------------------------------------------------------
    # 3. Allocate bit-widths
    # -------------------------------------------------------------------
    keys = list(noise_sensitivities.keys())
    sensitivities = torch.tensor(
        [noise_sensitivities[k] for k in keys], dtype=torch.float
    )

    # Allocate bit-widths proportional to relative sensitivity.
    #
    # Each layer gets bitwidth proportional to its sensitivity relative to
    # the mean, scaled by target_bit_width, then clamped to [1, 16].
    # This avoids the temperature-compression issue of _anneal_bitwidths:
    # softmax(sens * T) forces a narrow distribution when sensitivities
    # span a modest dynamic range, which is typical for KV gradients.
    #
    # Degenerate case: if all layers have the same sensitivity (including
    # sigma=0 where all values are near zero), every layer gets target_bw.
    if sensitivities.max() < 1e-6:
        bitwidths = torch.full_like(sensitivities, float(args.bit_width))
    else:
        mean_sens = sensitivities.mean().clamp(min=1e-12)
        bitwidths = (sensitivities / mean_sens) * args.bit_width
        bitwidths = bitwidths.clamp(1.0, 16.0)

    result: dict[str, dict[str, float]] = {"bw": {}, "sens": {}}
    for k, bw in zip(keys, bitwidths.tolist()):
        result["bw"][k] = round(bw)
        result["sens"][k] = noise_sensitivities[k]

    print("[kv_sensitivity] Per-layer KV-cache bit widths")
    pp(result)
    return result["bw"]


# ---------------------------------------------------------------------------
#  Dataset utilities
# ---------------------------------------------------------------------------


def get_wikitext(tokenizer):
    def tokenize(example):
        return tokenizer(example["text"], truncation=True)

    def filter_empty(example):
        return len(example["text"]) > 0

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:100]")
    ds = ds.filter(filter_empty).map(tokenize, batched=True, remove_columns=["text"])
    return ds


def get_dataset(ds: str, tokenizer):
    if ds == "wikitext":
        return get_wikitext(tokenizer)
    raise NotImplementedError("Dataset Not Set Up")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

args = Args()
tokenizer = AutoTokenizer.from_pretrained(args.model)
fp_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
ds = get_dataset(args.finetune_dataset, tokenizer)

if args.my_method_kv:
    # ------------------------------------------------------------------
    #  KV-cache quantization: per-layer fake-quant bit widths
    # ------------------------------------------------------------------
    kv_bit_per_layer = get_kv_cache_optimal_config(fp_model, tokenizer, ds, args)
    # Save to a JSON file so it can be passed to vLLM's CacheConfig
    os.makedirs(args.output_dir, exist_ok=True)

    name = f"M{args.monte_carlo_steps}_S{args.sigma}_BW{args.bit_width}"

    out_path = os.path.join(args.output_dir, f"results_{name}.json")
    with open(out_path, "w") as f:
        json.dump(kv_bit_per_layer, f, indent=2)
    print(f"[main] KV-cache per-layer bit widths saved to {out_path}")
