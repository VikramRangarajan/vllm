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
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)


class Args(BaseSettings, cli_parse_args=True, cli_ignore_unknown_args=True):
    model: str = "google/gemma-4-31B-it"
    output_dir: str = "./outputs"
    evals: list[str] = ["wikitext", "arc_easy"]

    # Experiments
    baseline: bool = False
    quant: bool = True
    load_in_8bit: bool = True
    my_method_kv: bool = True

    # Quant hyperparameters
    bit_width: int = 8

    # Quant QAT hyperparameters
    finetune_dataset: str = "wikitext"

    # My method hyperparameters
    monte_carlo_steps: int = 1
    sigma: float = 0.001
    bsz: int = 1


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
    # Note: quantized params (e.g., Int8Params) are integer-typed and
    # cannot require gradients; we skip those and seed via lm_head below.
    for name, param in model.named_parameters():
        if name in kv_weight_names and param.is_floating_point():
            param.requires_grad_(True)

    # If the model is loaded in quantized format, the K/V weight params
    # are integer-typed and can't require gradients. Seed the autograd
    # graph via the lm_head or embedding layer instead.
    if not any(
        p.requires_grad for n, p in model.named_parameters() if n in kv_weight_names
    ):
        for name, param in model.named_parameters():
            if (
                "lm_head" in name or "embed_tokens" in name
            ) and param.is_floating_point():
                param.requires_grad_(True)
                print(f"[kv_sensitivity] Seeding autograd graph via: {name}")
                break

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
    # Keys: layer_idx -> {"rel_mse": ..., "rel_l2": ..., "raw_mse": ..., "raw_l2": ...}
    noise_sensitivities: dict[str, dict[str, float]] = defaultdict(
        lambda: {"rel_mse": 0.0, "rel_l2": 0.0, "raw_mse": 0.0, "raw_l2": 0.0}
    )

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
                    sen_rel_mse = 0.0
                    sen_rel_l2 = 0.0
                    sen_raw_mse = 0.0
                    sen_raw_l2 = 0.0
                    count = 0
                    for proj in ("k", "v"):
                        key = (layer_idx, proj)
                        if key in clean_grads and key in noisy_grads:
                            g = clean_grads[key]
                            d = g - noisy_grads[key]
                            norm_g = g.norm(p=2).clamp(min=1e-12)
                            norm_d = d.norm(p=2)
                            sen_rel_mse += norm_d.square() / norm_g.square()
                            sen_rel_l2 += norm_d / norm_g
                            sen_raw_mse += norm_d.square()
                            sen_raw_l2 += norm_d
                            count += 1
                    if count > 0:
                        scale = 1.0 / count / args.monte_carlo_steps
                        ns = noise_sensitivities[layer_idx]
                        ns["rel_mse"] += sen_rel_mse.item() * scale
                        ns["rel_l2"] += sen_rel_l2.item() * scale
                        ns["raw_mse"] += sen_raw_mse.item() * scale
                        ns["raw_l2"] += sen_raw_l2.item() * scale

            # Clean up after each batch to keep peak memory low
            del clean_grads, noisy_grads, out
            gc.collect()
            torch.accelerator.empty_cache()

    capture.cleanup()

    # -------------------------------------------------------------------
    # 3. Allocate bit-widths for each metric
    # -------------------------------------------------------------------
    keys = list(noise_sensitivities.keys())

    def _allocate_bits(sens_vals: list[float]) -> list[int]:
        s = torch.tensor(sens_vals, dtype=torch.float)
        if s.max() < 1e-6:
            bw = torch.full_like(s, float(args.bit_width))
        else:
            mean_s = s.mean().clamp(min=1e-12)
            bw = (s / mean_s) * args.bit_width
            bw = bw.clamp(1.0, 16.0)
        return [round(b) for b in bw.tolist()]

    metric_names = ("rel_mse", "rel_l2", "raw_mse", "raw_l2")

    for metric_name in metric_names:
        vals = [noise_sensitivities[k][metric_name] for k in keys]
        bitwidths = _allocate_bits(vals)
        print(f"\n[kv_sensitivity] Bit-widths from {metric_name}:")
        for k, bw in zip(keys, bitwidths):
            print(f"  layer {k}: {bw}")

    # Build result dict with all metrics and bit-width sets
    result: dict[str, dict[str, float | list]] = {}
    for k in keys:
        entry = {}
        for metric_name in metric_names:
            entry[f"sens_{metric_name}"] = noise_sensitivities[k][metric_name]
        result[k] = entry

    for metric_name in metric_names:
        vals = [noise_sensitivities[k][metric_name] for k in keys]
        bitwidths = _allocate_bits(vals)
        for k, bw in zip(keys, bitwidths):
            result[k][f"bw_{metric_name}"] = bw

    # Attach per-layer architectural metadata
    # Gemma-4-31B-it: 60 layers, 50 sliding + 10 full (every 6th).
    # Sliding: 16 KV heads, 256 head_dim, RoPE theta=1e4, window=1024.
    # Full:    4 KV heads (k_eq_v), 512 head_dim, proportional RoPE theta=1e6.
    full_layers = {5, 11, 17, 23, 29, 35, 41, 47, 53, 59}
    for k in keys:
        layer_idx = int(k)
        is_full = layer_idx in full_layers
        result[k]["layer_type"] = "full" if is_full else "sliding"
        result[k]["num_kv_heads"] = 4 if is_full else 16
        result[k]["head_dim"] = 512 if is_full else 256
        result[k]["k_eq_v"] = is_full

    print("\n[kv_sensitivity] Per-layer results")
    pp(result)
    return result


# ---------------------------------------------------------------------------
#  Dataset utilities
# ---------------------------------------------------------------------------


def get_wikitext(tokenizer):
    max_length = tokenizer.model_max_length
    if max_length > 8192:
        max_length = 512

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=max_length)

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

if args.quant:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=not args.load_in_8bit,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    quant_type = "8-bit" if args.load_in_8bit else "4-bit"
    print(f"[main] Loaded {args.model} in {quant_type} quantized format")
else:
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

ds = get_dataset(args.finetune_dataset, tokenizer)

if args.my_method_kv:
    # ------------------------------------------------------------------
    #  KV-cache quantization: per-layer fake-quant bit widths
    # ------------------------------------------------------------------
    kv_bit_per_layer = get_kv_cache_optimal_config(model, tokenizer, ds, args)
    # Save to a JSON file so it can be passed to vLLM's CacheConfig
    os.makedirs(args.output_dir, exist_ok=True)

    name = f"M{args.monte_carlo_steps}_S{args.sigma}_BW{args.bit_width}"

    out_path = os.path.join(args.output_dir, f"results_{name}.json")
    with open(out_path, "w") as f:
        json.dump(kv_bit_per_layer, f, indent=2)
    print(f"[main] KV-cache per-layer bit widths saved to {out_path}")
