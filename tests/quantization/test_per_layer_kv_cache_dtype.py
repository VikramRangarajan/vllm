# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-layer KV cache dtype overrides."""

import pytest


@pytest.mark.skip_if_model_unavailable("facebook/opt-125m")
def test_per_layer_kv_cache_dtype(vllm_runner):
    """Verify per-layer kv cache dtype overrides are applied correctly."""
    with vllm_runner(
        "facebook/opt-125m",
        kv_cache_dtype="fp8",
        per_layer_kv_cache_dtype={"0": "auto", "2": "auto"},
        enforce_eager=True,
    ) as llm:

        def check_layers(model):
            for i, layer in enumerate(model.model.decoder.layers):
                expected = "auto" if str(i) in ("0", "2") else "fp8"
                actual = layer.self_attn.attn.kv_cache_dtype
                assert actual == expected, (
                    f"Layer {i}: expected kv_cache_dtype={expected!r}, got {actual!r}"
                )
                print(
                    f"  Layer {i}: kv_cache_dtype={actual!r} ",
                    f"(expected {expected!r})  OK",
                )

        llm.apply_model(check_layers)


@pytest.mark.skip_if_model_unavailable("facebook/opt-125m")
def test_per_layer_kv_cache_dtype_skip_override(vllm_runner):
    """per_layer_kv_cache_dtype takes precedence over kv_cache_dtype_skip_layers."""
    with vllm_runner(
        "facebook/opt-125m",
        kv_cache_dtype="fp8",
        kv_cache_dtype_skip_layers=["0", "2"],
        per_layer_kv_cache_dtype={"0": "fp8"},  # re-enable fp8 for layer 0
        enforce_eager=True,
    ) as llm:

        def check_layers(model):
            for i, layer in enumerate(model.model.decoder.layers):
                if i == 0:
                    # per_layer overrides skip → fp8
                    expected = "fp8"
                elif str(i) in ("2",):
                    # skip layers stays as "auto"
                    expected = "auto"
                else:
                    expected = "fp8"
                actual = layer.self_attn.attn.kv_cache_dtype
                assert actual == expected, (
                    f"Layer {i}: expected kv_cache_dtype={expected!r}, got {actual!r}"
                )
                print(
                    f"  Layer {i}: kv_cache_dtype={actual!r} ",
                    f"(expected {expected!r})  OK",
                )

        llm.apply_model(check_layers)


@pytest.mark.skip_if_model_unavailable("facebook/opt-125m")
def test_per_layer_kv_cache_dtype_output(vllm_runner):
    """Verify the model can generate with per-layer dtype overrides."""
    with vllm_runner(
        "facebook/opt-125m",
        kv_cache_dtype="fp8",
        per_layer_kv_cache_dtype={"0": "auto", "2": "auto"},
        enforce_eager=True,
    ) as llm:
        output = llm.generate_greedy("Hello, my name is", max_tokens=10)
        print(f"  Generated output: {output}")
        assert len(output) > 0, "Generation should return output"


def test_per_layer_kv_cache_dtype_config():
    """Verify config parsing works without a model."""
    from vllm.config.cache import CacheConfig

    config = CacheConfig(
        block_size=16,
        cache_dtype="fp8",
        per_layer_kv_cache_dtype={
            "0": "auto",
            "2": "int8_per_token_head",
            "4": "nvfp4",
        },
    )
    assert config.per_layer_kv_cache_dtype == {
        "0": "auto",
        "2": "int8_per_token_head",
        "4": "nvfp4",
    }
    assert "0" in config.per_layer_kv_cache_dtype
    assert config.per_layer_kv_cache_dtype["0"] == "auto"
    assert config.per_layer_kv_cache_dtype["2"] == "int8_per_token_head"
    print("  Config parsing: OK")


def test_layer_index_extraction():
    """Verify extract_layer_index works with typical layer prefixes."""
    from vllm.model_executor.models.utils import extract_layer_index

    cases = [
        ("model.layers.0.self_attn", 0),
        ("model.layers.1.self_attn", 1),
        ("model.layers.5.self_attn", 5),
        ("model.layers.12.self_attn", 12),
        ("encoder.layers.0", 0),
        ("2.self_attn", 2),
    ]
    for prefix, expected in cases:
        result = extract_layer_index(prefix)
        assert result == expected, (
            f"extract_layer_index({prefix!r}) = {result}, expected {expected}"
        )
        print(f"  extract_layer_index({prefix!r}) = {result}  OK")
