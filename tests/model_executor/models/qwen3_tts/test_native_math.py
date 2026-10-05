# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the native arithmetic correctness mode (no seeding)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

spec = importlib.util.spec_from_file_location(
    "qwen_native_math", Path(__file__).parents[4] / "vllm_omni/model_executor/models/qwen3_tts/native_math.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_native_norm_rounds_residual_before_variance():
    x = (torch.arange(2048).float().sin() * 2).to(torch.bfloat16)[None]
    residual = (torch.arange(2048).float().cos() / 3).to(torch.bfloat16)[None]
    norm = SimpleNamespace(weight=torch.linspace(0.9, 1.1, 2048).to(torch.bfloat16), variance_epsilon=1e-6)
    result, updated = module.norm_forward(norm, x, residual)
    summed = x + residual
    fp32 = summed.float()
    expected = (fp32 * torch.rsqrt(module.legacy_mean(fp32.square()) + 1e-6)).to(x.dtype) * norm.weight
    assert torch.equal(updated, summed)
    assert torch.equal(result, expected)
    fused = x.float() + residual.float()
    fused = (fused * torch.rsqrt(module.legacy_mean(fused.square()) + 1e-6)).to(x.dtype) * norm.weight
    assert not torch.equal(result, fused)


def test_invalid_serving_configuration_fails():
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False, dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_seqs=2, enable_chunked_prefill=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
    )
    with pytest.raises(ValueError, match="max_num_seqs=1"):
        module.validate_config(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA flash attention required")
def test_cache_resets_between_requests_and_rejects_stale_decode():
    def values(length, phase):
        x = torch.arange(length * 128, device="cuda").float()
        k = (x.sin() + phase).to(torch.bfloat16).reshape(1, 1, length, 128)
        return k.repeat_interleave(2, 1), k, (x.cos() - phase).to(torch.bfloat16).reshape_as(k)

    cache = module.SerialFlashCache()
    q, k, v = values(3, 0)
    cache.attend(q, k, v, torch.arange(3, device="cuda"), 0.125)
    q, k, v = values(1, 1)
    cache.attend(q, k, v, torch.tensor([3], device="cuda"), 0.125)
    q, k, v = values(2, 2)
    actual = cache.attend(q, k, v, torch.arange(2, device="cuda"), 0.125)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        expected = F.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), scale=0.125, is_causal=True
        )
    assert torch.equal(actual, expected)
    assert cache.key.shape[2] == 2
    with pytest.raises(RuntimeError, match="complete prefill"):
        cache.attend(*values(1, 3), torch.tensor([7], device="cuda"), 0.125)


def test_fp32_temperature_can_change_top_k_membership():
    logits = torch.tensor([[8.0, 7.96875, 7.9375]], dtype=torch.bfloat16)
    from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapper

    native = CodePredictorWrapper._temperature_logits(logits, 0.9)
    rounded = logits * (1 / 0.9)
    # BF16 scaling collapses distinct candidates into tied top-k values.
    assert torch.unique(native).numel() == 3
    assert torch.unique(rounded).numel() < 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reduction golden values")
@pytest.mark.parametrize(
    "cols,rows,bits",
    [
        (64, 1, 1176330284),
        (128, 1, 1174187117),
        (128, 8, 1174187117),
        (128, 16, 1174187117),
        (1024, 1, 1175843924),
        (1024, 17, 1175843924),
        (2048, 1, 1175700491),
        (2048, 215, 1175700491),
    ],
)
def test_torch27_cuda_mean_golden(cols, rows, bits):
    # Captured independently using x.mean(-1) on native Torch 2.7 / H100.
    index = torch.arange(cols, device="cuda")
    x = (
        (((index * 113 % 997).float() / 32).square() * torch.exp2((index % 17 - 8).float()))
        .expand(rows, -1)
        .contiguous()
    )
    actual = module.legacy_mean(x)
    assert torch.all(actual.view(torch.int32) == bits)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Serving GPU penalty implementation")
def test_generated_only_penalty_does_not_penalize_embedding_placeholders():
    from dataclasses import dataclass

    from vllm.v1.sample.sampler import Sampler

    @dataclass
    class Metadata:
        prompt_token_ids: torch.Tensor
        no_penalties: bool
        presence_penalties: torch.Tensor
        frequency_penalties: torch.Tensor
        repetition_penalties: torch.Tensor

    raw = torch.tensor([[3.0, 2.0, -1.0], [3.0, 2.0, -1.0]], device="cuda")
    metadata = Metadata(
        torch.ones((2, 7), device="cuda", dtype=torch.long),
        False,
        torch.zeros(2, device="cuda"),
        torch.zeros(2, device="cuda"),
        torch.full((2,), 1.05, device="cuda"),
    )
    clean = module.generated_only_sampling_metadata(metadata)
    assert metadata.prompt_token_ids.shape == (2, 7)
    assert clean.prompt_token_ids.shape == (2, 0)
    actual = Sampler.apply_penalties(raw.clone(), clean, [[2], [1]])
    expected = raw.clone()
    expected[0, 2] *= metadata.repetition_penalties[0]
    expected[1, 1] /= metadata.repetition_penalties[1]
    assert torch.equal(actual, expected)
    old = Sampler.apply_penalties(raw.clone(), metadata, [[2], [1]])
    assert old[0, 1] != expected[0, 1]
