# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType

import pytest
import torch
from torch import nn
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock
from vllm.model_executor.layers.linear import QKVParallelLinear

from vllm_omni.model_executor.models.spev.attention_backend import checkpoint_attention
from vllm_omni.model_executor.models.spev.candidate_tree import CandidateTree
from vllm_omni.model_executor.models.spev.checkpoint_ops import CheckpointExperts, CheckpointQKV, canonical_projection
from vllm_omni.model_executor.models.spev.decision_layout import DecisionLayout
from vllm_omni.model_executor.models.spev.isolated_attention import isolated_batch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def layouts():
    item = DecisionLayout(CandidateTree(3, (3, 3)), (5, 8), ((3,), (6,)), ((20, 21), (22, 23)), (0, 1), (0, 1))
    return [item, item]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qkv_adapters_and_request_isolation(dtype):
    torch.manual_seed(78)
    source = nn.Linear(64, 128, bias=False).to(dtype)
    source.load_weights = MethodType(QKVParallelLinear.load_weights, source)
    model = CheckpointQKV(source, 64, 32)
    qa, qb, va, vb = torch.randn(2, 64), torch.randn(64, 2), torch.randn(2, 64), torch.randn(32, 2)
    model.bind_adapters(qa, qb, va, vb, 2)
    assert model.weight is source.weight
    value = torch.randn(18, 64, dtype=dtype)

    def projection(x, start, end, adapter=None):
        base = nn.functional.linear(x, source.weight[start:end])
        if adapter is None:
            return base
        a, b = adapter
        return base + (nn.functional.linear(nn.functional.linear(x.float(), a), b) * 2).to(x.dtype)

    with isolated_batch(layouts()):
        actual, _ = model(value)
        expected = torch.cat(
            [
                canonical_projection(value, lambda x: projection(x, 0, 64, (qa, qb))),
                canonical_projection(value, lambda x: projection(x, 64, 96)),
                canonical_projection(value, lambda x: projection(x, 96, 128, (va, vb))),
            ],
            dim=-1,
        )
    with isolated_batch(layouts()[:1]):
        alone, _ = model(value[:9])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[:9], alone, atol=0, rtol=0)


def test_qkv_delegates_loading_with_native_shard_metadata():
    source = nn.Linear(64, 128, bias=False)
    source.load_weights = MethodType(QKVParallelLinear.load_weights, source)
    source.validate_shard_id = lambda shard: None
    source.prefix = "qkv"
    forwarded = []
    source.weight.weight_loader = lambda parameter, value, shard: forwarded.append((parameter, value, shard))
    model = CheckpointQKV(source, 64, 32)
    value = torch.randn(32, 64)
    value.shard_id = "k"
    assert list(model.load_weights(iter([("weight", value)]))) == ["weight"]
    assert forwarded == [(model.weight, value, "k")]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_experts_reuse_loaded_weights_and_match_hf_grouped_mm(dtype):
    torch.manual_seed(78)
    config = Qwen3OmniMoeTextConfig(hidden_size=16, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8)
    config._experts_implementation = "grouped_mm"
    reference = Qwen3OmniMoeThinkerTextSparseMoeBlock(config).to(dtype).eval()
    with torch.no_grad():
        for value in reference.parameters():
            value.normal_(0, 0.1)
    source = nn.Module()
    source.gate = nn.Linear(16, 4, bias=False).to(dtype)
    source.gate.weight = reference.gate.weight
    source.experts = nn.Module()
    source.experts.routed_experts = nn.Module()
    source.experts.routed_experts.w13_weight = reference.experts.gate_up_proj
    source.experts.routed_experts.w2_weight = reference.experts.down_proj
    model = CheckpointExperts(source, config)
    assert model.gate_up_proj is reference.experts.gate_up_proj
    assert model.down_proj is reference.experts.down_proj
    assert len(list(model.parameters())) == 3
    value = torch.randn(18, 16, dtype=dtype)
    with torch.no_grad(), isolated_batch(layouts()):
        actual = model(value)
        expected = torch.cat([reference(value[:9][None])[0], reference(value[9:][None])[0]])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_attention_scope_restores_imported_runtime_defaults():
    original = torch.backends.cuda.cudnn_sdp_enabled()
    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
        with pytest.raises(ValueError), checkpoint_attention():
            assert torch.backends.cuda.cudnn_sdp_enabled()
            raise ValueError("test scope cleanup")
        assert not torch.backends.cuda.cudnn_sdp_enabled()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(original)
