# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextSparseMoeBlock
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.utils import process_weights_after_loading

from vllm_omni.model_executor.models.spev.attention_backend import checkpoint_attention
from vllm_omni.model_executor.models.spev.candidate_tree import CandidateTree
from vllm_omni.model_executor.models.spev.checkpoint_ops import (
    CheckpointExperts,
    CheckpointExpertWeights,
    CheckpointQKV,
    canonical_projection,
)
from vllm_omni.model_executor.models.spev.decision_layout import DecisionLayout
from vllm_omni.model_executor.models.spev.expert_operations import batched_linear, grouped_product
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
@pytest.mark.parametrize("lengths", [(9,), (9, 9), (9, 5, 13)])
def test_experts_reuse_loaded_weights_and_match_hf_grouped_mm(dtype, lengths, monkeypatch):
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
    source.experts.load_weights = lambda weights: (name for name, _ in weights)
    model = CheckpointExperts(source, config)
    assert model.gate_up_proj is reference.experts.gate_up_proj
    assert model.down_proj is reference.experts.down_proj
    assert len(list(model.parameters())) == 3
    assert list(model.experts.load_weights([("w13_weight", 1)])) == ["routed_experts.w13_weight"]
    assert model.experts is not source.experts
    assert model.experts.routed_experts is not source.experts.routed_experts
    assert all(not hasattr(module, "quant_method") for module in model.experts.modules())
    value = torch.randn(sum(lengths), 16, dtype=dtype)
    with torch.no_grad():
        expected = torch.cat([reference(part[None])[0] for part in value.split(lengths)])
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.spev import checkpoint_ops

    calls = []
    expert_forward = checkpoint_ops.expert_forward

    def experts(self, hidden, indices, weights):
        calls.append(("experts", len(hidden)))
        return expert_forward(self, hidden, indices, weights)

    monkeypatch.setattr(checkpoint_ops, "expert_forward", experts)
    items = [SimpleNamespace(tree=CandidateTree(1, (length - 1,))) for length in lengths]
    with torch.no_grad(), isolated_batch(items):
        actual = model(value)
    torch.testing.assert_close(
        actual, expected, atol=1e-8 if dtype == torch.float32 and lengths == (9, 5, 13) else 0, rtol=0
    )
    assert calls == [("experts", sum(lengths))]
    value.requires_grad_()
    with isolated_batch(items):
        model(value)[0].sum().backward()
    assert value.grad[0].abs().sum() > 0
    assert torch.count_nonzero(value.grad[1:]) == 0


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA grouped-matmul regression")
def test_cuda_expert_rows_are_invariant_to_other_expert_sizes(monkeypatch):
    torch.manual_seed(943)
    weight = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    prefix = torch.randn(12, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    other = torch.randn(68, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    calls = []
    original = torch.nn.functional.grouped_mm

    def product(value, weight, **kwargs):
        calls.append((tuple(value.shape), kwargs.get("offs")))
        return original(value, weight, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "grouped_mm", product)
    baseline = grouped_product(
        torch.cat((prefix, other)), weight, prefix.new_tensor([12, 12, 49, 80], dtype=torch.int32)
    )
    changed = grouped_product(
        torch.cat((other[:9], prefix, other)), weight, prefix.new_tensor([9, 21, 58, 89], dtype=torch.int32)
    )
    # Compare the same rows under the same expert while preceding groups and total work change.
    same_expert = grouped_product(
        torch.cat((other[:9], prefix, other)), weight, prefix.new_tensor([0, 9, 21, 89], dtype=torch.int32)
    )
    reference = grouped_product(prefix, weight, prefix.new_tensor([0, 0, 12, 12], dtype=torch.int32))
    torch.testing.assert_close(same_expert[9:21], reference, atol=0, rtol=0)
    same_expert[9].float().sum().backward()
    assert prefix.grad[0].abs().sum() > 0
    assert torch.count_nonzero(prefix.grad[1:]) == torch.count_nonzero(other.grad) == 0
    assert baseline.shape == (80, 16) and changed.shape == (89, 16)
    assert calls and all(shape == (4, 32, 32) and offsets is None for shape, offsets in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA projection batch regression")
@pytest.mark.parametrize("bias", [False, True])
def test_cuda_projection_is_invariant_to_request_batch_size(bias, monkeypatch):
    torch.manual_seed(943)
    weight = torch.randn(512, 2048, device="cuda", dtype=torch.bfloat16)
    offset = torch.randn(512, device="cuda", dtype=torch.bfloat16) if bias else None
    inputs = torch.randn(17, 56, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    calls = []
    name = "baddbmm" if bias else "bmm"
    original = getattr(torch, name)

    def product(*args, **kwargs):
        calls.append(tuple(args[-2].shape))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, name, product)
    expected = batched_linear(inputs[:1], weight, offset)
    for count in (2, 4, 8, 9, 16, 17):
        actual = batched_linear(inputs[:count], weight, offset)
        torch.testing.assert_close(actual[:1], expected, atol=0, rtol=0)
        last = batched_linear(inputs[count - 1 : count], weight, offset)
        torch.testing.assert_close(actual[-1:], last, atol=0, rtol=0)
    actual[0, 0].float().sum().backward()
    assert inputs.grad[0, 0].abs().sum() > 0
    assert torch.count_nonzero(inputs.grad[0, 1:]) == torch.count_nonzero(inputs.grad[1:]) == 0
    assert calls and all(shape == (8, 56, 2048) for shape in calls)


def test_checkpoint_expert_weights_skip_unused_backend_repacking():
    calls = []

    class RepackingMethod(QuantizeMethodBase):
        requires_device_loading = False

        def create_weights(self, layer, *args, **kwargs):
            raise AssertionError("Weights already exist")

        def apply(self, layer, *args, **kwargs):
            raise AssertionError("SPEV uses checkpoint expert execution")

        def process_weights_after_loading(self, layer):
            calls.append(layer)
            layer.w13_weight = nn.Parameter(layer.w13_weight.flip(1))

    source = nn.Module()
    source.routed_experts = nn.Module()
    weights = torch.arange(32.0).reshape(2, 4, 4)
    source.routed_experts.w13_weight = nn.Parameter(weights.clone())
    source.routed_experts.w2_weight = nn.Parameter(weights.clone())
    source.routed_experts.quant_method = RepackingMethod()
    source.load_weights = lambda values: iter(values)
    model = CheckpointExpertWeights(source)
    config = SimpleNamespace(word_embeddings_untied_by_checkpoint=False, dtype=torch.float32, quantization=None)
    process_weights_after_loading(model, config, torch.device("cpu"))
    assert calls == []
    torch.testing.assert_close(model.routed_experts.w13_weight, weights, atol=0, rtol=0)
    process_weights_after_loading(source, config, torch.device("cpu"))
    assert calls == [source.routed_experts]
    torch.testing.assert_close(source.routed_experts.w13_weight, weights.flip(1), atol=0, rtol=0)
    torch.testing.assert_close(model.routed_experts.w13_weight, weights, atol=0, rtol=0)
