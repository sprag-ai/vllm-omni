# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-row results of SPEV's invariant kernels across batch sizes and positions."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def rows_in_batches(operation, value):
    alone = operation(value[:7])
    small = operation(value[:33])[:7]
    shifted = operation(torch.cat([value[40:], value[:7]]))[-7:]
    return alone, small, shifted


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_invariant_linear_ignores_batch(dtype):
    from vllm_omni.model_executor.models.spev.invariant_ops import invariant_linear

    torch.manual_seed(5)
    value = torch.randn(300, 2048, device="cuda", dtype=dtype)
    weight = torch.randn(1024, 2048, device="cuda", dtype=dtype) / 45
    bias = torch.randn(1024, device="cuda", dtype=dtype)
    alone, small, shifted = rows_in_batches(lambda x: invariant_linear(x, weight, bias), value)
    assert torch.equal(alone, small)
    assert torch.equal(alone, shifted)


def test_invariant_linear_fp32_matches_ieee_reference():
    from vllm_omni.model_executor.models.spev.invariant_ops import invariant_linear

    torch.manual_seed(6)
    value = torch.randn(65, 2048, device="cuda", dtype=torch.float64)
    weight = torch.randn(96, 2048, device="cuda", dtype=torch.float64) / 45
    output = invariant_linear(value.float(), weight.float())
    torch.testing.assert_close(output.double(), value @ weight.t(), atol=1e-4, rtol=0)


def test_invariant_experts_ignore_batch():
    from vllm_omni.model_executor.models.spev.invariant_ops import invariant_experts

    torch.manual_seed(7)
    experts, hidden, intermediate, top_k = 16, 2048, 768, 8
    value = torch.randn(300, hidden, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.randn(experts, 2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16) / 45
    down = torch.randn(experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16) / 28
    logits = torch.randn(300, experts, device="cuda")
    weights, indices = logits.softmax(-1).topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)

    def run(selection):
        return invariant_experts(value[selection], gate_up, down, weights[selection], indices[selection])

    alone = run(torch.arange(7, device="cuda"))
    small = run(torch.arange(33, device="cuda"))[:7]
    shifted = run(torch.cat([torch.arange(40, 300, device="cuda"), torch.arange(7, device="cuda")]))[-7:]
    assert torch.equal(alone, small)
    assert torch.equal(alone, shifted)


def branch_requests(trees):
    from vllm_omni.model_executor.models.spev.branch_attention import RequestMaskCache

    requests, offset = [], 0
    for tree in trees:
        requests.append((offset, offset + tree.length, tree, RequestMaskCache()))
        offset += tree.length
    return requests, offset


def flash_attention_versions():
    return [pytest.param(2, id="fa2"), pytest.param(3, id="fa3")]


def require_flash_attention(version):
    from vllm.vllm_flash_attn.flash_attn_interface import is_fa_version_supported

    if not is_fa_version_supported(version):
        pytest.skip(f"FlashAttention {version} is unavailable on this GPU")


@pytest.mark.parametrize("version", flash_attention_versions())
def test_varlen_branch_attention_matches_masked_plan_and_ignores_batch(version):
    from vllm_omni.model_executor.models.spev.branch_attention import BatchAttentionPlan, VarlenAttentionPlan
    from vllm_omni.model_executor.models.spev.candidate_tree import CandidateTree

    require_flash_attention(version)
    torch.manual_seed(8)
    trees = [CandidateTree(301, (5, 33, 2)), CandidateTree(77, (40,)), CandidateTree(512, (3, 3, 9, 64))]
    requests, tokens = branch_requests(trees)
    query = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn(tokens, 4, 128, device="cuda", dtype=torch.bfloat16)
    joint = VarlenAttentionPlan(requests, query.device, version).forward(query, key, value, 128**-0.5)
    reference = BatchAttentionPlan(requests, query.device).forward(query, key, value, 128**-0.5)
    torch.testing.assert_close(joint.float(), reference.float(), atol=2e-2, rtol=0)
    start, end = requests[1][:2]
    alone, _ = branch_requests(trees[1:2])
    single = VarlenAttentionPlan(alone, query.device, version).forward(
        query[start:end], key[start:end], value[start:end], 128**-0.5
    )
    assert torch.equal(single, joint[start:end])
    reordered, _ = branch_requests([trees[2], trees[1]])
    first = slice(*requests[2][:2])
    swapped = VarlenAttentionPlan(reordered, query.device, version).forward(
        *(torch.cat([source[first], source[start:end]]) for source in (query, key, value)), 128**-0.5
    )
    assert torch.equal(swapped[trees[2].length :], joint[start:end])


def test_invariant_audio_convolution_matches_conv2d_and_ignores_batch():
    from vllm_omni.model_executor.models.spev.audio import InvariantConv2d

    torch.manual_seed(9)
    source = torch.nn.Conv2d(1, 480, 3, 2, padding=1).cuda().to(torch.bfloat16)
    conv = InvariantConv2d.from_conv(source)
    chunks = torch.randn(9, 1, 128, 100, device="cuda", dtype=torch.bfloat16)
    joint = conv(chunks)
    torch.testing.assert_close(joint.float(), source(chunks).float(), atol=5e-2, rtol=2e-2)
    assert torch.equal(conv(chunks[2:3]), joint[2:3])


@pytest.mark.parametrize("version", flash_attention_versions())
@pytest.mark.parametrize("window", ["int", "tensor", "none"])
def test_invariant_encoder_attention_ignores_other_sequences(version, window):
    from vllm_omni.model_executor.models.spev.audio import InvariantEncoderAttention

    require_flash_attention(version)
    torch.manual_seed(10)
    attention = InvariantEncoderAttention(SimpleNamespace(scale=64**-0.5))
    attention.version = version
    bounds = torch.tensor([0, 104, 208, 260], device="cuda", dtype=torch.int32)
    longest = {"int": 104, "tensor": (bounds[1:] - bounds[:-1]).max(), "none": None}[window]
    query, key, value = (torch.randn(1, 260, 20, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    joint = attention(query, key, value, bounds, longest)
    alone = attention(
        query[:, 208:], key[:, 208:], value[:, 208:], torch.tensor([0, 52], device="cuda", dtype=torch.int32), 52
    )
    assert torch.equal(alone, joint[208:])


@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("shape", ["heads", "hidden"])
def test_rms_norm_kernel_matches_checkpoint_rounding_and_ignores_batch(with_residual, shape):
    from vllm_omni.model_executor.models.spev.normalization import rms_norm

    torch.manual_seed(11)
    if shape == "heads":
        weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        packed = torch.randn(300, 32 * 128 + 512, device="cuda", dtype=torch.bfloat16)
        value = packed[:, : 32 * 128].view(300, 32, 128)
    else:
        weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
        value = torch.randn(300, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(value) if with_residual else None
    joint = rms_norm(value, weight, 1e-6, residual)
    combined = value + residual if with_residual else value
    expected = combined.float()
    expected = weight * (expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)).to(value.dtype)
    alone = rms_norm(value[5:6], weight, 1e-6, residual[5:6] if with_residual else None)
    if with_residual:
        (joint, carried), (alone, _) = joint, alone
        assert torch.equal(carried, combined)
    torch.testing.assert_close(joint.float(), expected.float(), atol=2e-2, rtol=1e-2)
    assert torch.equal(alone, joint[5:6])


@pytest.fixture(scope="module")
def single_gpu_parallel_state():
    import socket

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(1, 0, f"tcp://127.0.0.1:{port}", 0)
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()


def test_audio_tower_encodes_each_recording_as_if_alone(single_gpu_parallel_state):
    from transformers import Qwen3OmniMoeAudioEncoderConfig

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeAudioEncoder,
        _get_feat_extract_output_lengths,
    )
    from vllm_omni.model_executor.models.spev.audio import encode_audio, install_invariant_audio

    torch.manual_seed(12)
    config = Qwen3OmniMoeAudioEncoderConfig(
        encoder_layers=2, d_model=256, encoder_attention_heads=4, encoder_ffn_dim=512, output_dim=256
    )
    tower = Qwen3OmniMoeAudioEncoder(config).to("cuda", torch.bfloat16).eval()
    with torch.no_grad():
        for parameter in tower.parameters():
            parameter.normal_(0, 0.05)
    lengths = torch.tensor([80, 300, 150, 80, 230], device="cuda")
    after = _get_feat_extract_output_lengths(lengths)
    features = torch.randn(config.num_mel_bins, int(lengths.sum()), device="cuda", dtype=torch.bfloat16)
    recordings = features.split(lengths.tolist(), dim=-1)
    with torch.inference_mode():
        reference = [
            tower._forward_same_padding(r, lengths[i : i + 1], after[i : i + 1]) for i, r in enumerate(recordings)
        ]
        install_invariant_audio(tower)
        joint = encode_audio(tower, features, lengths, after)
        alone = [encode_audio(tower, r, lengths[i : i + 1], after[i : i + 1])[0] for i, r in enumerate(recordings)]
    for batched, single, expected in zip(joint, alone, reference):
        assert torch.equal(batched, single)
        assert (batched.float() - expected.float()).abs().max() <= 0.05 * expected.float().abs().max()
