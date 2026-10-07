# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch
from torch import nn

from vllm_omni.entrypoints.audio_decision.model import Qwen3OmniDecisionThinker, apply_qv_adapter


def test_fp32_adapter_keeps_k_and_bf16_output():
    torch.manual_seed(4)
    x = torch.randn(5, 8, dtype=torch.bfloat16)
    base = torch.randn(5, 12, dtype=torch.bfloat16)
    qa, qb = torch.randn(2, 8), torch.randn(8, 2)
    va, vb = torch.randn(2, 8), torch.randn(2, 2)
    actual, _ = apply_qv_adapter((base.clone(), None), x, qa, qb, va, vb, 2)
    expected = base.float()
    expected[:, :8] += (x.float() @ qa.T @ qb.T) * 2
    expected[:, 10:] += (x.float() @ va.T @ vb.T) * 2
    assert torch.equal(actual, expected.bfloat16())
    assert torch.equal(actual[:, 8:10], base[:, 8:10])


def test_accept_skips_layers_reject_continues_same_forward():
    model = Qwen3OmniDecisionThinker.__new__(Qwen3OmniDecisionThinker)
    nn.Module.__init__(model)
    calls = []

    def layer(pos, h, residual):
        calls.append(1)
        return h + 1, h.clone() if residual is None else residual

    decoder = SimpleNamespace(
        layers=[layer] * 3, embed_input_ids=lambda _: torch.zeros(2, 2), norm=lambda h, r: (h + r, None)
    )
    model.language_model = SimpleNamespace(model=decoder)
    model.decision_spec = {"depth": 1, "head_temperature": 1}
    model.decision_head = {
        "mean": torch.zeros(2, dtype=torch.float64),
        "scale": torch.ones(2, dtype=torch.float64),
        "weight": torch.zeros(2, 3, dtype=torch.float64),
        "bias": torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64),
    }
    model.decision_request = None
    model.arm_decision(0.95, "early")
    model.forward(torch.ones(2, dtype=torch.long), torch.arange(2))
    assert len(calls) == 1
    assert model.take_decision()["decoder_depth"] == 1
    calls.clear()
    model.arm_decision(1.0, "fallback")
    model.forward(torch.ones(2, dtype=torch.long), torch.arange(2))
    assert len(calls) == 3
    result = model.take_decision()
    assert result["used_full_decoder"] and result["decoder_depth"] == 3


def test_embedding_is_logical_unnormalized_residual():
    model = Qwen3OmniDecisionThinker.__new__(Qwen3OmniDecisionThinker)
    nn.Module.__init__(model)
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[lambda p, h, r: (h + 2, h + 3)],
            embed_input_ids=lambda _: torch.tensor([[1.0, 2.0], [4.0, 5.0]], dtype=torch.bfloat16),
        )
    )
    model.decision_spec = {"depth": 1, "head_temperature": 1}
    model.decision_head = {
        "mean": torch.zeros(2, dtype=torch.float64),
        "scale": torch.ones(2, dtype=torch.float64),
        "weight": torch.zeros(2, 3, dtype=torch.float64),
        "bias": torch.zeros(3, dtype=torch.float64),
    }
    model.decision_request = None
    model.arm_decision(0.95, "embedding", "embedding")
    model.forward(torch.ones(2, dtype=torch.long), torch.arange(2))
    result = model.take_decision()
    assert result["embedding"] == [13.0, 15.0]
    assert not result["used_full_decoder"]
