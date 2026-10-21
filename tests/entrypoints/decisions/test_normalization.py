# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextRMSNorm

from vllm_omni.model_executor.models.spev.normalization import CheckpointRMSNorm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_checkpoint_rounding_and_residual(dtype, with_residual):
    torch.manual_seed(78)
    reference = Qwen3OmniMoeThinkerTextRMSNorm(128).to(dtype)
    with torch.no_grad():
        reference.weight.copy_(torch.randn(128, dtype=dtype))
    model = CheckpointRMSNorm(reference)
    assert model.weight is reference.weight
    value = torch.randn(7, 128, dtype=dtype)
    residual = torch.randn_like(value) if with_residual else None
    combined = value + residual if with_residual else value
    expected = reference(combined)
    actual = model(value, residual)
    if with_residual:
        actual, carried = actual
        torch.testing.assert_close(carried, combined, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
