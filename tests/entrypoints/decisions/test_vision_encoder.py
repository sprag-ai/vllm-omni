# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn
from transformers import Qwen3OmniMoeVisionEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeVisionEncoder
from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.spev.checkpoint_ops import CheckpointDecoderLayer
from vllm_omni.model_executor.models.spev.vision import CheckpointVisionEncoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def config():
    result = Qwen3OmniMoeVisionEncoderConfig(
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=16,
        num_heads=4,
        depth=1,
        deepstack_visual_indexes=[0],
        num_position_embeddings=4,
        patch_size=16,
        spatial_merge_size=2,
    )
    result._attn_implementation = "sdpa"
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vision_loader_and_independent_media_match_checkpoint(dtype):
    torch.manual_seed(78)
    source = Qwen3OmniMoeVisionEncoder(config()).to(dtype).eval()
    model = CheckpointVisionEncoder(config()).to(dtype).eval()
    loaded = AutoWeightsLoader(model).load_weights(iter(source.state_dict().items()))
    assert set(dict(model.named_parameters())) <= loaded
    grids = torch.tensor([[1, 2, 2], [2, 2, 2]])
    pixels = torch.randn(12, 3 * 2 * 16 * 16, dtype=dtype)
    calls = []
    model.patch_embed.register_forward_hook(lambda *args: calls.append(1))
    expected = []
    with torch.no_grad():
        for start, end, grid in ((0, 4, grids[:1]), (4, 12, grids[1:])):
            output = source(pixels[start:end], grid_thw=grid)
            expected.append(torch.cat([output.pooler_output, *output.deepstack_features], dim=-1))
        actual = model(pixels, grids)
    assert len(calls) == 2
    torch.testing.assert_close(actual, torch.cat(expected), atol=0, rtol=0)
    with pytest.raises(ValueError, match="patches"):
        model(torch.cat([pixels, pixels[:1]]), grids)


def test_decoder_residual_completes_before_deepstack_rounding():
    class IdentityNorm(nn.Module):
        def forward(self, value, residual=None):
            return value if residual is None else (value, residual)

    class ZeroAttention(nn.Module):
        def forward(self, positions, hidden_states):
            return torch.zeros_like(hidden_states)

    class NegativeMLP(nn.Module):
        def forward(self, value):
            return torch.full_like(value, -12)

    source = object.__new__(Qwen3MoeDecoderLayer)
    nn.Module.__init__(source)
    source.hidden_size = 2
    source.self_attn, source.mlp = ZeroAttention(), NegativeMLP()
    source.input_layernorm = source.post_attention_layernorm = IdentityNorm()
    model = CheckpointDecoderLayer(source)
    value = torch.full((1, 2), 12, dtype=torch.bfloat16)
    vision = torch.full_like(value, 0.01)
    old, residual = source(None, value, None)
    new, zero = model(None, value, None)
    expected = (old + residual) + vision
    assert not torch.equal((old + vision) + residual, expected)
    torch.testing.assert_close((new + vision) + zero, expected, atol=0, rtol=0)
    assert dict(model.named_parameters()) == dict(source.named_parameters())
