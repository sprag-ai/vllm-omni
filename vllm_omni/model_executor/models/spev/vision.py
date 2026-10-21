# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint vision features for native image and video inputs."""

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder,
    Qwen3OmniMoeVisionRotaryEmbedding,
)

from .attention_backend import checkpoint_attention


class CheckpointVisionEncoder(Qwen3OmniMoeVisionEncoder):
    def __init__(self, config):
        super().__init__(config)
        with torch.device("cpu"):
            rotary = Qwen3OmniMoeVisionRotaryEmbedding(config.hidden_size // config.num_heads // 2)
        self.rotary_pos_emb = rotary.to(self.device)

    def forward(self, hidden_states, grid_thw):
        grid_thw = torch.as_tensor(grid_thw, device=hidden_states.device)
        outputs, offset = [], 0
        with checkpoint_attention():
            for grid in grid_thw:
                length = int(grid.prod())
                output = super().forward(hidden_states[offset : offset + length], grid_thw=grid[None])
                outputs.append(torch.cat([output.pooler_output, *output.deepstack_features], dim=-1))
                offset += length
        if offset != len(hidden_states):
            raise ValueError("SPEV visual patches do not match their grids")
        return torch.cat(outputs)
