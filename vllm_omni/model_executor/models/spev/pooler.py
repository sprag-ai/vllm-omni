# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read candidate anchors through vLLM's native pooling output."""

import torch
from torch import nn
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.v1.pool.metadata import PoolingMetadata


class OptionHead(nn.Module):
    def __init__(self, hidden_size: int, width: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.scorer(features.float()).squeeze(-1)


def anchor_positions(ids: list[int], marker: list[int]) -> list[int]:
    return [i + len(marker) - 1 for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker]


class CandidatePooler(Pooler):
    def __init__(self, head: nn.Module, marker_token_ids: list[int]):
        super().__init__()
        self.head = head
        self.marker_token_ids = marker_token_ids

    def get_supported_tasks(self):
        return {"plugin"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise ValueError("SPEV requires complete prefills")
        lengths = cursor.num_scheduled_tokens_cpu.tolist()
        if sum(lengths) != hidden_states.shape[0]:
            raise RuntimeError("SPEV pooling rows do not match request lengths")
        results = []
        offset = 0
        for ids, length in zip(pooling_metadata.get_prompt_token_ids_cpu(), lengths):
            positions = anchor_positions(ids.tolist(), self.marker_token_ids)
            anchors = [offset + p for p in positions]
            scores = self.head(hidden_states[anchors].float().cpu())
            if not torch.isfinite(scores).all():
                raise RuntimeError("Nonfinite SPEV head scores")
            results.append(scores)
            offset += length
        return results
