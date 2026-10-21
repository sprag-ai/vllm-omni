# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Read candidate anchors through vLLM's native pooling output."""

import torch
from torch import nn
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.v1.pool.metadata import PoolingMetadata

from .attention_backend import checkpoint_attention
from .decision_head import QuestionFeatures, score_questions
from .decision_layout import DecisionLayout


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


def decision_layouts(metadata: PoolingMetadata) -> list[DecisionLayout]:
    layouts = []
    for length, params in zip(metadata.prompt_lens.tolist(), metadata.pooling_params):
        if params.extra_kwargs is None or "spev_layout" not in params.extra_kwargs:
            raise ValueError("SPEV requires processor-generated decision layout metadata")
        layout = DecisionLayout.from_dict(params.extra_kwargs["spev_layout"]).rebase(length)
        layouts.append(layout)
    if len(layouts) != len(metadata.pooling_params):
        raise ValueError("Decision layout count does not match the request batch")
    return layouts


class IsolatedCandidatePooler(Pooler):
    def __init__(self, head, embed_tokens):
        super().__init__()
        self.head = head
        self.embed_tokens = embed_tokens

    def get_supported_tasks(self):
        return {"plugin"}

    def get_pooling_updates(self, task):
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata):
        cursor = pooling_metadata.get_pooling_cursor()
        if cursor.is_partial_prefill():
            raise ValueError("SPEV requires complete prefills")
        lengths = cursor.num_scheduled_tokens_cpu.tolist()
        if sum(lengths) != len(hidden_states):
            raise ValueError("SPEV pooling rows do not match the request batch")
        if pooling_metadata.is_dummy:
            records, offset = [], 0
            for length in lengths:
                hidden = hidden_states[offset : offset + length]
                records.append(QuestionFeatures(hidden, hidden[0], hidden[:1], hidden[:1], hidden[-1], 1))
                offset += length
            with checkpoint_attention():
                return score_questions(self.head, records)
        layouts = decision_layouts(pooling_metadata)
        records, request_counts, offset = [], [], 0
        for layout in layouts:
            tree = layout.tree
            hidden = hidden_states[offset : offset + tree.length]
            offset += tree.length
            lexical = [self.embed_tokens(torch.tensor(ids, device=hidden.device)).mean(0) for ids in layout.lexical_ids]
            starts = [tree.prefix_length]
            for length in tree.branch_lengths:
                starts.append(starts[-1] + length)
            for question, kind in enumerate(layout.question_types):
                indices = [i for i, owner in enumerate(layout.candidate_questions) if owner == question]
                records.append(
                    QuestionFeatures(
                        memory=torch.cat(
                            [hidden[: tree.prefix_length]] + [hidden[starts[i] : starts[i + 1]] for i in indices]
                        ),
                        question=hidden[list(layout.question_positions[question])].mean(0),
                        options=hidden[[layout.anchors[i] for i in indices]],
                        lexical=torch.stack([lexical[i] for i in indices]),
                        global_feature=hidden[tree.prefix_length - 1],
                        question_type=kind,
                    )
                )
            request_counts.append(len(layout.question_types))
        with checkpoint_attention():
            values = score_questions(self.head, records)
        results, offset = [], 0
        for layout, count in zip(layouts, request_counts):
            grouped = values[offset : offset + count]
            offset += count
            scores = torch.empty(len(layout.anchors), device=grouped[0].device, dtype=grouped[0].dtype)
            for question, value in enumerate(grouped):
                scores[[i for i, owner in enumerate(layout.candidate_questions) if owner == question]] = value
            if not torch.isfinite(scores).all():
                raise RuntimeError("Nonfinite SPEV head scores")
            results.append(scores)
        return results
