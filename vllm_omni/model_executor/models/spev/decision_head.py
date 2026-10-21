# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Question-local evidence routing for SPEV decision checkpoints."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class EvidenceBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff: int):
        super().__init__()
        self.qnorm = nn.LayerNorm(width)
        self.mnorm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.fnorm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, query, memory, memory_pad):
        memory = self.mnorm(memory)
        value, _ = self.attn(self.qnorm(query), memory, memory, key_padding_mask=memory_pad, need_weights=False)
        query = query + value
        return query + self.ff(self.fnorm(query))


class IsolatedDecisionHead(nn.Module):
    def __init__(self, hidden: int, width: int, heads: int, ff: int, depth: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.memory = nn.Linear(hidden, width, bias=False)
        self.qproj = nn.Linear(hidden, width, bias=False)
        self.oqproj = nn.Linear(hidden, width, bias=False)
        self.globalproj = nn.Linear(hidden, width, bias=False)
        self.context = nn.Linear(hidden, width, bias=False)
        self.lexical = nn.Linear(hidden, width, bias=False)
        self.types = nn.Embedding(3, width)
        self.evidence = nn.ModuleList([EvidenceBlock(width, heads, ff) for _ in range(depth)])
        self.summarynorm = nn.LayerNorm(width)
        self.joint = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    width, heads, ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
                )
                for _ in range(depth)
            ]
        )
        self.fnorm = nn.LayerNorm(width)
        self.onorm = nn.LayerNorm(width)
        self.scorer = nn.Sequential(nn.Linear(width * 4, width), nn.GELU(), nn.Linear(width, 1))
        self.prior_scale = nn.Parameter(torch.zeros(()))
        self.joint_scale = nn.Parameter(torch.zeros(()))
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = self.memory(self.norm(batch["memory"]))
        memory = nn.utils.rnn.pad_sequence(projected.split(batch["memory_lengths"].tolist()), batch_first=True)
        question = self.norm(batch["questions"])
        context = self.norm(batch["options"])
        global_feature = self.norm(batch["global"])
        options = self.context(context) + self.lexical(batch["lexical"]) + self.oqproj(question)
        for block in self.evidence:
            options = block(options, memory, batch["memory_pad"])
        fields = self.qproj(question)
        membership = ~batch["option_pad"][:, None, :]
        weights = torch.einsum("bqd,bcd->bqc", fields, options) / memory.shape[-1] ** 0.5
        weights = weights.masked_fill(~membership, -1e9).softmax(-1) * membership
        summary = torch.einsum("bqc,bcd->bqd", weights, options)
        fields = fields + self.summarynorm(summary) + self.globalproj(global_feature)[:, None, :]
        fields = fields + self.types(batch["types"])
        for block in self.joint:
            fields = block(fields, memory, memory_key_padding_mask=batch["memory_pad"])
        fields = self.fnorm(fields).expand(-1, options.shape[1], -1)
        options = self.onorm(options)
        prior = (
            F.normalize(question + global_feature[:, None, :], dim=-1) * F.normalize(batch["lexical"], dim=-1)
        ).sum(-1) * self.prior_scale.clamp(max=4.605170186).exp()
        joint = F.cosine_similarity(fields, options, dim=-1) * self.joint_scale.clamp(max=4.605170186).exp()
        residual = self.scorer(torch.cat([fields, options, fields * options, (fields - options).abs()], -1)).squeeze(-1)
        return prior + self.gate.sigmoid() * (joint + residual)


@dataclass(frozen=True)
class QuestionFeatures:
    memory: torch.Tensor
    question: torch.Tensor
    options: torch.Tensor
    lexical: torch.Tensor
    global_feature: torch.Tensor
    question_type: int


def score_questions(head: IsolatedDecisionHead, records: list[QuestionFeatures]) -> list[torch.Tensor]:
    """Score all questions in a request batch with one head invocation."""
    if not records:
        raise ValueError("At least one question is required")
    device = head.norm.weight.device
    hidden = head.norm.normalized_shape[0]
    memories = max(len(record.memory) for record in records)
    candidates = max(len(record.options) for record in records)
    batch = {
        name: torch.zeros(shape, device=device, dtype=torch.float32)
        for name, shape in {
            "questions": (len(records), 1, hidden),
            "options": (len(records), candidates, hidden),
            "lexical": (len(records), candidates, hidden),
            "global": (len(records), hidden),
        }.items()
    }
    batch["memory"] = torch.cat([record.memory.to(device=device, dtype=torch.float32) for record in records])
    batch["memory_lengths"] = torch.tensor([len(record.memory) for record in records], device="cpu")
    batch["memory_pad"] = torch.ones(len(records), memories, device=device, dtype=torch.bool)
    batch["option_pad"] = torch.ones(len(records), candidates, device=device, dtype=torch.bool)
    batch["types"] = torch.tensor([[record.question_type] for record in records], device=device)
    for index, record in enumerate(records):
        if not len(record.memory) or not len(record.options) or record.question_type not in (0, 1, 2):
            raise ValueError("Invalid question features")
        if record.lexical.shape != record.options.shape:
            raise ValueError("Criterion embeddings must match candidate features")
        for name, value in (("options", record.options), ("lexical", record.lexical)):
            batch[name][index, : len(value)] = value.to(device=device, dtype=torch.float32)
        batch["questions"][index, 0] = record.question.to(device=device, dtype=torch.float32)
        batch["global"][index] = record.global_feature.to(device=device, dtype=torch.float32)
        batch["memory_pad"][index, : len(record.memory)] = False
        batch["option_pad"][index, : len(record.options)] = False
    values = head(batch)
    return [values[index, : len(record.options)] for index, record in enumerate(records)]
