# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token layout for an evidence prefix and independent decision candidates."""

import json
from dataclasses import dataclass

from .candidate_tree import CandidateTree

MARKER = "\n<spev_score>\n"


def literal(value):
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def locate_markers(ids, marker, expected):
    positions = [i + len(marker) - 1 for i in range(len(ids) - len(marker) + 1) if ids[i : i + len(marker)] == marker]
    if len(positions) != expected:
        raise ValueError("Scoring anchor count mismatch")
    return positions


@dataclass(frozen=True)
class DecisionLayout:
    tree: CandidateTree
    anchors: tuple[int, ...]
    question_positions: tuple[tuple[int, ...], ...]
    lexical_ids: tuple[tuple[int, ...], ...]
    candidate_questions: tuple[int, ...]
    question_types: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "prefix_length": self.tree.prefix_length,
            "branch_lengths": self.tree.branch_lengths,
            "anchors": self.anchors,
            "question_positions": self.question_positions,
            "lexical_ids": self.lexical_ids,
            "candidate_questions": self.candidate_questions,
            "question_types": self.question_types,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            CandidateTree(data["prefix_length"], tuple(data["branch_lengths"])),
            tuple(data["anchors"]),
            tuple(tuple(row) for row in data["question_positions"]),
            tuple(tuple(row) for row in data["lexical_ids"]),
            tuple(data["candidate_questions"]),
            tuple(data["question_types"]),
        )

    def rebase(self, length: int):
        shift = length - self.tree.length
        layout = DecisionLayout(
            CandidateTree(self.tree.prefix_length + shift, self.tree.branch_lengths),
            tuple(index + shift for index in self.anchors),
            tuple(tuple(index + shift for index in row) for row in self.question_positions),
            self.lexical_ids,
            self.candidate_questions,
            self.question_types,
        )
        layout.validate(length)
        return layout

    def validate(self, length: int):
        count = len(self.tree.branch_lengths)
        if self.tree.length != length or not (
            len(self.anchors) == len(self.lexical_ids) == len(self.candidate_questions) == count
        ):
            raise ValueError("Decision layout does not match the packed sequence")
        if not self.question_types or len(self.question_positions) != len(self.question_types):
            raise ValueError("Decision question metadata is incomplete")
        if any(kind not in (0, 1, 2) for kind in self.question_types):
            raise ValueError("Invalid decision question type")
        if set(self.candidate_questions) != set(range(len(self.question_types))):
            raise ValueError("Every question requires candidate branches")
        start = self.tree.prefix_length
        for index, size in enumerate(self.tree.branch_lengths):
            if not start <= self.anchors[index] < start + size or not self.lexical_ids[index]:
                raise ValueError("Invalid decision candidate metadata")
            start += size
        for index, positions in enumerate(self.question_positions):
            branch = self.candidate_questions.index(index)
            start = self.tree.prefix_length + sum(self.tree.branch_lengths[:branch])
            end = start + self.tree.branch_lengths[branch]
            if not positions or any(position < start or position >= end for position in positions):
                raise ValueError("Question positions must belong to its first candidate branch")


def pack_candidates(tokenizer, prefix: list[int], questions: dict) -> tuple[list[int], DecisionLayout]:
    ids = list(prefix)
    lengths, anchors, positions, lexical, owners, kinds = [], [], [], [], [], []
    marker = tokenizer.encode(MARKER.lstrip("\n"), add_special_tokens=False)
    for question_index, question in enumerate(questions.values()):
        kinds.append({"predicate": 0, "choice": 1, "score": 2}[question["type"]])
        for candidate_index, description in enumerate(question["criteria"].values()):
            text = literal({"question": question["instructions"], "criterion": description}) + MARKER
            encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            start = len(ids)
            branch = encoded["input_ids"]
            ids.extend(branch)
            lengths.append(len(branch))
            anchors.append(start + locate_markers(branch, marker, 1)[0])
            owners.append(question_index)
            lexical.append(tuple(tokenizer.encode(literal(description), add_special_tokens=False)))
            if candidate_index == 0:
                phrase = literal(question["instructions"])
                begin = len('{"question": ')
                end = begin + len(phrase)
                positions.append(
                    tuple(start + i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > begin)
                )
    layout = DecisionLayout(
        CandidateTree(len(prefix), tuple(lengths)),
        tuple(anchors),
        tuple(positions),
        tuple(lexical),
        tuple(owners),
        tuple(kinds),
    )
    layout.validate(len(ids))
    return ids, layout
