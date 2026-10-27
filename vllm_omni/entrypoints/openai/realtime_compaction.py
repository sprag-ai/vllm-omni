# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compaction of a realtime conversation: the policy that keeps its prompt under the context cap."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from vllm_omni.entrypoints.openai.realtime_conversation import ConversationItem


@dataclass(frozen=True)
class ContextBudget:
    """The prompt-token cap and what the conversation's prompt costs against it."""

    max_tokens: int
    """Prompt-token cap of the conversation."""
    overhead_tokens: int
    """Prompt tokens no item accounts for: the system prompt, the tools and the generation prompt."""
    item_tokens: Mapping[str, int]
    """Expanded prompt tokens of each item by item id; a message rendered from several items counts under its first."""

    def tokens(self, items: Sequence[ConversationItem]) -> int:
        """Return the expanded prompt length of a conversation made of ``items``, as far as their counts are known."""
        return self.overhead_tokens + sum(self.item_tokens.get(item.item_id, 0) for item in items)


class Compactor(Protocol):
    """A policy that decides which items a conversation generates over."""

    async def compact(self, items: Sequence[ConversationItem], budget: ContextBudget) -> list[ConversationItem]:
        """Return the items to keep generating over, in order, before each generate of a turn.

        Items left out are dropped from the conversation and new items are added to it; item ids must stay unique.
        The newest user message must be kept.
        """
        ...


@dataclass(frozen=True)
class DropOldest:
    """Drops the oldest items once the prompt exceeds the cap, down to a share of the cap."""

    retention: float
    """Share of the cap a conversation is cut back to once it exceeds it."""

    async def compact(self, items: Sequence[ConversationItem], budget: ContextBudget) -> list[ConversationItem]:
        """Drop whole runs from the start, each ending before a user message, never the newest user message."""
        kept = list(items)
        if budget.tokens(kept) <= budget.max_tokens:
            return kept
        target = int(budget.max_tokens * self.retention)
        last_user = max((i for i, item in enumerate(kept) if item.role == "user"), default=0)
        while last_user > 0 and budget.tokens(kept) > target:
            cut = 1
            while cut < last_user and kept[cut].role != "user":
                cut += 1
            kept = kept[cut:]
            last_user -= cut
        return kept
