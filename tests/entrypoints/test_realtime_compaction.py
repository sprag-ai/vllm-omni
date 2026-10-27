# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compaction of a realtime conversation against the context cap."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime_compaction import ContextBudget, DropOldest
from vllm_omni.entrypoints.openai.realtime_conversation import ConversationItem

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _user(item_id: str) -> ConversationItem:
    return ConversationItem(item_id, "message", "user", audio=np.zeros(16_000, dtype=np.float32))


def _assistant(item_id: str) -> ConversationItem:
    return ConversationItem(item_id, "message", "assistant", text="ok", response_id=item_id)


def _budget(max_tokens: int, items: list[ConversationItem], each: int = 100, overhead: int = 10) -> ContextBudget:
    return ContextBudget(max_tokens, overhead, {item.item_id: each for item in items})


def _compact(retention: float, items: list[ConversationItem], budget: ContextBudget) -> list[str]:
    return [item.item_id for item in asyncio.run(DropOldest(retention).compact(items, budget))]


def test_the_budget_sums_the_overhead_and_the_items_it_knows() -> None:
    items = [_user("u1"), _assistant("a1")]

    assert ContextBudget(1000, 7, {"u1": 50}).tokens(items) == 57


def test_a_conversation_under_the_cap_is_untouched() -> None:
    items = [_user("u1"), _assistant("a1")]

    assert _compact(0.7, items, _budget(1000, items)) == ["u1", "a1"]


def test_over_the_cap_the_oldest_turns_go_down_to_the_retention_share() -> None:
    items = [item for turn in range(6) for item in (_user(f"u{turn}"), _assistant(f"a{turn}"))]

    kept = _compact(0.6, items, _budget(1000, items, overhead=0))

    assert kept == ["u3", "a3", "u4", "a4", "u5", "a5"]


def test_the_newest_user_message_is_never_dropped() -> None:
    items = [_assistant("a0"), _user("u1")]

    assert _compact(0.5, items, _budget(10, items)) == ["u1"]


def test_a_tool_call_is_dropped_with_its_output() -> None:
    items = [
        _user("u1"),
        ConversationItem("c1", "function_call", "assistant", call_id="call_1", name="f", response_id="r1"),
        ConversationItem("o1", "function_call_output", "tool", text="sunny", call_id="call_1"),
        _assistant("a1"),
        _user("u2"),
    ]

    assert _compact(0.9, items, _budget(200, items)) == ["u2"]
