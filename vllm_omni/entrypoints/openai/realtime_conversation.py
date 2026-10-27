# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The conversation a realtime session keeps between turns, and its rendering into chat messages."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import numpy as np

AUDIO_TOKENS_PER_SECOND = 13.1
"""Expanded prompt tokens per second of user audio, measured on Qwen3-Omni; used until a render reports the count."""

ItemKind = Literal["message", "function_call", "function_call_output"]
Role = Literal["user", "assistant", "tool"]


@dataclass
class ConversationItem:
    """One item of a realtime conversation."""

    item_id: str
    kind: ItemKind
    role: Role
    text: str = ""
    audio: np.ndarray | None = None
    """User audio as float32 PCM at the model's sampling rate."""
    sampling_rate: int = 16_000
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    response_id: str | None = None
    """The response that produced an assistant or function-call item; items of one response render as one message."""
    audio_out_seconds: float = 0.0
    """Seconds of reply audio sent for an assistant item; the basis for truncation."""
    audio_tokens: int | None = None
    """Expanded prompt tokens of the user audio, as the last render reported them."""
    audio_uuid: str = field(default_factory=lambda: uuid4().hex)
    """Cache key of the user audio; unique per item instance, so a reused item id never matches stale features."""

    def estimated_audio_tokens(self) -> int:
        """Return the expanded prompt tokens of this item's audio, measured when known."""
        if self.audio is None:
            return 0
        if self.audio_tokens is not None:
            return self.audio_tokens
        return math.ceil(len(self.audio) / self.sampling_rate * AUDIO_TOKENS_PER_SECOND)


@dataclass
class RenderedConversation:
    """A conversation as chat messages plus the audio its placeholders stand for, in order."""

    messages: list[dict[str, Any]]
    audio: list[np.ndarray]
    audio_item_ids: list[str]
    audio_uuids: list[str]
    item_ids: list[str]


@dataclass
class Conversation:
    """The ordered items of one realtime session."""

    items: list[ConversationItem] = field(default_factory=list)

    def get(self, item_id: str) -> ConversationItem | None:
        """Return the item with ``item_id``, or None."""
        return next((item for item in self.items if item.item_id == item_id), None)

    def add(self, item: ConversationItem, previous_item_id: str | None = None) -> None:
        """Insert ``item`` after ``previous_item_id``, or append it.

        Raises:
            KeyError: ``previous_item_id`` names no item, or ``item.item_id`` is already present.
        """
        if self.get(item.item_id) is not None:
            raise KeyError(item.item_id)
        if previous_item_id is None:
            self.items.append(item)
            return
        for index, existing in enumerate(self.items):
            if existing.item_id == previous_item_id:
                self.items.insert(index + 1, item)
                return
        raise KeyError(previous_item_id)

    def delete(self, item_id: str) -> bool:
        """Remove the item with ``item_id``; return whether it existed."""
        before = len(self.items)
        self.items = [item for item in self.items if item.item_id != item_id]
        return len(self.items) != before

    def truncate(self, item_id: str, audio_end_ms: int) -> bool:
        """Cut an assistant item's text to the share of its audio that was played.

        The cut keeps the same fraction of characters as of audio, moved back to a word boundary.

        Returns:
            False when the item is missing or not an assistant message.
        """
        item = self.get(item_id)
        if item is None or item.kind != "message" or item.role != "assistant":
            return False
        played = max(audio_end_ms, 0) / 1000
        if item.audio_out_seconds <= 0 or played >= item.audio_out_seconds:
            return True
        cut = int(len(item.text) * played / item.audio_out_seconds)
        if 0 < cut < len(item.text) and not item.text[cut].isspace():
            cut = max(item.text.rfind(" ", 0, cut), 0)
        item.text = item.text[:cut].rstrip()
        item.audio_out_seconds = played
        return True

    def render(self, audio_placeholder: str, instructions: str | None = None) -> RenderedConversation:
        """Return the conversation as chat messages, merging the items of one response into one assistant message."""
        messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}] if instructions else []
        audio: list[np.ndarray] = []
        audio_item_ids: list[str] = []
        audio_uuids: list[str] = []
        open_response: str | None = None
        for item in self.items:
            if item.kind == "function_call_output":
                messages.append({"role": "tool", "content": item.text})
                open_response = None
            elif item.role == "user":
                content = item.text
                if item.audio is not None:
                    content = audio_placeholder + content
                    audio.append(item.audio)
                    audio_item_ids.append(item.item_id)
                    audio_uuids.append(item.audio_uuid)
                messages.append({"role": "user", "content": content})
                open_response = None
            else:
                if open_response is None or item.response_id != open_response:
                    messages.append({"role": "assistant", "content": ""})
                    open_response = item.response_id
                message = messages[-1]
                if item.kind == "function_call":
                    message.setdefault("tool_calls", []).append(
                        {"type": "function", "function": {"name": item.name, "arguments": _arguments(item.arguments)}}
                    )
                else:
                    message["content"] += item.text
        return RenderedConversation(messages, audio, audio_item_ids, audio_uuids, [item.item_id for item in self.items])

    def fit(self, max_tokens: int, retention: float, count_tokens: Callable[[], int]) -> list[str]:
        """Drop the oldest items until the conversation fits ``max_tokens``.

        Once over the cap it drops down to ``retention * max_tokens`` rather than to the cap, so the prompt prefix stays
        unchanged for several turns between drops. A dropped run always ends before a user message, and the newest
        user message is never dropped.

        Args:
            count_tokens: Returns the current conversation's expanded prompt length.

        Returns:
            The ids of the dropped items, oldest first.
        """
        if count_tokens() <= max_tokens:
            return []
        target = int(max_tokens * retention)
        last_user = max((i for i, item in enumerate(self.items) if item.role == "user"), default=0)
        dropped: list[str] = []
        while last_user > 0 and count_tokens() > target:
            cut = 1
            while cut < last_user and self.items[cut].role != "user":
                cut += 1
            dropped += [item.item_id for item in self.items[:cut]]
            self.items = self.items[cut:]
            last_user -= cut
        return dropped


def _arguments(raw: str | None) -> Any:
    """Return tool-call arguments as the mapping the chat template expects, or the raw string if it is not JSON."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return raw
