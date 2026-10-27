# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The conversation a realtime session keeps between turns, and its rendering into chat messages."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import numpy as np

AUDIO_TOKENS_PER_SECOND = 13.1
"""Expanded prompt tokens per second of user audio, measured on Qwen3-Omni; used until a render reports the count."""

SPEAKING_CHARS_PER_SECOND = 11.4
"""Reply characters voiced per second of audio, the median over Qwen3-Omni replies; used until a session has its own."""

_MIN_RATE_SAMPLE_CHARS = 20
"""Shortest reply a session's speaking rate is measured on; a word or two over padded audio reads far too slow."""

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
    audio_complete: bool = True
    """Whether the reply audio voices all of ``text``; False for a reply cancelled while its text ran ahead."""
    audio_tokens: int | None = None
    """Expanded prompt tokens of the user audio, as the last render reported them."""
    token_ids: list[int] | None = None
    """The ids the model generated for an assistant reply, as it generated them; None once the text is edited."""
    revision: int = 0
    """Count of edits to the item's content."""
    instance_id: str = field(default_factory=lambda: uuid4().hex)
    """Unique per item instance; keys its cached audio features and prompt tokens, so a reused item id never matches
    stale entries."""

    def estimated_audio_tokens(self) -> int:
        """Return the expanded prompt tokens of this item's audio, measured when known."""
        if self.audio is None:
            return 0
        if self.audio_tokens is not None:
            return self.audio_tokens
        return math.ceil(len(self.audio) / self.sampling_rate * AUDIO_TOKENS_PER_SECOND)


@dataclass
class RenderedUnit:
    """Consecutive items that render as one stretch of the chat template: a message, or a run of tool results."""

    items: list[ConversationItem]
    messages: list[dict[str, Any]]


@dataclass
class RenderedConversation:
    """A conversation as chat messages plus the audio its placeholders stand for, in order."""

    system: list[dict[str, Any]]
    """The system message, or nothing when the session has no instructions."""
    units: list[RenderedUnit]
    audio: list[np.ndarray]
    audio_item_ids: list[str]
    audio_instance_ids: list[str]

    @property
    def messages(self) -> list[dict[str, Any]]:
        """The whole conversation as chat messages."""
        return self.system + [message for unit in self.units for message in unit.messages]


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

    def speaking_rate(self) -> float:
        """Return the reply characters voiced per second of audio, measured on this conversation's complete replies."""
        rates = sorted(
            len(item.text) / item.audio_out_seconds
            for item in self.items
            if item.kind == "message"
            and item.role == "assistant"
            and item.audio_complete
            and item.audio_out_seconds >= 1.0
            and len(item.text) >= _MIN_RATE_SAMPLE_CHARS
        )
        return rates[len(rates) // 2] if rates else SPEAKING_CHARS_PER_SECOND

    def truncate(self, item_id: str, audio_end_ms: int) -> bool:
        """Cut an assistant item's text to what its played audio voiced, at a word boundary.

        A complete reply keeps the share of characters it played of its audio; a cancelled reply keeps the characters
        the played seconds voice at the conversation's speaking rate.

        Returns:
            False when the item is missing or not an assistant message.
        """
        item = self.get(item_id)
        if item is None or item.kind != "message" or item.role != "assistant":
            return False
        played = min(max(audio_end_ms, 0) / 1000, item.audio_out_seconds)
        if item.audio_complete:
            if item.audio_out_seconds <= 0 or played >= item.audio_out_seconds:
                return True
            cut = int(len(item.text) * played / item.audio_out_seconds)
        else:
            cut = min(int(played * self.speaking_rate()), len(item.text))
        if 0 < cut < len(item.text) and not item.text[cut].isspace():
            cut = max(item.text.rfind(" ", 0, cut), 0)
        text = item.text[:cut].rstrip()
        if text != item.text:
            item.text = text
            item.token_ids = None
            item.revision += 1
        item.audio_out_seconds = played
        item.audio_complete = True
        return True

    def render(self, audio_placeholder: str, instructions: str | None = None) -> RenderedConversation:
        """Return the conversation as chat messages, merging the items of one response into one assistant message."""
        units: list[RenderedUnit] = []
        audio: list[np.ndarray] = []
        audio_item_ids: list[str] = []
        audio_instance_ids: list[str] = []
        open_response: str | None = None
        for item in self.items:
            if item.kind == "function_call_output":
                message = {"role": "tool", "content": item.text}
                if units and units[-1].items[-1].kind == "function_call_output":
                    units[-1].items.append(item)
                    units[-1].messages.append(message)
                else:
                    units.append(RenderedUnit([item], [message]))
                open_response = None
            elif item.role == "user":
                content = item.text
                if item.audio is not None:
                    content = audio_placeholder + content
                    audio.append(item.audio)
                    audio_item_ids.append(item.item_id)
                    audio_instance_ids.append(item.instance_id)
                units.append(RenderedUnit([item], [{"role": "user", "content": content}]))
                open_response = None
            else:
                if open_response is None or item.response_id != open_response:
                    units.append(RenderedUnit([], [{"role": "assistant", "content": ""}]))
                    open_response = item.response_id
                unit = units[-1]
                unit.items.append(item)
                message = unit.messages[0]
                if item.kind == "function_call":
                    message.setdefault("tool_calls", []).append(
                        {"type": "function", "function": {"name": item.name, "arguments": _arguments(item.arguments)}}
                    )
                else:
                    message["content"] += item.text
        system = [{"role": "system", "content": instructions}] if instructions else []
        return RenderedConversation(system, units, audio, audio_item_ids, audio_instance_ids)


def _arguments(raw: str | None) -> Any:
    """Return tool-call arguments as the mapping the chat template expects, or the raw string if it is not JSON."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return raw
