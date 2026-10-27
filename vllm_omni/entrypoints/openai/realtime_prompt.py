# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token ids of a realtime conversation's chat prompt, joined from pieces that are each tokenized once."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.realtime_conversation import Conversation, RenderedConversation, RenderedUnit

logger = init_logger(__name__)

RenderChat = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None, bool], str]
"""Renders chat messages and tools through the chat template, with or without the generation prompt."""

EncodeTexts = Callable[[list[str]], Awaitable[list[list[int]]]]
"""Tokenizes texts, special tokens included, without adding any of its own."""

_BEFORE = {"role": "user", "content": "x"}
_AFTER = {"role": "user", "content": "y"}

_CHECK_TOOL_CALL = {"type": "function", "function": {"name": "lookup", "arguments": {"query": "Austin weather"}}}


class _NotJoinableError(Exception):
    """The chat template renders a message differently depending on what surrounds it."""


@dataclass
class ChatPrompt:
    """A conversation's prompt token ids, with the audio its placeholders stand for and its cost per item."""

    token_ids: list[int]
    """Prompt ids with one placeholder per audio item, before the audio is expanded."""
    audio: list[np.ndarray]
    audio_item_ids: list[str]
    audio_instance_ids: list[str]
    overhead_tokens: int
    """Prompt tokens no item accounts for: the system prompt, the tools and the generation prompt."""
    item_tokens: dict[str, int]
    """Expanded prompt tokens of each item by item id; a message rendered from several items counts under its first."""


@dataclass(frozen=True)
class _Frame:
    """The template text around a conversation's messages, for one system prompt and tool list."""

    joinable: bool
    """Whether the template renders this conversation as its pieces joined, verified on a sample conversation."""
    header: list[int]
    generation: list[int]
    close: list[int] | None
    """Ids that end an assistant message after its content; None when generated ids cannot be framed."""


class ChatPromptBuilder:
    """Builds a conversation's prompt ids by joining per-message pieces, each tokenized once.

    A template whose pieces fail the join self-check gets a full render instead. Each builder serves one conversation
    at a time.
    """

    def __init__(self, render: RenderChat, encode: EncodeTexts, audio_placeholder: str) -> None:
        self._render = render
        self._encode = encode
        self._placeholder = audio_placeholder
        self._frames: dict[str, _Frame] = {}
        self._pieces: dict[tuple[tuple[str, int], ...], list[int]] = {}
        self._full_counts: dict[tuple[tuple[str, int], ...], int] = {}
        self._before: str | None = None
        self._after: str | None = None

    async def build(
        self, conversation: Conversation, instructions: str | None, tools: list[dict[str, Any]] | None
    ) -> ChatPrompt:
        """Return the prompt that generates the next assistant reply over ``conversation``."""
        rendered = conversation.render(self._placeholder, instructions)
        frame = await self._frame(rendered.system, tools)
        if frame.joinable:
            try:
                return await self._joined(rendered, frame)
            except _NotJoinableError:
                logger.warning_once("chat template renders a realtime message by its context; using full renders")
        return await self._full(rendered, tools)

    async def _joined(self, rendered: RenderedConversation, frame: _Frame) -> ChatPrompt:
        """Join the frame and the cached pieces, tokenizing only the pieces not seen before."""
        units = rendered.units
        tail = units[-1] if units and units[-1].messages[0]["role"] == "assistant" else None
        body = units[:-1] if tail is not None else units
        keys = [_unit_key(unit) for unit in body]
        texts: dict[tuple[tuple[str, int], ...], str] = {}
        for unit, key in zip(body, keys):
            if key in self._pieces or key in texts:
                continue
            generated = _generated_ids(unit, frame)
            if generated is not None:
                self._pieces[key] = frame.generation + generated + (frame.close or [])
            else:
                texts[key] = self._between(unit.messages)
        tail_text = self._lead(tail.messages, generation=True) if tail is not None else None
        new_texts = list(texts.values()) + ([tail_text] if tail_text is not None else [])
        encoded = await self._encode(new_texts) if new_texts else []
        self._pieces.update(zip(texts, encoded))
        self._pieces = {key: self._pieces[key] for key in keys}
        tail_ids = encoded[-1] if tail_text is not None else frame.generation

        token_ids = list(frame.header)
        item_tokens: dict[str, int] = {}
        for unit, key in zip(body, keys):
            token_ids += self._pieces[key]
            _count(item_tokens, unit, len(self._pieces[key]))
        token_ids += tail_ids
        if tail is not None:
            _count(item_tokens, tail, len(tail_ids) - len(frame.generation))
        return _prompt(rendered, token_ids, len(frame.header) + len(frame.generation), item_tokens)

    async def _full(self, rendered: RenderedConversation, tools: list[dict[str, Any]] | None) -> ChatPrompt:
        """Tokenize the whole rendered conversation, counting each message on its own for the compactor."""
        keys = [_unit_key(unit) for unit in rendered.units]
        texts = {key: self._alone(unit) for unit, key in zip(rendered.units, keys) if key not in self._full_counts}
        full_text = self._render(rendered.messages, tools, True) if rendered.messages else ""
        encoded = await self._encode([full_text, *texts.values()])
        self._full_counts.update((key, len(ids)) for key, ids in zip(texts, encoded[1:]))
        self._full_counts = {key: self._full_counts[key] for key in keys}
        item_tokens: dict[str, int] = {}
        for unit, key in zip(rendered.units, keys):
            _count(item_tokens, unit, self._full_counts[key])
        overhead = max(len(encoded[0]) - sum(self._full_counts.values()), 0)
        return _prompt(rendered, encoded[0], overhead, item_tokens)

    async def _frame(self, system: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> _Frame:
        """Return the frame for this system prompt and tool list, checking once whether its pieces join."""
        cache_key = json.dumps([system, tools], sort_keys=True, default=str)
        frame = self._frames.get(cache_key)
        if frame is not None:
            return frame
        try:
            frame = await self._checked_frame(system, tools)
        except _NotJoinableError:
            frame = _Frame(joinable=False, header=[], generation=[], close=None)
        if not frame.joinable:
            logger.warning_once("chat template does not render realtime messages as joined pieces; using full renders")
        self._frames[cache_key] = frame
        return frame

    async def _checked_frame(self, system: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> _Frame:
        """Derive the frame and verify, on a sample conversation, that joined pieces equal the full render."""
        first = self._between([_BEFORE])
        opening = self._render(system + [_BEFORE], tools, False)
        if not opening.endswith(first):
            raise _NotJoinableError
        header = opening[: len(opening) - len(first)]
        generation = self._lead([], generation=True)
        empty_reply = self._between([{"role": "assistant", "content": ""}])
        close = empty_reply[len(generation) :] if empty_reply.startswith(generation) else None

        sample = [
            {"role": "user", "content": self._placeholder + "What's the weather?"},
            {"role": "assistant", "content": "Let me check.", "tool_calls": [_CHECK_TOOL_CALL]},
            {"role": "tool", "content": "sunny"},
            {"role": "tool", "content": "warm"},
            {"role": "assistant", "content": "Sunny and warm."},
            {"role": "user", "content": " Thanks!\n"},
        ]
        units = [[sample[0]], [sample[1]], sample[2:4], [sample[4]], [sample[5]]]
        reply = {"role": "assistant", "content": "You're welcome."}
        pieces = [self._between(unit) for unit in units]
        texts = [
            header,
            generation,
            close or "",
            self._lead([reply], generation=True),
            *pieces,
            self._render(system + sample, tools, True),
            self._render(system + sample + [reply], tools, True),
        ]
        encoded = await self._encode(texts)
        header_ids, generation_ids, close_ids, tail_ids = encoded[:4]
        piece_ids = encoded[4 : 4 + len(pieces)]
        full_ids, full_tail_ids = encoded[4 + len(pieces) :]
        joined = header_ids + [i for ids in piece_ids for i in ids]
        joinable = (
            header + "".join(pieces) + generation == texts[-2]
            and header + "".join(pieces) + texts[3] == texts[-1]
            and joined + generation_ids == full_ids
            and joined + tail_ids == full_tail_ids
        )
        return _Frame(joinable, header_ids, generation_ids, close_ids if close is not None else None)

    def _between(self, messages: list[dict[str, Any]]) -> str:
        """Return the text the template renders for ``messages`` between two user messages."""
        before, after = self._sentinels()
        whole = self._render([_BEFORE, *messages, _AFTER], None, False)
        if not whole.startswith(before) or not whole.endswith(after) or len(whole) < len(before) + len(after):
            raise _NotJoinableError
        return whole[len(before) : len(whole) - len(after)]

    def _lead(self, messages: list[dict[str, Any]], generation: bool = False) -> str:
        """Return the text the template renders for ``messages`` after a user message, ending the conversation."""
        before, _ = self._sentinels()
        whole = self._render([_BEFORE, *messages], None, generation)
        if not whole.startswith(before):
            raise _NotJoinableError
        return whole[len(before) :]

    def _alone(self, unit: RenderedUnit) -> str:
        """Return a unit's text after a user message, or rendered on its own when the template will not lead it."""
        try:
            return self._lead(unit.messages)
        except _NotJoinableError:
            return self._render(unit.messages, None, False)

    def _sentinels(self) -> tuple[str, str]:
        """Return the rendered text of the user message before a piece, and of the one after it."""
        if self._before is None or self._after is None:
            before = self._render([_BEFORE], None, False)
            both = self._render([_BEFORE, _AFTER], None, False)
            if not both.startswith(before):
                raise _NotJoinableError
            self._before, self._after = before, both[len(before) :]
        return self._before, self._after


def _unit_key(unit: RenderedUnit) -> tuple[tuple[str, int], ...]:
    """Return the cache key of a unit's piece: which item instances it holds, at which revisions."""
    return tuple((item.instance_id, item.revision) for item in unit.items)


def _generated_ids(unit: RenderedUnit, frame: _Frame) -> list[int] | None:
    """Return a reply's generated ids without their end token, or None when the unit must be rendered."""
    if frame.close is None or len(unit.items) != 1:
        return None
    ids = unit.items[0].token_ids
    if ids is None or unit.items[0].kind != "message":
        return None
    if ids and frame.close and ids[-1] == frame.close[0]:
        ids = ids[:-1]
    return list(ids)


def _count(item_tokens: dict[str, int], unit: RenderedUnit, piece_tokens: int) -> None:
    """Record a unit's expanded prompt tokens under its first item, and zero under the rest."""
    audio = sum(item.estimated_audio_tokens() - 1 for item in unit.items if item.audio is not None)
    item_tokens[unit.items[0].item_id] = piece_tokens + audio
    for item in unit.items[1:]:
        item_tokens[item.item_id] = 0


def _prompt(
    rendered: RenderedConversation, token_ids: list[int], overhead: int, item_tokens: dict[str, int]
) -> ChatPrompt:
    return ChatPrompt(
        token_ids=token_ids,
        audio=rendered.audio,
        audio_item_ids=rendered.audio_item_ids,
        audio_instance_ids=rendered.audio_instance_ids,
        overhead_tokens=overhead,
        item_tokens=item_tokens,
    )
