# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import uuid4

import numpy as np
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.openai.chat_completion.serving import _get_mm_token_counts
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.renderers.hf import safe_apply_chat_template
from vllm.renderers.inputs.preprocess import parse_model_prompt
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_conversation import Conversation, ConversationItem
from vllm_omni.entrypoints.openai.realtime_tool_calls import ToolCallDelta, ToolCallStreamState, extract_deltas
from vllm_omni.entrypoints.utils import coerce_param_message_types
from vllm_omni.model_executor.models.qwen3_omni.realtime_segments import segment_separator

logger = init_logger(__name__)

# How long to block on a client tool result before re-checking the connection.
# Only bounds the wait between liveness checks, not the total wait: a tool may
# legitimately take a long time, but a client that has gone away must not leave
# the generation task parked forever.
_TOOL_RESULT_POLL_S = 0.5

_SESSION_TYPES = frozenset({"realtime", "transcription"})

_DEFAULT_CONTEXT_TOKENS = 16_384
"""Prompt-token cap of a conversation session."""

_DEFAULT_CONTEXT_RETENTION = 0.7
"""Share of the cap a conversation is cut back to once it exceeds it."""


def _declared_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the entries of ``kwargs`` that ``fn`` declares, or all of them when it takes ``**kwargs``.

    A set value the function does not declare is dropped with a one-time warning.
    """
    parameters = inspect.signature(fn).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    declared = {name: value for name, value in kwargs.items() if name in parameters}
    dropped = sorted(name for name, value in kwargs.items() if name not in declared and value)
    if dropped:
        logger.warning_once("%s does not accept %s; ignoring them", fn.__qualname__, ", ".join(dropped))
    return declared


@dataclass
class _PendingToolCall:
    """A tool call being accumulated from the model's streamed output."""

    call_id: str
    name: str
    arguments: str = ""


@dataclass
class _Round:
    """What one generate of a conversation turn produced."""

    spoken: str = ""
    tool_calls: list[_PendingToolCall] = field(default_factory=list)
    sent_audio: bool = False
    audio_seconds: float = 0.0
    cancelled: bool = False


@dataclass
class _TurnUsage:
    """Token usage summed over the rounds of one conversation turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    multimodal_tokens: dict[str, int] = field(default_factory=dict)

    def add(self, prompt: int, completion: int, cached: int | None, multimodal: dict[str, int]) -> None:
        """Fold one round's counts into the turn."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cached_tokens += cached or 0
        for modality, count in multimodal.items():
            self.multimodal_tokens[modality] = self.multimodal_tokens.get(modality, 0) + count

    def info(self) -> UsageInfo:
        """Return the turn's totals without details."""
        return UsageInfo(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the turn's usage in the chat-completions shape."""
        return {
            **self.info().model_dump(exclude_none=True),
            "prompt_tokens_details": {
                "cached_tokens": self.cached_tokens,
                "multimodal_tokens": dict(self.multimodal_tokens),
            },
        }


def _new_item_id() -> str:
    """Return a fresh conversation item id."""
    return f"item_{uuid4().hex[:24]}"


def _item_id(item: dict) -> str:
    """Return a client item's `id`, or a fresh one when it carries none."""
    item_id = item.get("id")
    return item_id if isinstance(item_id, str) and item_id else _new_item_id()


def _context_limits() -> tuple[int, float]:
    """Return the conversation's prompt-token cap and retention fraction.

    Read from ``SPRAG_REALTIME_CONTEXT_TOKENS`` and ``SPRAG_REALTIME_CONTEXT_RETENTION``; an unparseable or
    out-of-range value falls back to its default.
    """
    try:
        cap = int(os.environ.get("SPRAG_REALTIME_CONTEXT_TOKENS", _DEFAULT_CONTEXT_TOKENS))
    except ValueError:
        cap = 0
    try:
        retention = float(os.environ.get("SPRAG_REALTIME_CONTEXT_RETENTION", _DEFAULT_CONTEXT_RETENTION))
    except ValueError:
        retention = -1.0
    if cap <= 0:
        logger.warning_once("invalid SPRAG_REALTIME_CONTEXT_TOKENS; using %d", _DEFAULT_CONTEXT_TOKENS)
        cap = _DEFAULT_CONTEXT_TOKENS
    if not 0 < retention <= 1:
        logger.warning_once("invalid SPRAG_REALTIME_CONTEXT_RETENTION; using %s", _DEFAULT_CONTEXT_RETENTION)
        retention = _DEFAULT_CONTEXT_RETENTION
    return cap, retention


class RealtimeConnection(VllmRealtimeConnection):
    """Omni realtime connection with audio-only server events, plus
    OpenAI-Realtime-shaped conversation history, tool/function calling and
    voice (TTS speaker) selection.

    Reuses upstream vLLM websocket/session lifecycle and customizes
    generation output handling to emit audio deltas and tool-call events.

    Conversation sessions keep their history for the life of the socket. Each
    user turn is one generate over the whole conversation, rendered through the
    chat template; past user audio stays decoded in the session and carries a
    session-scoped `multi_modal_uuids` entry, so the processor cache serves its
    features and the prefix cache serves its KV. Item ids are the client's.

    Conversation protocol (client -> server):
      - `input_audio_buffer.commit` with `final: false` may carry `item_id` (the
        user audio item) and `response_item_id` (the assistant reply item).
      - `conversation.item.create` adds a `message` (user or assistant text, or
        user `input_audio`) or a `function_call_output`, after
        `previous_item_id` when given.
      - `conversation.item.delete` removes an item; `conversation.item.truncate`
        cuts an assistant item to the `audio_end_ms` the user heard. A truncate
        naming the reply still being generated applies once that reply ends.
      - `response.create` starts a turn without new audio; `response.cancel`
        stops the running one and keeps the text generated so far.

    Each spoken reply item is announced by `response.output_item.added` with
    its `id` before its first text or audio. Each conversation turn ends with
    `response.done`, carrying the turn's usage
    in the chat-completions shape over the whole context (summed over tool
    rounds), the ids of the items it created, the ids in context, and the ids
    the context cap dropped. The cap is `SPRAG_REALTIME_CONTEXT_TOKENS`; once it
    is exceeded the oldest items are dropped down to
    `SPRAG_REALTIME_CONTEXT_RETENTION` of it.

    Tool calling mirrors OpenAI's Realtime API: `session.update` carries
    `tools`; the server streams `response.output_item.added`,
    `response.function_call_arguments.delta` and `.done`; the client answers
    each call with a `function_call_output` item, and the turn continues with
    the results in the conversation. Audio for a tool-call round is synthesized
    but not forwarded once the `<tool_call>` tag has been parsed. A chain of
    tool calls is bounded by MAX_TOOL_ROUNDS. Once the model has requested a
    tool, the turn blocks until a result arrives for every pending call; a
    client that disconnects releases the wait, and malformed or unknown results
    are reported back.

    Transcription sessions keep no history: each turn is decoded on its own,
    in segments when `SPRAG_REALTIME_TRANSCRIPTION_SEGMENT_S` is set.
    """

    # Upper bound on consecutive tool-call rounds within one user turn.
    MAX_TOOL_ROUNDS = 8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = cast(AsyncOmni, self.serving.engine_client)
        self._realtime_audio_ref: np.ndarray | None = None
        self._tools: list[dict[str, Any]] | None = None
        # parser-assigned index (per generation) -> the call being accumulated
        self._pending_tool_calls: dict[int, _PendingToolCall] = {}
        self._tool_result_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._awaiting_tool_results = False
        self._speaker: str | None = None
        self._instructions: str | None = None
        self._transcription = False
        self._conversation = Conversation()
        self._next_user_item_id: str | None = None
        self._next_response_item_id: str | None = None
        self._cancel_requested = False
        self._live_item_id: str | None = None
        self._live_truncate_ms: int | None = None

    async def handle_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "session.update":
            tools = event.get("tools")
            if tools is not None:
                self._tools = tools
            speaker = event.get("voice") or event.get("speaker")
            if speaker is not None:
                self._speaker = speaker
            instructions = event.get("instructions")
            if instructions is not None:
                self._instructions = instructions
            session_type = event.get("session_type")
            if session_type is not None and (not isinstance(session_type, str) or session_type not in _SESSION_TYPES):
                await self.send_error(
                    f"session_type must be one of {sorted(_SESSION_TYPES)}, got {session_type!r}",
                    "invalid_session_type",
                )
                return
            # The base refuses an update without a supported model; a refused update must not change the mode.
            model = event.get("model")
            sets_type = session_type is not None and model is not None and self._check_model(model) is None
            await super().handle_event(event)
            if sets_type:
                self._transcription = session_type == "transcription"
        elif event_type == "input_audio_buffer.commit":
            event = dict(event)
            # The base's commit model knows only `final`; the conversation ids are ours.
            user_item_id = event.pop("item_id", None)
            response_item_id = event.pop("response_item_id", None)
            if not event.get("final"):
                self._next_user_item_id = await self._free_item_id(user_item_id)
                self._next_response_item_id = await self._free_item_id(response_item_id, taken=self._next_user_item_id)
            await super().handle_event(event)
        elif event_type == "conversation.item.create":
            await self._create_item(event)
        elif event_type == "conversation.item.delete":
            item_id = event.get("item_id")
            if not isinstance(item_id, str) or not self._conversation.delete(item_id):
                await self.send_error(f"No conversation item {item_id!r}", "item_not_found")
        elif event_type == "conversation.item.truncate":
            item_id = event.get("item_id")
            audio_end_ms = event.get("audio_end_ms")
            if not isinstance(audio_end_ms, int) or isinstance(audio_end_ms, bool) or audio_end_ms < 0:
                await self.send_error(
                    "conversation.item.truncate requires a non-negative audio_end_ms", "invalid_value"
                )
            elif item_id is not None and item_id == self._live_item_id and self._conversation.get(item_id) is None:
                self._live_truncate_ms = audio_end_ms
            elif not isinstance(item_id, str) or not self._conversation.truncate(item_id, audio_end_ms):
                await self.send_error(f"No assistant message item {item_id!r}", "item_not_found")
        elif event_type == "response.create":
            if self._transcription:
                await self.send_error("response.create is not available on a transcription session", "invalid_event")
            elif self.generation_task is not None and not self.generation_task.done():
                await self.send_error("A response is already in progress", "response_in_progress")
            else:
                self._tool_rounds_reset()
                self.generation_task = asyncio.create_task(self._run_conversation_turn(None))
        elif event_type == "response.cancel":
            if self.generation_task is not None and not self.generation_task.done():
                self._cancel_requested = True
        else:
            await super().handle_event(event)

    async def _free_item_id(self, item_id: Any, taken: str | None = None) -> str | None:
        """Return a client-chosen id for a new item, or None after reporting an id that is already in use."""
        if not isinstance(item_id, str) or not item_id:
            return None
        if item_id == taken or self._conversation.get(item_id) is not None:
            await self.send_error(f"Conversation item {item_id!r} already exists; using a new id", "item_conflict")
            return None
        return item_id

    async def _create_item(self, event: dict) -> None:
        """Add a client item to the conversation, or hand a tool result to the turn waiting for it."""
        item = event.get("item") or {}
        item_type = item.get("type")
        if item_type == "function_call_output":
            if not await self._valid_tool_result(item):
                return
            if self._awaiting_tool_results:
                self._tool_result_queue.put_nowait(item)
                return
            new_item = ConversationItem(
                item_id=_item_id(item),
                kind="function_call_output",
                role="tool",
                text=item["output"],
                call_id=item["call_id"],
            )
        elif item_type == "message":
            new_item = await self._message_item(item)
            if new_item is None:
                return
        else:
            await self.send_error(f"Unsupported conversation.item type: {item_type!r}", "unsupported_item")
            return
        previous_item_id = event.get("previous_item_id")
        try:
            self._conversation.add(new_item, previous_item_id if isinstance(previous_item_id, str) else None)
        except KeyError as missing:
            await self.send_error(f"Conversation item {missing.args[0]!r} is unknown or taken", "item_conflict")

    async def _message_item(self, item: dict) -> ConversationItem | None:
        """Build a user or assistant message item from its content parts, or report why it is invalid."""
        role = item.get("role")
        if role not in ("user", "assistant"):
            await self.send_error(f"Unsupported message role: {role!r}", "unsupported_item")
            return None
        text = ""
        audio: list[np.ndarray] = []
        for part in item.get("content") or []:
            part_type = part.get("type") if isinstance(part, dict) else None
            if part_type in ("input_text", "text", "output_text") and isinstance(part.get("text"), str):
                text += part["text"]
            elif part_type == "input_audio" and role == "user" and isinstance(part.get("audio"), str):
                try:
                    pcm = base64.b64decode(part["audio"], validate=True)
                except ValueError:
                    await self.send_error("input_audio must be base64 PCM16", "invalid_audio")
                    return None
                audio.append(np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0)
            else:
                await self.send_error(f"Unsupported content part: {part_type!r}", "unsupported_item")
                return None
        return ConversationItem(
            item_id=_item_id(item),
            kind="message",
            role=role,
            text=text,
            audio=np.concatenate(audio) if audio else None,
            sampling_rate=self._sampling_rate(),
        )

    async def _valid_tool_result(self, item: dict) -> bool:
        """Report a malformed `function_call_output` to the client; return whether it is well formed."""
        call_id = item.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            await self.send_error(
                "function_call_output requires a non-empty string 'call_id'",
                "invalid_function_call_output",
            )
            return False
        output = item.get("output")
        if not isinstance(output, str):
            await self.send_error(
                f"function_call_output 'output' must be a string, got {type(output).__name__}",
                "invalid_function_call_output",
            )
            return False
        return True

    def _tool_rounds_reset(self) -> None:
        """Discard tool results left over from an earlier turn, which would otherwise answer this turn's calls."""
        while not self._tool_result_queue.empty():
            self._tool_result_queue.get_nowait()

    def _sampling_rate(self) -> int:
        """Return the model's audio input rate."""
        return int(cached_processor_from_config(self.serving.model_config).feature_extractor.sampling_rate)

    async def start_generation(self):
        if self.generation_task is not None and not self.generation_task.done():
            logger.warning("Generation already in progress, ignoring commit")
            return

        self._tool_rounds_reset()
        audio_stream = self.audio_stream_generator()
        if not self._transcription:
            self.generation_task = asyncio.create_task(self._run_conversation_turn(audio_stream))
            return
        input_stream: asyncio.Queue[list[int]] = asyncio.Queue()
        streaming_input_gen = self._buffer_realtime_audio(audio_stream, input_stream)
        self.generation_task = asyncio.create_task(self._run_generation(streaming_input_gen, input_stream))

    async def _render_prompt(self, prompt: PromptType) -> StreamingInput:
        model_config = self.serving.model_config
        parsed_prompt = parse_model_prompt(model_config, prompt)
        (engine_input,) = await self.serving.renderer.render_cmpl_async([parsed_prompt])
        # render_cmpl_async's internal pipeline (BaseRenderer.process_for_engine_async)
        # only carries over fields it explicitly knows about - additional_information
        # set on the pre-render prompt (buffer_realtime_audio's `speaker`) is silently
        # dropped unless reapplied to the *rendered* engine_input, exactly like
        # serving_chat.py._preprocess_chat does for /v1/chat/completions.
        additional_information = (
            parsed_prompt.get("additional_information") if isinstance(parsed_prompt, dict) else None
        )
        if additional_information:
            engine_input["additional_information"] = additional_information
        return StreamingInput(prompt=engine_input)

    async def _buffer_realtime_audio(
        self,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
    ) -> AsyncGenerator[StreamingInput, None]:
        """Equivalent to `OpenAIServingRealtime.transcribe_realtime`, but
        threads `self._speaker`/`self._instructions` through to the model's
        `buffer_realtime_audio`. The base class's `transcribe_realtime` has a
        fixed (audio_stream, input_stream, model_config) call signature with
        no seam for extra per-connection state like these, so this
        reimplements its (short) body directly rather than patching upstream
        vLLM."""
        buffer_realtime_audio = self.serving.model_cls.buffer_realtime_audio
        session_kwargs = _declared_kwargs(
            buffer_realtime_audio,
            {"speaker": self._speaker, "instructions": self._instructions, "transcription": True},
        )
        stream_input_iter = buffer_realtime_audio(
            audio_stream, input_stream, self.serving.model_config, **session_kwargs
        )
        async for prompt in stream_input_iter:
            yield await self._render_prompt(prompt)

    @staticmethod
    def _tensor_to_numpy(value) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value
        elif hasattr(value, "detach"):
            arr = value.detach().float().cpu().numpy()
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return None
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _numpy_audio_prefix_match(prev: np.ndarray, curr: np.ndarray) -> bool:
        n = prev.shape[0]
        if n == 0:
            return True
        if curr.shape[0] < n:
            return False
        return bool(np.allclose(curr[:n], prev, rtol=1e-3, atol=2e-4))

    def _raw_waveform_to_deltas(self, arr: np.ndarray) -> list[np.ndarray]:
        """Convert one streaming PCM f32 chunk into incremental piece(s) for the client.

        Some engine paths emit a growing cumulative waveform each step; others emit
        true per-step deltas. We support both without duplicating audio on the client.
        """
        if arr.size == 0:
            return []
        ref = self._realtime_audio_ref
        if ref is None:
            self._realtime_audio_ref = arr.copy()
            return [arr]
        if self._numpy_audio_prefix_match(ref, arr):
            delta = arr[ref.shape[0] :]
            self._realtime_audio_ref = arr.copy()
            return [delta] if delta.size > 0 else []
        # True per-step delta (not a prefix extension of what we have seen).
        self._realtime_audio_ref = np.concatenate([ref, arr])
        return [arr]

    def _extract_audio_chunks(self, output) -> tuple[list[np.ndarray], int]:
        mm = getattr(output, "multimodal_output", None)
        if mm is None:
            return [], 24000
        # Support both MultimodalPayload and plain dict
        if not isinstance(mm, Mapping):
            return [], 24000

        sr = mm.get("sr") or mm.get("sample_rate") or mm.get("audio_sample_rate") or 24000
        if isinstance(sr, (list, tuple)) and sr:
            sr = sr[-1]
        if hasattr(sr, "item"):
            sr = sr.item()
        sample_rate_hz = int(sr)
        key = "audio" if "audio" in mm else ("model_outputs" if "model_outputs" in mm else None)
        if key is None:
            return [], sample_rate_hz

        raw_audio = mm.get(key)
        chunks: list[np.ndarray] = []
        if isinstance(raw_audio, (list, tuple)):
            if len(raw_audio) > 0:
                arr = self._tensor_to_numpy(raw_audio[-1])
                if arr is not None and arr.size > 0:
                    chunks.extend(self._raw_waveform_to_deltas(arr))
        else:
            arr = self._tensor_to_numpy(raw_audio)
            if arr is not None and arr.size > 0:
                chunks.extend(self._raw_waveform_to_deltas(arr))
        return chunks, sample_rate_hz

    @staticmethod
    def _pcm16_b64(audio_f32: np.ndarray) -> str:
        clipped = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return base64.b64encode(pcm16.tobytes()).decode("utf-8")

    async def _emit_tool_call_deltas(self, tool_deltas: list[ToolCallDelta]) -> None:
        for delta in tool_deltas:
            if delta.name is not None:
                call = _PendingToolCall(call_id=f"call_{uuid4().hex[:24]}", name=delta.name)
                self._pending_tool_calls[delta.index] = call
                await self.send_json(
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "function_call", "name": call.name, "call_id": call.call_id},
                    }
                )
            if delta.arguments_delta:
                call = self._pending_tool_calls.get(delta.index)
                if call is None:
                    continue  # shouldn't happen: name delta always precedes argument deltas for the same index
                call.arguments += delta.arguments_delta
                await self.send_json(
                    {
                        "type": "response.function_call_arguments.delta",
                        "call_id": call.call_id,
                        "delta": delta.arguments_delta,
                    }
                )

    async def _run_generation(
        self,
        streaming_input_gen: AsyncGenerator,
        input_stream: asyncio.Queue[list[int]],
    ):
        """Decode one transcription turn, streaming text deltas and ending with `transcription.done`."""
        request_id = f"rt-{self.connection_id}-{uuid4()}"
        full_text = ""
        prompt_token_ids_len = 0
        completion_tokens_len = 0

        # Coerce cumulative outputs to delta outputs; this ensures
        # we don't emit redundant MM data & drain after emitting.
        sampling_params_list = coerce_param_message_types(
            list(self.engine.default_sampling_params_list),
            is_streaming=True,
        )

        segment_finished = False

        result_gen = None
        try:
            result_gen = self.engine.generate(
                prompt=streaming_input_gen,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["text"],
            )

            async for output in result_gen:
                if getattr(output, "stage_id", None) == 0 and output.outputs:
                    first_output = output.outputs[0]
                    new_token_ids = list(first_output.token_ids)
                    if new_token_ids:
                        input_stream.put_nowait(new_token_ids)
                    if output.prompt_token_ids:
                        prompt_token_ids_len = max(prompt_token_ids_len, len(output.prompt_token_ids))

                    delta_text = first_output.text or ""
                    if segment_finished and delta_text:
                        delta_text = segment_separator(full_text, delta_text) + delta_text
                        segment_finished = False
                    if first_output.finish_reason is not None:
                        segment_finished = True
                    full_text += delta_text
                    completion_tokens_len += len(new_token_ids)
                    if delta_text:
                        await self.send(TranscriptionDelta(delta=delta_text))

                if not self._is_connected:
                    break

            usage = UsageInfo(
                prompt_tokens=prompt_token_ids_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_token_ids_len + completion_tokens_len,
            )
            await self.send(TranscriptionDone(text=full_text, usage=usage))
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            # Close the generator explicitly so AsyncOmni.generate's cleanup
            # (input-pump cancellation and engine-side abort) runs now rather
            # than whenever the event loop garbage-collects the async
            # generator; the delay window is where a disconnected session
            # keeps cycling through the stages (issue #4271).
            if result_gen is not None:
                try:
                    await result_gen.aclose()
                except Exception:
                    logger.exception("Failed to close realtime result generator")
            if self._is_connected:
                try:
                    await self.send_json({"type": "response.output_audio.done", "has_audio": False})
                except Exception:
                    logger.exception("Failed to send response.output_audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    async def _run_conversation_turn(self, audio_stream: AsyncGenerator[np.ndarray, None] | None) -> None:
        """Run one conversation turn: add the user's audio, generate over the whole conversation, and record the reply.

        A turn without `audio_stream` (`response.create`) generates over the conversation as it stands.
        """
        user_item_id, self._next_user_item_id = self._next_user_item_id or _new_item_id(), None
        response_id, self._next_response_item_id = self._next_response_item_id or _new_item_id(), None
        self._cancel_requested = False
        usage = _TurnUsage()
        created: list[str] = []
        dropped: list[str] = []
        spoken = ""
        sent_audio = False
        status = "completed"
        try:
            if audio_stream is not None:
                chunks = [chunk async for chunk in audio_stream]
                if chunks:
                    self._conversation.add(
                        ConversationItem(
                            item_id=user_item_id,
                            kind="message",
                            role="user",
                            audio=np.concatenate(chunks),
                            sampling_rate=self._sampling_rate(),
                        )
                    )
                    created.append(user_item_id)
            for tool_round in range(self.MAX_TOOL_ROUNDS + 1):
                dropped += self._fit_context()
                message_item_id = response_id if tool_round == 0 else _new_item_id()
                self._live_item_id, self._live_truncate_ms = message_item_id, None
                result = await self._generate_round(usage, message_item_id)
                sent_audio |= result.sent_audio
                spoken += result.spoken
                created += self._record_round(result, message_item_id, response_id)
                self._live_item_id = None
                if result.cancelled:
                    status = "cancelled"
                    break
                if not result.tool_calls:
                    break
                for call in result.tool_calls:
                    await self.send_json(
                        {
                            "type": "response.function_call_arguments.done",
                            "call_id": call.call_id,
                            "arguments": call.arguments,
                        }
                    )
                if tool_round == self.MAX_TOOL_ROUNDS:
                    # A model that keeps re-emitting tool calls instead of answering
                    # would otherwise recurse without bound.
                    logger.warning("tool-call chain exceeded %d rounds; abandoning turn", self.MAX_TOOL_ROUNDS)
                    await self.send_error(
                        f"Tool-call chain exceeded {self.MAX_TOOL_ROUNDS} rounds without a final response",
                        "tool_call_loop",
                    )
                    status = "failed"
                    break
                results = await self._await_tool_results(result.tool_calls)
                if not self._is_connected:
                    return
                if results is None:
                    status = "cancelled"
                    break
                for call, output in results:
                    item_id = _new_item_id()
                    self._conversation.add(
                        ConversationItem(
                            item_id=item_id, kind="function_call_output", role="tool", text=output, call_id=call.call_id
                        )
                    )
                    created.append(item_id)

            await self.send(TranscriptionDone(text=spoken, usage=usage.info()))
            await self.send_json(
                {
                    "type": "response.done",
                    "response_id": response_id,
                    "status": status,
                    "item_ids": created,
                    "context_item_ids": [item.item_id for item in self._conversation.items],
                    "dropped_item_ids": dropped,
                    "usage": usage.as_dict(),
                }
            )
        except Exception as e:
            logger.exception("Error in generation: %s", e)
            await self.send_error(str(e), "processing_error")
        finally:
            self._live_item_id = None
            # Always send the terminal event so clients don't hang forever.
            if self._is_connected:
                try:
                    await self.send_json({"type": "response.output_audio.done", "has_audio": sent_audio})
                except Exception:
                    logger.exception("Failed to send response.output_audio.done")
            while not self.audio_queue.empty():
                self.audio_queue.get_nowait()

    def _chat_text(self, messages: list[dict[str, Any]]) -> str:
        """Render chat messages and the session's tools through the checkpoint's own chat template."""
        model_config = self.serving.model_config
        # Pass the processor's chat_template explicitly: with `tools` set,
        # safe_apply_chat_template's own resolution skips the processor, where
        # this checkpoint's template lives, and falls back to a generic template.
        return safe_apply_chat_template(
            model_config,
            cached_tokenizer_from_config(model_config),
            messages,
            tools=self._tools,
            chat_template=cached_processor_from_config(model_config).chat_template,
            add_generation_prompt=True,
            tokenize=False,
        )

    def _audio_placeholder(self) -> str:
        """Return the prompt text that stands for one audio item."""
        return self.serving.model_cls.realtime_audio_placeholder()

    def _fit_context(self) -> list[str]:
        """Drop the oldest conversation items over the context cap; return their ids."""
        max_tokens, retention = _context_limits()
        tokenizer = cached_tokenizer_from_config(self.serving.model_config)

        def count_tokens() -> int:
            rendered = self._conversation.render(self._audio_placeholder(), self._instructions)
            # Each audio placeholder expands from one pad token to the audio's token run.
            audio_tokens = sum(
                item.estimated_audio_tokens() - 1 for item in self._conversation.items if item.audio is not None
            )
            return len(tokenizer.encode(self._chat_text(rendered.messages))) + audio_tokens

        return self._conversation.fit(max_tokens, retention, count_tokens)

    async def _generate_round(self, usage: _TurnUsage, message_item_id: str) -> _Round:
        """Generate one reply over the current conversation, streaming its text, audio and tool-call events.

        Args:
            message_item_id: The id the round's spoken reply is announced and recorded under.
        """
        rendered = self._conversation.render(self._audio_placeholder(), self._instructions)
        tokenizer = cached_tokenizer_from_config(self.serving.model_config)
        prompt = TokensPrompt(prompt_token_ids=tokenizer.encode(self._chat_text(rendered.messages)))
        if rendered.audio:
            prompt["multi_modal_data"] = {"audio": rendered.audio}
            # Session-scoped: another session's identical id must never stand in for this audio.
            prompt["multi_modal_uuids"] = {"audio": [f"{self.connection_id}:{u}" for u in rendered.audio_uuids]}
        if self._speaker:
            prompt["additional_information"] = {"speaker": [self._speaker]}
        streaming_input = await self._render_prompt(prompt)
        engine_input = streaming_input.prompt
        placeholders = (engine_input.get("mm_placeholders") or {}).get("audio") or []
        for item_id, placeholder in zip(rendered.audio_item_ids, placeholders):
            item = self._conversation.get(item_id)
            if item is not None:
                item.audio_tokens = placeholder.length

        async def single_input():
            yield streaming_input

        result = _Round()
        announced = False

        async def announce() -> None:
            nonlocal announced
            if not announced:
                announced = True
                await self.send_json(
                    {
                        "type": "response.output_item.added",
                        "item": {"type": "message", "role": "assistant", "id": message_item_id},
                    }
                )

        self._realtime_audio_ref = None
        self._pending_tool_calls = {}
        tool_state = ToolCallStreamState()
        full_text = ""
        prompt_tokens = completion_tokens = 0
        cached_tokens: int | None = None
        sampling_params_list = coerce_param_message_types(
            list(self.engine.default_sampling_params_list),
            is_streaming=True,
        )
        result_gen = self.engine.generate(
            prompt=single_input(),
            request_id=f"rt-{self.connection_id}-{uuid4()}",
            sampling_params_list=sampling_params_list,
        )
        try:
            async for output in result_gen:
                if self._cancel_requested:
                    result.cancelled = True
                    break
                if getattr(output, "stage_id", None) == 0 and output.outputs:
                    first_output = output.outputs[0]
                    completion_tokens += len(first_output.token_ids)
                    if output.prompt_token_ids:
                        prompt_tokens = max(prompt_tokens, len(output.prompt_token_ids))
                    if getattr(output, "num_cached_tokens", None) is not None:
                        cached_tokens = max(cached_tokens or 0, output.num_cached_tokens)
                    delta_text = first_output.text or ""
                    if delta_text:
                        full_text += delta_text
                        content_delta, tool_deltas = extract_deltas(full_text, tool_state)
                        if content_delta:
                            await announce()
                            result.spoken += content_delta
                            await self.send(TranscriptionDelta(delta=content_delta))
                        if tool_deltas:
                            await self._emit_tool_call_deltas(tool_deltas)

                audio_chunks, sample_rate = self._extract_audio_chunks(output)
                if audio_chunks and not tool_state.has_tool_calls():
                    await announce()
                    for chunk in audio_chunks:
                        result.sent_audio = True
                        result.audio_seconds += len(chunk) / sample_rate
                        await self.send_json(
                            {
                                "type": "response.output_audio.delta",
                                "audio": self._pcm16_b64(chunk),
                                "format": "pcm16",
                                "sample_rate_hz": sample_rate,
                            }
                        )
                # else: a tool-call round - the pipeline still synthesizes audio for the
                # raw <tool_call> text (no cheap hook to skip talker/code2wav for just
                # this round), we just don't forward it to the client.

                if not self._is_connected:
                    break
        finally:
            # Closing aborts the engine request now; a cancelled or disconnected
            # round must not keep cycling through the stages (issue #4271).
            try:
                await result_gen.aclose()
            except Exception:
                logger.exception("Failed to close realtime result generator")
        usage.add(prompt_tokens, completion_tokens, cached_tokens, _get_mm_token_counts(engine_input))
        if tool_state.has_tool_calls():
            result.tool_calls = [self._pending_tool_calls[i] for i in sorted(self._pending_tool_calls)]
        return result

    def _record_round(self, result: _Round, message_item_id: str, response_id: str) -> list[str]:
        """Add a round's spoken reply and tool calls to the conversation; return the new item ids."""
        created: list[str] = []
        if result.spoken.strip():
            self._conversation.add(
                ConversationItem(
                    item_id=message_item_id,
                    kind="message",
                    role="assistant",
                    text=result.spoken.strip(),
                    response_id=response_id,
                    audio_out_seconds=result.audio_seconds,
                    audio_complete=not result.cancelled,
                )
            )
            if self._live_truncate_ms is not None:
                self._conversation.truncate(message_item_id, self._live_truncate_ms)
            created.append(message_item_id)
        for call in result.tool_calls:
            item_id = _new_item_id()
            self._conversation.add(
                ConversationItem(
                    item_id=item_id,
                    kind="function_call",
                    role="assistant",
                    call_id=call.call_id,
                    name=call.name,
                    arguments=call.arguments,
                    response_id=response_id,
                )
            )
            created.append(item_id)
        return created

    async def _await_tool_results(self, calls: list[_PendingToolCall]) -> list[tuple[_PendingToolCall, str]] | None:
        """Block until the client has submitted a `function_call_output` for every call, in call order.

        Returns:
            None when the turn was cancelled or the client disconnected first.
        """
        call_ids = {call.call_id for call in calls}
        results: dict[str, str] = {}
        self._awaiting_tool_results = True
        try:
            while len(results) < len(call_ids) and self._is_connected and not self._cancel_requested:
                # Bounded wait so a client that disappears mid-tool-call cannot park
                # this task forever; a slow-but-live client is unaffected.
                try:
                    item = await asyncio.wait_for(self._tool_result_queue.get(), timeout=_TOOL_RESULT_POLL_S)
                except TimeoutError:
                    continue
                call_id = item.get("call_id")
                if call_id not in call_ids:
                    # Tell the client: silently dropping this would leave the turn
                    # waiting for a result that is never going to match. Keep waiting
                    # afterwards, since the correct result may still arrive.
                    logger.warning("received function_call_output for unknown call_id=%s", call_id)
                    await self.send_error(
                        f"No pending tool call with call_id={call_id!r}; expected one of {sorted(call_ids)}",
                        "unknown_tool_call_id",
                    )
                    continue
                results[call_id] = item["output"]
        finally:
            self._awaiting_tool_results = False
        if len(results) < len(call_ids):
            return None
        return [(call, results[call.call_id]) for call in calls]

    async def send_json(self, payload: dict):
        try:
            await self.websocket.send_text(json.dumps(payload))
        except Exception:
            # A failed send means the client is gone; flag it so the
            # generation loop stops instead of retrying into a dead socket.
            self._is_connected = False
            raise
