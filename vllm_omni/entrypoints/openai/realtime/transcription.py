# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transcription sessions on the Realtime endpoint: each committed audio turn transcribed on its own."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING

from openai.types import realtime as types
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.openai.realtime.session import _gen_id, _merge_config_models

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.realtime.connection import OpenAIFullDuplexConnection

logger = init_logger(__name__)

DEFAULT_TRANSCRIPTION_PROMPT = "Transcribe the audio verbatim."
# A transcript cannot outrun its speech: 8 tokens/s covers the 99.99th
# percentile measured rate, and the constant covers short clips.
TRANSCRIPT_TOKENS_BASE = 128
TRANSCRIPT_TOKENS_PER_SECOND = 8


def transcript_token_limit(audio_seconds: float) -> int:
    """Return the most tokens a transcript of ``audio_seconds`` of speech may take."""
    return TRANSCRIPT_TOKENS_BASE + TRANSCRIPT_TOKENS_PER_SECOND * math.ceil(audio_seconds)


class TranscriptionTurns:
    """Transcribes a transcription session's committed audio turns one at a time, in commit order."""

    def __init__(self, connection: OpenAIFullDuplexConnection, config: types.RealtimeTranscriptionSessionCreateRequest):
        self._connection = connection
        self.config = config
        self._previous_item_id: str | None = None
        self._order = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._request_ids: set[str] = set()

    @property
    def pending(self) -> int:
        """The committed turns not yet transcribed, the one in the engine included."""
        return len(self._tasks)

    @property
    def prompt(self) -> str:
        """The instructions each turn is transcribed under."""
        audio_input = self.config.audio.input if self.config.audio is not None else None
        transcription = audio_input.transcription if audio_input is not None else None
        prompt = transcription.prompt if transcription is not None else None
        return prompt or DEFAULT_TRANSCRIPTION_PROMPT

    async def commit(self, audio: str, audio_seconds: float, item_id: str | None) -> None:
        """Announce a committed turn and start transcribing it.

        Args:
            audio: The turn's base64 PCM16 at the session rate.
            audio_seconds: Duration of ``audio``.
            item_id: Id for the turn's item; one is generated when None.
        """
        connection = self._connection
        item_id = item_id or _gen_id("item")
        previous_item_id = self._previous_item_id
        self._previous_item_id = item_id
        await connection._send_event(
            types.InputAudioBufferCommittedEvent(
                event_id=_gen_id("evt"),
                type="input_audio_buffer.committed",
                item_id=item_id,
                previous_item_id=previous_item_id,
            )
        )
        wire_item = types.RealtimeConversationItemUserMessage(
            type="message", role="user", status="completed", id=item_id, content=[]
        )
        await connection._send_conversation_item_added_and_done(wire_item, previous_item_id)
        task = asyncio.create_task(self._transcribe_in_order(item_id, audio, audio_seconds))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        """Stop every pending transcription and abort the one in the engine."""
        for task in list(self._tasks):
            task.cancel()
        for request_id in list(self._request_ids):
            try:
                await self._connection.engine.abort(request_id)
            except Exception:
                logger.debug("[realtime] abort of %s failed", request_id, exc_info=True)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _transcribe_in_order(self, item_id: str, audio: str, audio_seconds: float) -> None:
        async with self._order:
            try:
                await self._transcribe(item_id, audio, audio_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("[realtime] transcription of %s failed", item_id)
                await self._connection._send_event(
                    types.ConversationItemInputAudioTranscriptionFailedEvent(
                        event_id=_gen_id("evt"),
                        type="conversation.item.input_audio_transcription.failed",
                        item_id=item_id,
                        content_index=0,
                        error={"type": "server_error", "code": "transcription_failed", "message": str(exc)},
                    )
                )

    async def _transcribe(self, item_id: str, audio: str, audio_seconds: float) -> None:
        connection = self._connection
        clip = types.RealtimeConversationItemUserMessage(
            type="message", role="user", content=[{"type": "input_audio", "audio": audio}]
        )
        engine_input = await connection._build_full_prompt(instructions=self.prompt, items=[clip])

        sampling_params_list = [
            params.clone() if isinstance(params, SamplingParams) else params
            for params in connection.engine.default_sampling_params_list
        ]
        thinker = next((params for params in sampling_params_list if isinstance(params, SamplingParams)), None)
        if thinker is not None:
            thinker.output_kind = RequestOutputKind.DELTA
            thinker.max_tokens = transcript_token_limit(audio_seconds)

        request_id = f"rt-tx-{item_id}-{_gen_id('req')}"
        self._request_ids.add(request_id)
        transcript = ""
        output_tokens = 0
        input_tokens = len(engine_input.get("prompt_token_ids") or ())
        placeholders = engine_input.get("mm_placeholders") or {}
        audio_tokens = sum(placeholder.length for placeholder in placeholders.get("audio", ()))
        try:
            async for output in connection.engine.generate(
                prompt=engine_input,
                request_id=request_id,
                sampling_params_list=sampling_params_list,
                output_modalities=["text"],
            ):
                if getattr(output, "final_output_type", "text") != "text" or not output.outputs:
                    continue
                piece = output.outputs[0].text or ""
                output_tokens += len(output.outputs[0].token_ids)
                if piece:
                    transcript += piece
                    await connection._send_event(
                        types.ConversationItemInputAudioTranscriptionDeltaEvent(
                            event_id=_gen_id("evt"),
                            type="conversation.item.input_audio_transcription.delta",
                            item_id=item_id,
                            content_index=0,
                            delta=piece,
                        )
                    )
        finally:
            self._request_ids.discard(request_id)

        await connection._send_event(
            types.ConversationItemInputAudioTranscriptionCompletedEvent(
                event_id=_gen_id("evt"),
                type="conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                content_index=0,
                transcript=transcript.strip(),
                usage={
                    "type": "tokens",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "input_token_details": {
                        "audio_tokens": audio_tokens,
                        "text_tokens": max(0, input_tokens - audio_tokens),
                    },
                },
            )
        )


def merge_transcription_config(
    current: types.RealtimeTranscriptionSessionCreateRequest | None,
    update: types.RealtimeTranscriptionSessionCreateRequest,
) -> types.RealtimeTranscriptionSessionCreateRequest:
    """Apply a partial transcription ``session.update`` to the current config."""
    if current is None:
        return update.model_copy(deep=True)
    return _merge_config_models(current, update)
