# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One ``/v1/turns`` WebSocket session: audio and context in, turn events out."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError
from vllm.logger import init_logger

from vllm_omni.engine.duplex.vad import SileroStreamingVAD, SileroVADConfig
from vllm_omni.entrypoints.openai.turns.detector import Reading, TurnTracker
from vllm_omni.entrypoints.openai.turns.protocol import (
    CLIENT_EVENT,
    SAMPLE_RATE_HZ,
    SessionCreated,
    SessionUpdated,
    TurnContextUpdate,
    TurnError,
    TurnErrorBody,
    TurnSessionClose,
    TurnSessionConfig,
    TurnSessionInfo,
    TurnSessionUpdate,
)
from vllm_omni.entrypoints.openai.turns.scorer import TurnReadingError, TurnScorer

logger = init_logger(__name__)

MAX_AUDIO_FRAME_BYTES = 10 * SAMPLE_RATE_HZ * 2
VAD_CONFIG = SileroVADConfig(threshold=0.5, prefix_padding_ms=0, silence_duration_ms=96, min_speech_duration_ms=96)


class AudioWindow:
    """The most recent caller audio, addressable by absolute stream sample."""

    def __init__(self, seconds: float) -> None:
        self.capacity = int(seconds * SAMPLE_RATE_HZ) + 2 * SAMPLE_RATE_HZ
        self.samples = np.zeros(0, dtype=np.float32)
        self.start = 0

    def append(self, samples: np.ndarray) -> None:
        """Add samples at the end, dropping the oldest beyond capacity."""
        merged = np.concatenate((self.samples, samples))
        drop = max(0, merged.size - self.capacity)
        self.samples = merged[drop:]
        self.start += drop

    def until(self, end_ms: int, seconds: float) -> np.ndarray:
        """Return up to ``seconds`` of audio ending at stream time ``end_ms``."""
        end = end_ms * SAMPLE_RATE_HZ // 1000 - self.start
        begin = max(0, end - int(seconds * SAMPLE_RATE_HZ))
        return self.samples[begin : max(begin, end)].copy()


class TurnsConnection:
    """Run one turns session over an accepted WebSocket."""

    def __init__(self, websocket: WebSocket, scorer: TurnScorer, vad: SileroStreamingVAD) -> None:
        self.websocket = websocket
        self.scorer = scorer
        self.session_id = "turns_" + uuid4().hex[:24]
        self.tracker = TurnTracker(config=TurnSessionConfig())
        self.vad = vad
        self.window = AudioWindow(scorer.prompt.window_seconds)
        self.context = ""
        self.inflight: asyncio.Task[None] | None = None
        self.readings = 0
        self.send_lock = asyncio.Lock()

    def info(self) -> TurnSessionInfo:
        """Describe the session as it stands."""
        return TurnSessionInfo(
            id=self.session_id,
            model=self.scorer.model,
            sample_rate=SAMPLE_RATE_HZ,
            encoding="pcm16",
            config=self.tracker.config,
        )

    async def send(self, event: BaseModel) -> None:
        """Send one server event."""
        async with self.send_lock:
            await self.websocket.send_text(event.model_dump_json())

    async def error(self, code: str, message: str) -> None:
        """Send an error event."""
        await self.send(TurnError(error=TurnErrorBody(code=code, message=message)))

    async def run(self) -> None:
        """Serve the session until the client closes it or disconnects."""
        await self.send(SessionCreated(session=self.info()))
        try:
            while True:
                message = await self.websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await self.audio(message["bytes"])
                elif message.get("text") is not None and await self.event(message["text"]):
                    break
        except WebSocketDisconnect:
            return
        finally:
            if self.inflight is not None:
                self.inflight.cancel()
        await self.websocket.close()

    async def event(self, text: str) -> bool:
        """Apply one JSON client event; return True when it closes the session."""
        try:
            event = CLIENT_EVENT.validate_json(text)
        except ValidationError as exc:
            await self.error("invalid_event", exc.errors(include_url=False, include_input=False)[0]["msg"])
            return False
        if isinstance(event, TurnContextUpdate):
            self.context = event.agent_text
        elif isinstance(event, TurnSessionUpdate):
            try:
                self.tracker.config = event.session.apply(self.tracker.config)
            except ValidationError as exc:
                await self.error("invalid_event", exc.errors(include_url=False, include_input=False)[0]["msg"])
                return False
            await self.send(SessionUpdated(session=self.info()))
        elif isinstance(event, TurnSessionClose):
            if self.inflight is not None:
                await asyncio.gather(self.inflight, return_exceptions=True)
            return True
        return False

    async def audio(self, data: bytes) -> None:
        """Feed one binary audio frame through the VAD and the turn tracker."""
        if len(data) % 2 or len(data) > MAX_AUDIO_FRAME_BYTES:
            await self.error("invalid_audio", "audio frames must be whole PCM16 samples, at most 10 s each")
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) * np.float32(1.0 / 32768.0)
        self.window.append(samples)
        offset = 0
        while offset < samples.size:
            step = min(self.vad.samples_until_frame, samples.size - offset)
            result = await asyncio.to_thread(self.vad.process, samples[offset : offset + step])
            offset += step
            events = []
            if result.speech_started and result.speech_start_ms is not None:
                events += self.tracker.speech_started(result.speech_start_ms)
            if result.speech_stopped and result.silence_start_ms is not None:
                self.tracker.speech_stopped(result.silence_start_ms)
            reading, ended = self.tracker.advance(self.vad.processed_ms)
            events += ended
            for event in events:
                await self.send(event)
            if reading is not None and self.inflight is None:
                self.inflight = asyncio.create_task(self.take(reading))

    async def take(self, reading: Reading) -> None:
        """Score one reading and send the events it produces."""
        self.readings += 1
        samples = self.window.until(reading.audio_ms, self.scorer.prompt.window_seconds)
        try:
            p_eot = await self.scorer.score(samples, self.context, f"{self.session_id}-{self.readings}")
        except TurnReadingError as exc:
            logger.warning("turn reading failed in %s: %s", self.session_id, exc)
            await self.error("reading_failed", str(exc))
            return
        finally:
            self.inflight = None
        for event in self.tracker.reading(reading, p_eot):
            await self.send(event)
