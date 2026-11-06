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
from vllm_omni.entrypoints.openai.turns.scorer import TurnScorer

logger = init_logger(__name__)

MAX_AUDIO_FRAME_SECONDS = 10
MAX_AUDIO_FRAME_BYTES = MAX_AUDIO_FRAME_SECONDS * SAMPLE_RATE_HZ * 2
VAD_CONFIG = SileroVADConfig(threshold=0.5, prefix_padding_ms=0, silence_duration_ms=96, min_speech_duration_ms=96)


class AudioWindow:
    """The most recent caller audio in a fixed ring, addressable by absolute stream sample."""

    def __init__(self, seconds: float) -> None:
        self.capacity = int(seconds * SAMPLE_RATE_HZ) + 2 * SAMPLE_RATE_HZ
        self.ring = np.zeros(self.capacity, dtype=np.float32)
        self.written = 0

    def append(self, samples: np.ndarray) -> None:
        """Add samples at the end, overwriting the oldest beyond capacity."""
        dropped = max(samples.size - self.capacity, 0)
        self.written += dropped
        samples = samples[dropped:]
        head = self.written % self.capacity
        first = min(samples.size, self.capacity - head)
        self.ring[head : head + first] = samples[:first]
        self.ring[: samples.size - first] = samples[first:]
        self.written += samples.size

    def until(self, end_ms: int, seconds: float) -> np.ndarray:
        """Return up to ``seconds`` of audio ending at stream time ``end_ms``; older audio is gone."""
        end = min(end_ms * SAMPLE_RATE_HZ // 1000, self.written)
        begin = max(end - int(seconds * SAMPLE_RATE_HZ), self.written - self.capacity, 0)
        if end <= begin:
            return np.zeros(0, dtype=np.float32)
        indices = np.arange(begin, end) % self.capacity
        return self.ring[indices]


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
        self.inflight_pause = -1
        self.inflight_delivering = False
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
        await self.send_all([event])

    async def send_all(self, events: list[BaseModel]) -> None:
        """Send events in order, with no other sender's events between them."""
        if not events:
            return
        async with self.send_lock:
            for event in events:
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
            await self.error(
                "invalid_audio", f"audio frames must be whole PCM16 samples, at most {MAX_AUDIO_FRAME_SECONDS} s each"
            )
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) * np.float32(1.0 / 32768.0)
        offset = 0
        while offset < samples.size:
            step = min(self.vad.samples_until_frame, samples.size - offset)
            piece = samples[offset : offset + step]
            offset += step
            self.window.append(piece)
            if step < self.vad.samples_until_frame:
                result = self.vad.process(piece)
            else:
                result = await asyncio.to_thread(self.vad.process, piece)
            events = []
            if result.speech_started and result.speech_start_ms is not None:
                events += self.tracker.speech_started(result.speech_start_ms)
            if result.speech_stopped and result.silence_start_ms is not None:
                self.tracker.speech_stopped(result.silence_start_ms)
            reading, ended = self.tracker.advance(self.vad.processed_ms, self.vad.speech_pending)
            await self.send_all(events + ended)
            if reading is not None:
                self.schedule(reading)

    def schedule(self, reading: Reading) -> None:
        """Start a reading unless one is delivering events or scoring the same pause; an older pause's is dropped."""
        if self.inflight is not None:
            if self.inflight_pause == reading.pause or self.inflight_delivering:
                return
            self.inflight.cancel()
        self.inflight_pause = reading.pause
        self.inflight = asyncio.create_task(self.take(reading))

    async def take(self, reading: Reading) -> None:
        """Score one reading and send the events it produces."""
        self.readings += 1
        samples = self.window.until(reading.audio_ms, self.scorer.prompt.window_seconds)
        task = asyncio.current_task()
        try:
            try:
                p_eot = await self.scorer.score(samples, self.context, f"{self.session_id}-{self.readings}")
            except Exception:
                logger.warning("turn reading failed in %s", self.session_id, exc_info=True)
                self.inflight_delivering = True
                await self.error("reading_failed", "the end-of-turn reading failed")
                return
            self.inflight_delivering = True
            await self.send_all(self.tracker.reading(reading, p_eot))
        finally:
            if self.inflight is task:
                self.inflight = None
                self.inflight_delivering = False
