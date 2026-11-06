# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn state machine: speech boundaries and end-of-turn readings in, turn events out.

The machine is pure and clocked by stream time, so it behaves the same whether audio arrives in real time or faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

from vllm_omni.entrypoints.openai.turns.protocol import (
    TurnEagerEnd,
    TurnEnded,
    TurnEvent,
    TurnResumed,
    TurnSessionConfig,
    TurnStarted,
    TurnUpdate,
)

READING_INTERVAL_MS = 100


class TurnPhase(StrEnum):
    """Where the caller is in a turn."""

    IDLE = "idle"
    SPEAKING = "speaking"
    PAUSED = "paused"


@dataclass(frozen=True, slots=True)
class Reading:
    """One end-of-turn reading to take.

    Attributes:
        pause: Pause counter at scheduling time; a reading from an older pause is discarded.
        silence_ms: Silence between the end of speech and ``audio_ms``.
        audio_ms: Stream time the reading's audio window ends at.
    """

    pause: int
    silence_ms: int
    audio_ms: int


@dataclass(slots=True)
class TurnTracker:
    """Track one caller's turns from VAD boundaries and model readings.

    Attributes:
        config: Thresholds in force; replace it to apply a session.update.
        phase: Current phase.
        turn_id: Id of the open turn; None while idle.
        pause: Incremented at every phase change, so readings from an earlier pause are recognisable.
        silence_start_ms: Stream time the current pause began.
        next_reading_ms: Silence at which the next reading is due.
        eager_sent: Whether turn.eager_end was emitted in the current pause.
        last_p_eot: Latest reading in the current pause.
    """

    config: TurnSessionConfig
    phase: TurnPhase = TurnPhase.IDLE
    turn_id: str | None = None
    pause: int = 0
    silence_start_ms: int = 0
    next_reading_ms: int = READING_INTERVAL_MS
    eager_sent: bool = False
    last_p_eot: float | None = None

    def speech_started(self, onset_ms: int) -> list[TurnEvent]:
        """Open a turn, or resume the open one, at a confirmed speech onset."""
        if self.phase is TurnPhase.IDLE:
            self.turn_id = "turn_" + uuid4().hex[:24]
            self.phase = TurnPhase.SPEAKING
            return [TurnStarted(turn_id=self.turn_id, audio_ms=onset_ms)]
        if self.phase is TurnPhase.PAUSED:
            assert self.turn_id is not None
            resumed = self.eager_sent
            self._enter(TurnPhase.SPEAKING)
            return [TurnResumed(turn_id=self.turn_id, audio_ms=onset_ms)] if resumed else []
        return []

    def speech_stopped(self, silence_start_ms: int) -> None:
        """Start a pause whose silence began at ``silence_start_ms``."""
        if self.phase is TurnPhase.SPEAKING:
            self._enter(TurnPhase.PAUSED)
            self.silence_start_ms = silence_start_ms

    def advance(self, now_ms: int, speech_pending: bool = False) -> tuple[Reading | None, list[TurnEvent]]:
        """Move the clock to ``now_ms``; return the reading now due, if any, and a timeout's turn.ended.

        Args:
            now_ms: Stream time the VAD has processed up to.
            speech_pending: The VAD is confirming a speech onset; the timeout waits until it resolves.
        """
        if self.phase is not TurnPhase.PAUSED:
            return None, []
        silence = now_ms - self.silence_start_ms
        if silence >= self.config.eot_timeout_ms:
            if speech_pending:
                return None, []
            return None, [self._end("timeout", self.last_p_eot, silence, now_ms)]
        if silence < self.next_reading_ms:
            return None, []
        self.next_reading_ms = (silence // READING_INTERVAL_MS + 1) * READING_INTERVAL_MS
        return Reading(pause=self.pause, silence_ms=silence, audio_ms=now_ms), []

    def reading(self, reading: Reading, p_eot: float) -> list[TurnEvent]:
        """Apply a finished reading; a reading from a pause that already ended yields nothing."""
        if self.phase is not TurnPhase.PAUSED or reading.pause != self.pause:
            return []
        assert self.turn_id is not None
        self.last_p_eot = p_eot
        events: list[TurnEvent] = [
            TurnUpdate(turn_id=self.turn_id, p_eot=p_eot, silence_ms=reading.silence_ms, audio_ms=reading.audio_ms)
        ]
        eager = self.config.eager_eot_threshold
        if eager is not None and not self.eager_sent and p_eot >= eager:
            self.eager_sent = True
            events.append(
                TurnEagerEnd(
                    turn_id=self.turn_id, p_eot=p_eot, silence_ms=reading.silence_ms, audio_ms=reading.audio_ms
                )
            )
        if reading.silence_ms >= self.config.min_silence_ms and p_eot >= self.config.eot_threshold:
            events.append(self._end("model", p_eot, reading.silence_ms, reading.audio_ms))
        return events

    def _enter(self, phase: TurnPhase) -> None:
        self.phase = phase
        self.pause += 1
        self.next_reading_ms = READING_INTERVAL_MS
        self.eager_sent = False
        self.last_p_eot = None

    def _end(self, reason: str, p_eot: float | None, silence_ms: int, audio_ms: int) -> TurnEnded:
        assert self.turn_id is not None
        event = TurnEnded(turn_id=self.turn_id, reason=reason, p_eot=p_eot, silence_ms=silence_ms, audio_ms=audio_ms)
        self._enter(TurnPhase.IDLE)
        self.turn_id = None
        return event
