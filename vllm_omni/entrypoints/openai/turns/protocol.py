# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wire events of the ``/v1/turns`` WebSocket.

Audio travels as binary frames of 16 kHz mono PCM16 little-endian samples; every other frame is one JSON event.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SAMPLE_RATE_HZ = 16_000

Probability = Annotated[float, Field(ge=0.0, le=1.0)]
SilenceMs = Annotated[int, Field(ge=0, le=10_000)]
TimeoutMs = Annotated[int, Field(ge=100, le=60_000)]


class TurnSessionConfig(BaseModel):
    """Thresholds that turn the model's end-of-turn probability into turn events."""

    model_config = ConfigDict(extra="forbid")

    eot_threshold: Probability = Field(default=0.6, description="probability that ends the turn")
    eager_eot_threshold: Probability | None = Field(
        default=None, description="lower probability that emits turn.eager_end once per pause; null disables it"
    )
    min_silence_ms: SilenceMs = Field(default=200, description="silence the turn needs before the model can end it")
    eot_timeout_ms: TimeoutMs = Field(default=3_000, description="silence that ends the turn whatever the model scores")


class TurnSessionPatch(BaseModel):
    """Threshold changes; a field left out keeps its current value."""

    model_config = ConfigDict(extra="forbid")

    eot_threshold: Probability | None = Field(default=None, description="new eot_threshold")
    eager_eot_threshold: Probability | None = Field(default=None, description="new eager_eot_threshold; null disables")
    min_silence_ms: SilenceMs | None = Field(default=None, description="new min_silence_ms")
    eot_timeout_ms: TimeoutMs | None = Field(default=None, description="new eot_timeout_ms")

    def apply(self, config: TurnSessionConfig) -> TurnSessionConfig:
        """Return ``config`` with this patch's explicitly set fields replaced.

        Raises:
            pydantic.ValidationError: A field other than eager_eot_threshold was set to null.
        """
        return TurnSessionConfig.model_validate(config.model_dump() | self.model_dump(exclude_unset=True))


class TurnSessionUpdate(BaseModel):
    """Client event that changes the session's thresholds."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["session.update"] = Field(description="event type")
    session: TurnSessionPatch = Field(description="the thresholds to change")


class TurnContextUpdate(BaseModel):
    """Client event that sets the agent's last utterance, read by every later reading."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["context.update"] = Field(description="event type")
    agent_text: str = Field(max_length=20_000, description="what the agent last said; empty clears the context")


class TurnSessionClose(BaseModel):
    """Client event that ends the session once in-flight readings finish."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["session.close"] = Field(description="event type")


TurnClientEvent = Annotated[TurnSessionUpdate | TurnContextUpdate | TurnSessionClose, Field(discriminator="type")]
CLIENT_EVENT = TypeAdapter(TurnClientEvent)


class TurnSessionInfo(BaseModel):
    """Session description sent on creation and after every update."""

    id: str = Field(description="session id")
    model: str = Field(description="served model name")
    sample_rate: int = Field(description="required input sample rate in Hz")
    encoding: Literal["pcm16"] = Field(description="required binary audio encoding")
    config: TurnSessionConfig = Field(description="thresholds in force")


class SessionCreated(BaseModel):
    """Server event sent once, right after the socket is accepted."""

    type: Literal["session.created"] = "session.created"
    session: TurnSessionInfo = Field(description="the new session")


class SessionUpdated(BaseModel):
    """Server event acknowledging a session.update."""

    type: Literal["session.updated"] = "session.updated"
    session: TurnSessionInfo = Field(description="the session with its new thresholds")


class TurnStarted(BaseModel):
    """Server event: the caller started speaking and a new turn opened."""

    type: Literal["turn.started"] = "turn.started"
    turn_id: str = Field(description="id of the opened turn")
    audio_ms: int = Field(description="stream time of the speech onset, from the first audio byte")


class TurnUpdate(BaseModel):
    """Server event carrying one reading of the end-of-turn probability during a pause."""

    type: Literal["turn.update"] = "turn.update"
    turn_id: str = Field(description="id of the open turn")
    p_eot: float = Field(description="probability that the caller has finished the turn")
    silence_ms: int = Field(description="silence before the reading, measured from the end of speech")
    audio_ms: int = Field(description="stream time the reading covers up to")


class TurnEagerEnd(BaseModel):
    """Server event: p_eot first reached eager_eot_threshold in this pause."""

    type: Literal["turn.eager_end"] = "turn.eager_end"
    turn_id: str = Field(description="id of the open turn")
    p_eot: float = Field(description="probability that crossed the eager threshold")
    silence_ms: int = Field(description="silence before the reading")
    audio_ms: int = Field(description="stream time the reading covers up to")


class TurnResumed(BaseModel):
    """Server event: the caller spoke again after a turn.eager_end, so the turn stays open."""

    type: Literal["turn.resumed"] = "turn.resumed"
    turn_id: str = Field(description="id of the open turn")
    audio_ms: int = Field(description="stream time of the renewed speech onset")


class TurnEnded(BaseModel):
    """Server event: the turn closed."""

    type: Literal["turn.ended"] = "turn.ended"
    turn_id: str = Field(description="id of the closed turn")
    reason: Literal["model", "timeout"] = Field(
        description="model when p_eot reached eot_threshold, timeout when eot_timeout_ms of silence passed"
    )
    p_eot: float | None = Field(description="last reading of the turn; null when the turn timed out unread")
    silence_ms: int = Field(description="silence before the turn closed")
    audio_ms: int = Field(description="stream time the turn closed at")


class TurnErrorBody(BaseModel):
    """Error details."""

    code: str = Field(description="machine-readable error code")
    message: str = Field(description="human-readable description")


class TurnError(BaseModel):
    """Server event reporting a rejected client frame or a failed reading."""

    type: Literal["error"] = "error"
    error: TurnErrorBody = Field(description="what went wrong")


TurnEvent = TurnStarted | TurnUpdate | TurnEagerEnd | TurnResumed | TurnEnded
