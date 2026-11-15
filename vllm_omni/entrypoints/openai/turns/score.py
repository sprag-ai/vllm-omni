# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``POST /v1/turns``: one end-of-turn probability for an uploaded clip."""

from __future__ import annotations

import io
import wave
from http import HTTPStatus
from uuid import uuid4

import numpy as np
from fastapi import Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vllm_omni.entrypoints.openai.turns.protocol import SAMPLE_RATE_HZ, Probability
from vllm_omni.entrypoints.openai.turns.scorer import TurnReadingError, TurnScorer


class TurnScore(BaseModel):
    """The body of a successful ``POST /v1/turns``."""

    p_eot: Probability = Field(description="probability that the speaker has finished their turn")


class ClipError(ValueError):
    """The uploaded clip is not 16 kHz mono PCM16 WAV audio."""


def wav_samples(data: bytes) -> np.ndarray:
    """Decode a 16 kHz mono PCM16 WAV file into float32 samples in [-1, 1].

    Raises:
        ClipError: The bytes are not a WAV file, or its format is not 16 kHz mono PCM16, or it holds no samples.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as clip:
            if (clip.getframerate(), clip.getnchannels(), clip.getsampwidth()) != (SAMPLE_RATE_HZ, 1, 2):
                raise ClipError(f"audio must be {SAMPLE_RATE_HZ} Hz mono 16-bit PCM WAV")
            frames = clip.readframes(clip.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ClipError("audio is not a readable WAV file") from exc
    if not frames:
        raise ClipError("audio holds no samples")
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def _error(status: HTTPStatus, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": "invalid_request_error", "code": code}}
    )


async def score_turn(request: Request, file: UploadFile, agent_text: str = Form(default="")) -> JSONResponse:
    """Score whether the speaker in ``file`` has finished their turn, with ``agent_text`` as the agent's last line.

    Only the last ``window_seconds`` of the clip reach the model, as in a WebSocket reading.
    """
    scorer: TurnScorer = request.app.state.turns_scorer
    try:
        samples = wav_samples(await file.read())
    except ClipError as exc:
        return _error(HTTPStatus.BAD_REQUEST, str(exc), "invalid_audio")
    window = int(scorer.prompt.window_seconds * SAMPLE_RATE_HZ)
    try:
        p_eot = await scorer.score(samples[-window:], agent_text, f"turns-http-{uuid4().hex}")
    except TurnReadingError as exc:
        return JSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"error": {"message": str(exc), "type": "server_error", "code": "reading_failed"}},
        )
    return JSONResponse(content=TurnScore(p_eot=p_eot).model_dump())
