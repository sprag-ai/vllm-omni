# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.turns.plugin import TurnsEndpointPlugin
from vllm_omni.entrypoints.openai.turns.scorer import TurnPrompt, TurnReadingError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT = TurnPrompt(
    question="Done?\nA. hold\nB. complete", context_prefix="State: ", window_seconds=8.0, max_context_chars=300
)


class RecordingScorer:
    def __init__(self, answer: float | Exception) -> None:
        self.prompt = PROMPT
        self.answer = answer
        self.calls: list[tuple[np.ndarray, str]] = []

    async def score(self, samples: np.ndarray, context: str, request_id: str) -> float:
        self.calls.append((samples, context))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def wav(seconds: float, *, rate: int = 16_000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.full(int(seconds * rate) * channels, 1000, dtype="<i2").tobytes())
    return buffer.getvalue()


def client(scorer: RecordingScorer) -> TestClient:
    app = FastAPI()
    TurnsEndpointPlugin().attach_router(app)
    app.state.turns_scorer = scorer
    return TestClient(app)


def post(scorer: RecordingScorer, audio: bytes, **form: str):
    return client(scorer).post("/v1/turns", files={"file": ("clip.wav", audio, "audio/wav")}, data=form)


def test_a_clip_returns_the_scorer_probability_with_the_agent_text():
    scorer = RecordingScorer(0.83)
    response = post(scorer, wav(2.0), agent_text="What's your zip code?")
    assert response.status_code == 200
    assert response.json() == {"p_eot": pytest.approx(0.83)}
    samples, context = scorer.calls[0]
    assert context == "What's your zip code?"
    assert samples.size == 32_000
    assert samples[0] == pytest.approx(1000 / 32768)


def test_agent_text_defaults_to_empty():
    scorer = RecordingScorer(0.5)
    assert post(scorer, wav(1.0)).status_code == 200
    assert scorer.calls[0][1] == ""


def test_only_the_last_window_reaches_the_scorer():
    scorer = RecordingScorer(0.5)
    assert post(scorer, wav(10.0)).status_code == 200
    assert scorer.calls[0][0].size == 8 * 16_000


@pytest.mark.parametrize(
    "audio",
    [wav(1.0, rate=8_000), wav(1.0, channels=2), b"not a wav file", wav(0.0)],
    ids=["8 kHz", "stereo", "not wav", "empty"],
)
def test_audio_that_is_not_16_khz_mono_pcm16_is_refused(audio):
    scorer = RecordingScorer(0.5)
    response = post(scorer, audio)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_audio"
    assert scorer.calls == []


def test_a_failed_reading_is_a_server_error():
    response = post(RecordingScorer(TurnReadingError("engine died")), wav(1.0))
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "reading_failed"


def test_the_websocket_route_is_still_registered():
    app = FastAPI()
    TurnsEndpointPlugin().attach_router(app)
    kinds = {(route.path, type(route).__name__) for route in app.routes}
    assert ("/v1/turns", "APIWebSocketRoute") in kinds
    assert ("/v1/turns", "APIRoute") in kinds
