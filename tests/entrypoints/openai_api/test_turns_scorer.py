# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.turns.connection import AudioWindow
from vllm_omni.entrypoints.openai.turns.protocol import TurnSessionConfig, TurnSessionPatch
from vllm_omni.entrypoints.openai.turns.scorer import TurnPrompt, TurnReadingError, TurnScorer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT = TurnPrompt(
    question="Done?\nA. hold\nB. complete", context_prefix="State: ", window_seconds=8.0, max_context_chars=300
)


class Response:
    def __init__(self, body: bytes) -> None:
        self.body = body


def scorer_answering(answer: bytes | Exception) -> TurnScorer:
    async def classification(request) -> Response:
        if isinstance(answer, Exception):
            raise answer
        return Response(answer)

    return TurnScorer(classification=classification, model="turns", prompt=PROMPT)


def score(answer: bytes | Exception) -> float:
    return asyncio.run(scorer_answering(answer).score(np.zeros(1_600, dtype=np.float32), "", "r-1"))


def test_probability_is_read_from_the_classification_body():
    assert score(json.dumps({"data": [{"probs": [0.7]}]}).encode()) == pytest.approx(0.7)


@pytest.mark.parametrize(
    "answer",
    [
        b"not json",
        json.dumps({"error": {"message": "engine died"}}).encode(),
        json.dumps({"data": [{"probs": [1.5]}]}).encode(),
        json.dumps({"data": [{"probs": [float("nan")]}]}).encode(),
        json.dumps(["unexpected"]).encode(),
        RuntimeError("engine raised"),
    ],
)
def test_bad_classification_raises_a_reading_error(answer):
    with pytest.raises(TurnReadingError):
        score(answer)


def test_classification_error_message_is_kept():
    with pytest.raises(TurnReadingError, match="engine died"):
        score(json.dumps({"error": {"message": "engine died"}}).encode())


def test_audio_window_returns_the_latest_audio_across_wraparound():
    stream = np.arange(100_000, dtype=np.float32)
    chunked, whole = AudioWindow(1.0), AudioWindow(1.0)
    for start in range(0, stream.size, 7_000):
        chunked.append(stream[start : start + 7_000])
    whole.append(stream)
    for window in (chunked, whole):
        np.testing.assert_array_equal(window.until(6_250, 1.0), stream[-16_000:])
        np.testing.assert_array_equal(window.until(5_000, 1.0), stream[64_000:80_000])
        assert window.until(3_000, 1.0).size == 0


def test_patch_can_disable_eager_but_not_null_other_thresholds():
    config = TurnSessionConfig(eager_eot_threshold=0.4)
    assert TurnSessionPatch(eager_eot_threshold=None).apply(config).eager_eot_threshold is None
    with pytest.raises(ValidationError):
        TurnSessionPatch(eot_threshold=None).apply(config)
