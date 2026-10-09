# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tests.entrypoints.audio_choice.test_decisions import payload
from vllm_omni.entrypoints.audio_choice.contract import ChoiceResponse, Usage, primitive_answer, scoring_question
from vllm_omni.entrypoints.audio_choice.decisions import DecisionsRequest
from vllm_omni.entrypoints.audio_choice.decisions_adapter import to_choice
from vllm_omni.entrypoints.openai.serving_choice import ChoiceServing, build_choice_app

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def args():
    return SimpleNamespace(
        disable_fastapi_docs=False,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=["test-secret"],
        enable_request_id_headers=True,
        middleware=[],
        served_model_name=["spev"],
        decision_max_pending=1,
        enable_server_load_tracking=True,
        disable_log_stats=True,
        log_error_stack=True,
    )


class Engine:
    is_async = True
    errored = False

    def __init__(self):
        self.calls = []
        self.stopped = False

    async def evaluate(self, request, wave, visual=None):
        self.calls.append((request, wave))
        self.visual = visual
        return ChoiceResponse(
            model="spev-choice-test",
            answers={
                key: primitive_answer(q, {name: -float(i) for i, name in enumerate(scoring_question(q).criteria)})
                for key, q in request.questions.items()
            },
            usage=Usage(input_tokens=42, output_tokens=0),
        )

    def shutdown(self):
        self.stopped = True


HEADERS = {"Authorization": "Bearer test-secret"}


def test_route_auth_contract_and_openapi():
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        assert client.post("/v1/decisions", json=payload()).status_code == 401
        assert not engine.calls
        response = client.post("/v1/decisions", json=payload(), headers=HEADERS)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["answers"][0]["choice"] == "billing"
        assert result["answers"][0]["name"] == "route"
        assert result["usage"]["total_tokens"] == 42
        schema = client.get("/openapi.json").json()
        assert "/v1/decisions" in schema["paths"]
        retired = client.post("/v1/systemone", content=b"not JSON", headers=HEADERS)
        assert retired.status_code == 410
        assert retired.json()["error"]["code"] == "endpoint_removed"
        assert "POST /v1/decisions" in retired.json()["error"]["message"]
        assert schema["paths"]["/v1/systemone"]["post"]["deprecated"]
        assert "/v1/chat/completions" not in schema["paths"]
        assert len(engine.calls) == 1
        assert schema["components"]["schemas"]["BoundDecisionsRequest"]["additionalProperties"] is False
    assert engine.stopped


@pytest.mark.parametrize(
    "change",
    [
        {"model": "wrong"},
        {"stream": True},
        {"questions": {}},
        {"input": "x" * 65537},
        {"input_audio": {"data": "PRIVATE EVIDENCE", "format": "wav"}},
        {"input": [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/private"}]}]},
    ],
)
def test_validation_before_inference_and_redaction(change):
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post("/v1/decisions", json={**payload(), **change}, headers=HEADERS)
        assert response.status_code == 422, response.text
        assert not engine.calls
        assert "PRIVATE EVIDENCE" not in response.text
        assert "example.com/private" not in response.text


def test_native_audio_decoded_once():
    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600), 16000, format="WAV")
    data = {**payload(), "input_audio": {"data": base64.b64encode(audio.getvalue()).decode(), "format": "wav"}}
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post("/v1/decisions", json=data, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert len(engine.calls) == 1 and len(engine.calls[0][1]) == 1600


def test_queue_retained_after_waiter_disconnect_and_drained():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        engine = Engine()
        original = engine.evaluate

        async def delayed(*args):
            entered.set()
            await release.wait()
            return await original(*args)

        engine.evaluate = delayed
        handler = ChoiceServing(engine, "spev", 1)
        native = to_choice(DecisionsRequest.model_validate(payload()))
        waiter = asyncio.create_task(handler.evaluate(native))
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert handler.pending == 1
        with pytest.raises(HTTPException) as exc:
            await handler.evaluate(native)
        assert exc.value.status_code == 429
        release.set()
        await handler.close()
        assert handler.pending == 0 and engine.stopped

    asyncio.run(run())


@pytest.mark.parametrize("fatal", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_failure_and_engine_health(fatal, error_type):
    engine = Engine()
    original = engine.evaluate

    async def fail(*args):
        engine.errored = fatal
        raise error_type("PRIVATE FAILURE")

    engine.evaluate = fail
    with TestClient(build_choice_app(args(), engine), raise_server_exceptions=False) as client:
        response = client.post("/v1/decisions", json=payload(), headers=HEADERS)
        assert response.status_code == (503 if fatal else (422 if error_type is ValueError else 500))
        if fatal or error_type is RuntimeError:
            assert "PRIVATE FAILURE" not in response.text
        assert client.get("/load").json() == {"server_load": 0}
        engine.evaluate = original
        assert client.post("/v1/decisions", json=payload(), headers=HEADERS).status_code == (503 if fatal else 200)
