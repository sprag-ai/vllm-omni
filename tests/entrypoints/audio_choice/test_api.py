# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.audio_choice.contract import ChoiceResponse, Usage, primitive_answer, scoring_question
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceRequest
from vllm_omni.entrypoints.openai.serving_choice import ChoiceServing, build_choice_app


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


def body():
    return {
        "model": "spev",
        "state": {"transcript": "Please help with my invoice"},
        "questions": {
            "routing": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payments", "technical": "Software"},
            }
        },
    }


class Engine:
    is_async = True
    errored = False

    def __init__(self):
        self.calls = []
        self.stopped = False

    async def evaluate(self, request, wave):
        self.calls.append((request, wave))
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


def test_native_route_auth_wire_response_and_openapi():
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        assert client.post("/v1/systemone", json=body()).status_code == 401
        assert engine.calls == []
        r = client.post("/v1/systemone", json=body(), headers=HEADERS)
        assert r.status_code == 200, r.text
        out = r.json()
        assert set(out) == {"model", "answers", "usage"}
        assert set(out["answers"]["routing"]) == {"type", "choice", "probabilities", "confidence"}
        assert out["answers"]["routing"]["choice"] == "billing"
        assert set(out["answers"]["routing"]["probabilities"]) == {"billing", "technical"}
        ChoiceResponse.model_validate(out)
        schema = client.get("/openapi.json").json()
        operation = schema["paths"]["/v1/systemone"]["post"]
        assert operation["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/ChoiceResponse"
        )
        request_schema = schema["components"]["schemas"]["BoundChoiceRequest"]
        assert {"model", "state", "questions"} <= set(request_schema["required"])
        assert request_schema["additionalProperties"] is False
        assert client.get("/v1/models", headers=HEADERS).json()["data"][0]["max_model_len"] == 8192
        assert "/v1/embeddings" not in schema["paths"]
        assert "/v1/completions" not in schema["paths"]
    assert engine.stopped


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"messages": []},
        {"state": None},
        {"state": 17},
        {"questions": {}},
        {"stream": True},
        {"temperature": 0.5},
        {"max_tokens": 1},
        {"questions": {"q": {"type": "unknown", "instructions": "Is it urgent?"}}},
        {"questions": {"q": {"instructions": "Decide", "criteria": {"a": None, "b": None}}}},
        {"questions": {"q": {"type": "choice", "instructions": "Decide", "criteria": {"a": None}}}},
        {"questions": {"q": {"type": "choice", "instructions": "Decide", "criteria": {"a": True, "b": None}}}},
        {
            "questions": {
                "q": {"type": "choice", "instructions": "Decide", "criteria": {"a": None, "b": None}, "ignored": 1}
            }
        },
        {"input_audio": {"format": "wav", "data": "invalid PRIVATE EVIDENCE"}},
        {"input_audio": {"format": "wav", "data": ""}},
        {"input_audio": {"url": "https://example.com/audio.wav"}},
    ],
)
def test_invalid_request_rejected_at_edge_without_inference(change):
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        r = client.post("/v1/systemone", json={**body(), **change}, headers=HEADERS)
        assert r.status_code == 422, r.text
        assert not engine.calls
        assert "PRIVATE EVIDENCE" not in r.text
        assert "detail" in r.json()


def test_multiple_question_ids_structured_rubrics_and_audio_extension():
    b = body()
    b["questions"]["caller-owned-id"] = copy.deepcopy(b["questions"]["routing"])
    b["questions"]["caller-owned-id"]["criteria"] = {"長い名前": {"rubric": ["one", "two"]}, "name with spaces": None}
    audio = io.BytesIO()
    sf.write(audio, np.zeros(1600), 16000, format="WAV")
    b["input_audio"] = {"format": "anything-informational", "data": base64.b64encode(audio.getvalue()).decode()}
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        r = client.post("/v1/systemone", json=b, headers=HEADERS)
        assert r.status_code == 200, r.text
        assert set(r.json()["answers"]) == set(b["questions"])
        assert len(engine.calls[0][1]) == 1600


def test_chat_wrapper_preserves_same_typed_result():
    b = body()
    payload = {k: v for k, v in b.items() if k != "model"}
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        native = client.post("/v1/systemone", json=b, headers=HEADERS).json()
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "spev",
                "messages": [{"role": "user", "content": [{"type": "text", "text": json.dumps(payload)}]}],
            },
            headers=HEADERS,
        )
        assert r.status_code == 200, r.text
        assert json.loads(r.json()["choices"][0]["message"]["content"]) == native


def test_request_bound_and_queue_cleanup_after_disconnected_waiter():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        engine = Engine()
        original = engine.evaluate

        async def delayed(*a):
            entered.set()
            await release.wait()
            return await original(*a)

        engine.evaluate = delayed
        handler = ChoiceServing(engine, "spev", 1)
        req = ChoiceRequest.model_validate(body())
        waiter = asyncio.create_task(handler.evaluate(req))
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert handler.pending == 1
        with pytest.raises(HTTPException) as exc:
            await handler.evaluate(req)
        assert exc.value.status_code == 429
        release.set()
        await handler.close()
        await asyncio.sleep(0)
        assert handler.pending == 0 and engine.stopped

    asyncio.run(run())


@pytest.mark.parametrize("fatal", [False, True])
def test_request_failure_does_not_latch_healthy_engine(fatal):
    engine = Engine()
    original = engine.evaluate

    async def fail(*a):
        engine.errored = fatal
        raise RuntimeError("PRIVATE INTERNAL FAILURE")

    engine.evaluate = fail
    with TestClient(build_choice_app(args(), engine), raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", json=body(), headers=HEADERS)
        assert response.status_code == (503 if fatal else 500)
        assert "PRIVATE INTERNAL" not in response.text
        assert client.get("/load").json() == {"server_load": 0}
        assert client.get("/health").status_code == (500 if fatal else 200)
        engine.evaluate = original
        assert client.post("/v1/systemone", json=body(), headers=HEADERS).status_code == (503 if fatal else 200)


@pytest.mark.parametrize("route", ["/v1/systemone", "/v1/chat/completions"])
def test_load_counts_admitted_work_and_health_remains_responsive(route):
    import httpx

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        engine = Engine()
        original = engine.evaluate

        async def delayed(*a):
            entered.set()
            await release.wait()
            return await original(*a)

        engine.evaluate = delayed
        app = build_choice_app(args(), engine)
        payload = body()
        if route.endswith("chat/completions"):
            payload = {
                "model": "spev",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": json.dumps({k: v for k, v in body().items() if k != "model"})}
                        ],
                    }
                ],
            }
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=HEADERS
        ) as client:
            request = asyncio.create_task(client.post(route, json=payload))
            await entered.wait()
            assert (await client.get("/load")).json() == {"server_load": 1}
            assert (await client.get("/health")).status_code == 200
            assert (await client.post(route, json=payload)).status_code == 429
            assert (await client.get("/load")).json() == {"server_load": 1}
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request
            assert (await client.get("/load")).json() == {"server_load": 1}
            release.set()
            await app.state.engine_client.close()
            assert (await client.get("/load")).json() == {"server_load": 0}

    asyncio.run(run())


@pytest.mark.parametrize("state", ["x" * 65537, [None] * 4097])
def test_text_work_budget_rejected_before_inference(state):
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        assert client.post("/v1/systemone", json={**body(), "state": state}, headers=HEADERS).status_code == 422
        assert not engine.calls


def test_native_mixed_primitives_have_discriminated_responses():
    b = body()
    b["questions"].update(
        {
            "assertion": {"type": "noul", "instructions": "Is it urgent?"},
            "rating": {"type": "score", "instructions": "Urgency", "criteria": ["routine", "urgent"]},
        }
    )
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        r = client.post("/v1/systemone", json=b, headers=HEADERS)
        assert r.status_code == 200, r.text
        answers = r.json()["answers"]
        assert set(answers["assertion"]) == {"type", "noul"}
        assert set(answers["rating"]) == {"type", "score", "confidence", "legend", "probabilities"}
        ChoiceResponse.model_validate(r.json())
