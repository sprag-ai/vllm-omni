# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import copy
import io
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from tests.entrypoints.audio_choice.helpers import fake_llm
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


def visual_engine(processor):
    import threading

    from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine

    engine = Engine()
    engine.enable_vision = True
    engine.processor = processor
    engine.visual_lock = threading.Lock()
    engine.max_video_seconds, engine.max_video_frames = 60, 1800
    engine.decode_visual = lambda request: AsyncChoiceEngine.decode_visual(engine, request)
    return engine


def wire_body(route, modality, data):
    payload = body()
    media = {"format": "hint", "data": base64.b64encode(data).decode()}
    if route == "/v1/systemone":
        return {**payload, "input_" + modality: media}
    return {
        "model": "spev",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps({k: v for k, v in payload.items() if k != "model"})},
                    {"type": "input_" + modality, "input_" + modality: media},
                ],
            }
        ],
    }


@pytest.mark.parametrize("route", ["/v1/systemone", "/v1/chat/completions"])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_visual_http_decode_redaction_and_recovery(route, modality, visual_processor, video_bytes):
    data = io.BytesIO()
    Image.new("RGB", (56, 56), "red").save(data, format="PNG")
    engine = visual_engine(visual_processor)
    with TestClient(build_choice_app(args(), engine)) as client:
        invalid = client.post(route, json=wire_body(route, modality, b"PRIVATE MEDIA"), headers=HEADERS)
        assert invalid.status_code == 422, invalid.text
        assert invalid.json()["error"]["message"] == "Invalid or unsupported visual media"
        assert not engine.calls
        valid = client.post(
            route,
            json=wire_body(route, modality, data.getvalue() if modality == "image" else video_bytes),
            headers=HEADERS,
        )
        assert valid.status_code == 200, valid.text
        assert getattr(engine.visual, modality) is not None
        assert client.get("/load").json() == {"server_load": 0}
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("modality", ["audio", "image", "video"])
def test_chat_duplicate_media_rejected(modality):
    payload = wire_body("/v1/chat/completions", modality, b"private")
    payload["messages"][0]["content"].append(copy.deepcopy(payload["messages"][0]["content"][1]))
    engine = Engine()
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post("/v1/chat/completions", json=payload, headers=HEADERS)
        assert response.status_code == 422
        assert not engine.calls


def test_visual_decode_and_preflight_once_for_sixteen_questions(visual_processor, monkeypatch):
    from vllm_omni.entrypoints.audio_choice.media import VisualEvidence

    count = []
    original = VisualEvidence.token_expansion

    def measured(self, processor):
        count.append(1)
        return original(self, processor)

    monkeypatch.setattr(VisualEvidence, "token_expansion", measured)
    data = io.BytesIO()
    Image.new("RGB", (56, 56)).save(data, format="PNG")
    payload = wire_body("/v1/systemone", "image", data.getvalue())
    payload["questions"] = {str(i): body()["questions"]["routing"] for i in range(16)}
    engine = visual_engine(visual_processor)
    from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine

    engine.config, engine.temperature = {"model": "test"}, 1
    engine.tokenizer = visual_processor.tokenizer
    engine.prepare_lock = threading.Lock()
    engine.prompt = AsyncChoiceEngine.prompt.__get__(engine)
    engine.prepare = AsyncChoiceEngine.prepare.__get__(engine)
    engine.evaluate = AsyncChoiceEngine.evaluate.__get__(engine)

    async def score(*a):
        return 0.0, 100

    engine.score = score
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post("/v1/systemone", json=payload, headers=HEADERS)
        assert response.status_code == 200, response.text
        assert len(response.json()["answers"]) == 16
        assert count == [1]


def test_disabled_visual_http_rejects_without_decode():
    engine = visual_engine(None)
    engine.enable_vision = False
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post("/v1/systemone", json=wire_body("/v1/systemone", "image", b"private"), headers=HEADERS)
        assert response.status_code == 422 and "choice-enable-vision" in response.text
        assert not engine.calls


@pytest.mark.parametrize("log_error_stack", [False, True])
@pytest.mark.parametrize("error_type", ["client", "generate"])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_native_processor_error_is_422_over_http(
    error_type, modality, visual_processor, video_bytes, caplog, log_error_stack
):
    from vllm.exceptions import VLLMClientError
    from vllm.v1.engine.exceptions import EngineGenerateError

    from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine

    engine = visual_engine(visual_processor)
    engine.config, engine.temperature = {"model": "test"}, 1
    engine.slots = asyncio.Semaphore(1)
    engine.prepare = lambda question, *a: ([10], {key: [11] for key in question.criteria})
    engine.score = AsyncChoiceEngine.score.__get__(engine)
    engine.evaluate = AsyncChoiceEngine.evaluate.__get__(engine)
    fail = [True]
    aborted = []

    async def generate(prompt, *a):
        if fail[0]:
            raise (VLLMClientError if error_type == "client" else EngineGenerateError)(
                "PRIVATE PROCESSOR DATA"
            ) from ValueError("PRIVATE ROOT CAUSE")
        yield SimpleNamespace(prompt_token_ids=[10, 11], prompt_logprobs=[None, {11: SimpleNamespace(logprob=-0.5)}])

    async def abort(request_id):
        aborted.append(request_id)

    engine.llm = fake_llm(generate=generate, abort=abort, errored=False)
    data = io.BytesIO()
    Image.new("RGB", (56, 56)).save(data, format="PNG")
    payload = wire_body("/v1/systemone", modality, data.getvalue() if modality == "image" else video_bytes)
    app_args = args()
    app_args.log_error_stack = log_error_stack
    with TestClient(build_choice_app(app_args, engine)) as client:
        response = client.post("/v1/systemone", json=payload, headers=HEADERS)
        assert response.status_code == 422, response.text
        assert response.json()["error"]["message"] == "Invalid or unsupported model input"
        assert "Choice request failed; exception chain:" in caplog.text
        assert "ValueError" in caplog.text
        assert "PRIVATE PROCESSOR DATA" not in caplog.text and "PRIVATE ROOT CAUSE" not in caplog.text
        assert aborted
        assert client.get("/load").json() == {"server_load": 0}
        assert client.get("/health").status_code == 200
        fail[0] = False
        assert client.post("/v1/systemone", json=payload, headers=HEADERS).status_code == 200


@pytest.mark.parametrize("route", ["/v1/systemone", "/v1/chat/completions"])
def test_zero_frame_video_http_rejected(route, visual_processor, video_bytes, monkeypatch):
    from vllm.multimodal.media import MediaWithBytes, VideoMediaIO

    monkeypatch.setattr(
        VideoMediaIO, "load_bytes", lambda *a: MediaWithBytes((np.zeros((0, 28, 28, 3)), {}), b"private")
    )
    engine = visual_engine(visual_processor)
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post(route, json=wire_body(route, "video", video_bytes), headers=HEADERS)
        assert response.status_code == 422
        assert response.json()["error"]["message"] == "Invalid or unsupported visual media"
        assert not engine.calls
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("route", ["/v1/systemone", "/v1/chat/completions"])
@pytest.mark.parametrize("audio", [False, True])
@pytest.mark.parametrize("cause", [None, RuntimeError, ValueError])
def test_generate_fault_classification_on_text_and_audio(route, audio, cause):
    from vllm.v1.engine.exceptions import EngineGenerateError

    from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine

    engine = Engine()
    engine.config, engine.temperature = {"model": "test"}, 1
    engine.slots = asyncio.Semaphore(1)
    engine.prepare = lambda question, *a: ([10], {key: [11] for key in question.criteria})
    engine.score = AsyncChoiceEngine.score.__get__(engine)
    engine.evaluate = AsyncChoiceEngine.evaluate.__get__(engine)
    fail = [True]
    aborted = []

    async def generate(prompt, *a):
        if fail[0]:
            raise EngineGenerateError("PRIVATE FAILURE") from (cause("PRIVATE CAUSE") if cause else None)
        yield SimpleNamespace(prompt_token_ids=[10, 11], prompt_logprobs=[None, {11: SimpleNamespace(logprob=-0.5)}])

    async def abort(request_id):
        aborted.append(request_id)

    engine.llm = fake_llm(generate=generate, abort=abort, errored=False)
    payload = body()
    if audio:
        data = io.BytesIO()
        sf.write(data, np.zeros(1600), 16000, format="WAV")
        payload = wire_body(route, "audio", data.getvalue())
    elif route.endswith("chat/completions"):
        payload = {
            "model": "spev",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps({k: v for k, v in payload.items() if k != "model"})}
                    ],
                }
            ],
        }
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post(route, json=payload, headers=HEADERS)
        assert response.status_code == (422 if cause is ValueError else 500), response.text
        assert "PRIVATE" not in response.text
        assert aborted
        assert client.get("/load").json() == {"server_load": 0}
        assert client.get("/health").status_code == 200
        fail[0] = False
        assert client.post(route, json=payload, headers=HEADERS).status_code == 200


@pytest.mark.parametrize("route", ["/v1/systemone", "/v1/chat/completions"])
@pytest.mark.parametrize("counts", [{"audio": 2, "image": 3, "video": 4}, {"image": 3}, {}])
def test_modality_usage_details_survive_both_http_transports(route, counts):
    from vllm_omni.entrypoints.audio_choice.contract import InputTokensDetails, MultimodalTokens

    engine = Engine()
    original = engine.evaluate

    async def evaluate(*args, **kwargs):
        result = await original(*args, **kwargs)
        result.usage.input_tokens_details = InputTokensDetails(
            multimodal_tokens=MultimodalTokens(**counts) if counts else None
        )
        return result

    engine.evaluate = evaluate
    payload = body()
    if route.endswith("chat/completions"):
        payload = {
            "model": "spev",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps({k: v for k, v in payload.items() if k != "model"})}
                    ],
                }
            ],
        }
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post(route, json=payload, headers=HEADERS)
        assert response.status_code == 200, response.text
        key = "input_tokens_details" if route == "/v1/systemone" else "prompt_tokens_details"
        expected = {"cached_tokens": 0, "multimodal_tokens": counts or None}
        if route.endswith("chat/completions"):
            expected["created_cache_tokens"] = 0
        elif counts:
            expected["multimodal_tokens"] = {modality: counts.get(modality) for modality in ("audio", "image", "video")}
        assert response.json()["usage"][key] == expected


@pytest.mark.parametrize("route", ["/v1/systemone", "/v1/chat/completions"])
def test_renderer_value_error_after_engine_failure_is_redacted_503(route, caplog):
    from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine

    class FailingEngine(Engine):
        @property
        def errored(self):
            return self.llm.errored

    engine = FailingEngine()
    engine.config, engine.temperature = {"model": "test"}, 1
    engine.slots = asyncio.Semaphore(1)
    engine.prepare = lambda question, *a: ([10], {key: [11] for key in question.criteria})
    engine.score = AsyncChoiceEngine.score.__get__(engine)
    engine.evaluate = AsyncChoiceEngine.evaluate.__get__(engine)
    aborted = []

    async def render(prompts):
        engine.llm.errored = True
        raise ValueError("PRIVATE PROCESSOR DATA")

    async def abort(request_id):
        aborted.append(request_id)

    engine.llm = fake_llm(abort=abort)
    engine.llm.renderer.render_cmpl_async = render
    payload = body()
    if route.endswith("chat/completions"):
        payload = {
            "model": "spev",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps({k: v for k, v in payload.items() if k != "model"})}
                    ],
                }
            ],
        }
    with TestClient(build_choice_app(args(), engine)) as client:
        response = client.post(route, json=payload, headers=HEADERS)
        assert response.status_code == 503, response.text
        assert response.json()["error"]["message"] == "Choice engine unavailable"
        assert "PRIVATE PROCESSOR DATA" not in response.text
        assert "PRIVATE PROCESSOR DATA" not in caplog.text
        assert "ValueError" in caplog.text
        assert "Choice request failed; exception chain:" in caplog.text
        assert "Choice input rejected" not in caplog.text
        assert aborted
        assert client.get("/load").json() == {"server_load": 0}
        assert client.post(route, json=payload, headers=HEADERS).status_code == 503
