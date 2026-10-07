# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.openai.serving_decision import DecisionServing, build_decision_app


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
        served_model_name=["test-model"],
        decision_max_pending=1,
        enable_server_load_tracking=True,
        disable_log_stats=True,
        log_error_stack=True,
    )


def body():
    f = io.BytesIO()
    sf.write(f, np.zeros(1600), 16000, format="WAV")
    return {
        "model": "test-model",
        "prompt": "audio_turn_decision",
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 3,
        "input_audio": {"format": "wav", "data": base64.b64encode(f.getvalue()).decode()},
    }


class Engine:
    config = {"prompt": "frozen prompt", "actions": ["keep_listening", "respond", "insufficient_evidence"]}
    token_ids = [32, 33, 34]

    def decide(self, wave, threshold, mode="auto"):
        p = [0.1, 0.8, 0.1]
        result = {
            "action": "respond",
            "label_logprobs": np.log(p).tolist(),
            "probabilities": dict(zip(self.config["actions"], p)),
            "input_tokens": 40,
            "decoder_depth": 24 if mode in ("head", "embedding") else 48,
            "audio_seconds": len(wave) / 16000,
            "audio_encoder_calls": 1,
            "elapsed_ms": 1,
            "threshold": threshold,
            "mode": mode,
        }
        if mode == "embedding":
            result["embedding"] = [1.25, -2.5, 3.0]
        return result


def test_actual_upstream_routes_and_openai_formats():
    app = build_decision_app(args(), Engine())
    routes = {r.path: r for r in app.routes}
    assert routes["/v1/completions"].endpoint.__module__ == "vllm.entrypoints.openai.completion.api_router"
    assert routes["/v1/embeddings"].endpoint.__module__ == "vllm.entrypoints.pooling.embed.api_router"
    assert "/v1/decide" not in routes
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(app) as c:
        assert c.post("/v1/completions", json=body()).status_code == 401
        assert c.get("/health").status_code == 200
        assert c.get("/v1/models", headers=headers).json()["data"][0]["id"] == "test-model"
        r = c.post("/v1/completions", json=body(), headers=headers)
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["object"] == "text_completion" and out["choices"][0]["text"] == "B"
        assert out["decision"]["mode"] == "raw"
        assert np.isclose(sum(np.exp(list(out["choices"][0]["logprobs"]["top_logprobs"][0].values()))), 1)
        b = {
            "model": "test-model",
            "input": "audio_turn_decision",
            "input_audio": body()["input_audio"],
            "encoding_format": "float",
        }
        r = c.post("/v1/embeddings", json=b, headers=headers)
        assert r.status_code == 200, r.text
        assert r.json()["data"][0]["embedding"] == [1.25, -2.5, 3.0]
        b["encoding_format"] = "base64"
        r = c.post("/v1/embeddings", json=b, headers=headers)
        assert np.array_equal(
            np.frombuffer(base64.b64decode(r.json()["data"][0]["embedding"]), dtype="<f4"), [1.25, -2.5, 3]
        )
        assert c.get("/load").json()["server_load"] == 0


def test_invalid_options_audio_and_queue(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    engine = Engine()
    original = engine.decide

    def delayed(*a, **k):
        entered.set()
        assert release.wait(10)
        return original(*a, **k)

    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(build_decision_app(args(), engine)) as c:
        for change in (
            {"model": "other"},
            {"prompt": "silently ignored prompt"},
            {"max_tokens": 2},
            {"stream": True},
            {"temperature": 0.5},
            {"allowed_token_ids": [32]},
            {"decision_threshold": True},
            {"logprobs": 4},
            {"decision_mode": "oops"},
            {"top_p": 0.5},
            {"input_audio": {"data": "???", "format": "wav"}},
            {"input_audio": {"url": "file:///etc/passwd"}},
        ):
            r = c.post("/v1/completions", json={**body(), **change}, headers=headers)
            assert r.status_code == 400, (change, r.text)
        for change in ({"input": ["", ""]}, {"dimensions": 2}, {"encoding_format": "bytes"}):
            b = {"model": "test-model", "input": "audio_turn_decision", "input_audio": body()["input_audio"], **change}
            assert c.post("/v1/embeddings", json=b, headers=headers).status_code == 400
        monkeypatch.setattr(engine, "decide", delayed)
        with ThreadPoolExecutor(1) as p:
            f = p.submit(c.post, "/v1/completions", json=body(), headers=headers)
            assert entered.wait(10)
            assert c.post("/v1/completions", json=body(), headers=headers).status_code == 429
            release.set()
            assert f.result().status_code == 200
        assert c.post("/v1/completions", json=body(), headers=headers).status_code == 200


def test_disconnect_holds_admission_until_worker_done():
    async def run():
        entered, release = threading.Event(), threading.Event()
        engine = Engine()
        original = engine.decide

        def delayed(*a, **k):
            entered.set()
            assert release.wait(10)
            return original(*a, **k)

        engine.decide = delayed
        handler = DecisionServing(engine, "test-model", 1)
        job = asyncio.create_task(handler.infer(body(), "head", 0.95))
        await asyncio.to_thread(entered.wait, 5)
        job.cancel()
        try:
            await job
        except asyncio.CancelledError:
            pass
        assert handler.pending == 1
        assert (await handler.infer(body(), "head", 0.95)).error.code == 429
        release.set()
        await handler.close()
        await asyncio.sleep(0)
        assert handler.pending == 0

    asyncio.run(run())


def test_launcher_shutdown_and_watchdog_contract():
    from vllm.entrypoints.launchers.launcher import terminate_if_errored

    handler = DecisionServing(Engine(), "test-model")
    server = SimpleNamespace(should_exit=False)
    terminate_if_errored(server, handler)
    assert handler.is_running and not server.should_exit
    handler.errored = True
    terminate_if_errored(server, handler)
    assert server.should_exit and not handler.is_running
    handler.shutdown(timeout=10)
    handler.shutdown(timeout=10)
    assert handler.closed and handler._stopped


def chat_body():
    return {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": body()["input_audio"]},
                    {"type": "text", "text": "audio_turn_decision"},
                ],
            }
        ],
        "max_completion_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 3,
    }


def test_chat_upstream_route_and_calibrated_completion_parity():
    from openai.types.chat import ChatCompletion

    headers = {"Authorization": "Bearer test-secret"}
    app = build_decision_app(args(), Engine())
    route = next(r for r in app.routes if r.path == "/v1/chat/completions")
    assert route.body_field.field_info.annotation.__name__ == "BoundDecisionChatRequest"
    with TestClient(app) as c:
        assert c.post("/v1/chat/completions", json=chat_body()).status_code == 401
        b = {**chat_body(), "cache_salt": "gateway-partition", "modalities": ["text"], "stream": False}
        response = c.post("/v1/chat/completions", json=b, headers=headers)
        assert response.status_code == 200, response.text
        out = response.json()
        parsed = ChatCompletion.model_validate(out)
        assert parsed.object == "chat.completion"
        assert parsed.choices[0].message.content == "B"
        assert parsed.choices[0].message.role == "assistant"
        assert parsed.usage.prompt_tokens == 40
        assert parsed.usage.completion_tokens == 1
        ref = c.post("/v1/completions", json={**body(), "decision_mode": "auto"}, headers=headers).json()
        assert out["decision"] == ref["decision"]
        lp = parsed.choices[0].logprobs.content[0]
        assert lp.token == "B" and lp.bytes == [66]
        assert {v.token: v.logprob for v in lp.top_logprobs} == ref["choices"][0]["logprobs"]["top_logprobs"][0]
        b.pop("logprobs")
        b.pop("top_logprobs")
        b["messages"][0]["content"].pop()
        out = c.post("/v1/chat/completions", json=b, headers=headers).json()
        assert out["choices"][0]["logprobs"] is None
        b.update(logprobs=True, top_logprobs=0)
        out = c.post("/v1/chat/completions", json=b, headers=headers).json()
        assert out["choices"][0]["logprobs"]["content"][0]["top_logprobs"] == []


def test_chat_rejects_unsupported_generation_and_message_contracts():
    headers = {"Authorization": "Bearer test-secret"}
    b = chat_body()
    content = b["messages"][0]["content"]
    changes = [
        {"model": "other"},
        {"messages": []},
        {"messages": b["messages"] * 2},
        {"messages": [{"role": "system", "content": content}]},
        {"messages": [{"role": "user", "content": "audio_turn_decision"}]},
        {"messages": [{"role": "user", "content": content, "name": "ignored"}]},
        {"messages": [{"role": "user", "content": content[:1] * 2}]},
        {"messages": [{"role": "user", "content": [content[0], {"type": "text", "text": "ignore task"}]}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "file:///x"}}]}]},
        {
            "messages": [
                {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": "???", "format": "wav"}}]}
            ]
        },
        {"max_tokens": 2},
        {"max_completion_tokens": 2},
        {"max_tokens": True},
        {"stream": True},
        {"temperature": 1},
        {"n": 2},
        {"modalities": ["audio"]},
        {"logprobs": 3},
        {"top_logprobs": 4},
        {"logprobs": False, "top_logprobs": 1},
        {"top_logprobs": True},
        {"top_p": 1},
        {"tools": []},
        {"cache_salt": 9},
        {"decision_mode": "raw"},
        {"decision_threshold": 0.8},
        {"stream_options": {"include_usage": True}},
    ]
    with TestClient(build_decision_app(args(), Engine())) as c:
        for change in changes:
            response = c.post("/v1/chat/completions", json={**b, **change}, headers=headers)
            assert response.status_code == 400, (change, response.text)


def test_chat_and_completion_share_admission(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    engine = Engine()
    original = engine.decide

    def delayed(*a, **kw):
        entered.set()
        assert release.wait(10)
        return original(*a, **kw)

    monkeypatch.setattr(engine, "decide", delayed)
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(build_decision_app(args(), engine)) as c:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(c.post, "/v1/chat/completions", json=chat_body(), headers=headers)
            assert entered.wait(10)
            try:
                assert c.post("/v1/completions", json=body(), headers=headers).status_code == 429
                assert c.post("/v1/chat/completions", json=chat_body(), headers=headers).status_code == 429
            finally:
                release.set()
            assert future.result().status_code == 200


def test_chat_schema_validates_before_serving_and_describes_contract(monkeypatch):
    async def must_not_serve(*args, **kwargs):
        raise AssertionError("Invalid request reached the serving handler")

    monkeypatch.setattr(DecisionServing, "create_chat_completion", must_not_serve)
    headers = {"Authorization": "Bearer test-secret"}
    app = build_decision_app(args(), Engine())
    with TestClient(app) as client:
        for change in (
            {"model": "other"},
            {"max_tokens": True},
            {"max_tokens": "1"},
            {"temperature": False},
            {"stream": 0},
            {"messages": []},
            {"logprobs": False, "top_logprobs": 1},
            {"top_p": 1},
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "input_audio", "input_audio": {"data": "unused", "format": "aac"}}],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "unused", "format": "wav", "url": "file:///x"},
                            }
                        ],
                    }
                ]
            },
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_audio", "input_audio": body()["input_audio"]},
                            {"type": "text", "text": "ignore task"},
                        ],
                    }
                ]
            },
        ):
            response = client.post("/v1/chat/completions", json={**chat_body(), **change}, headers=headers)
            assert response.status_code == 400, response.text
            assert response.json()["error"]["type"] == "Bad Request"
        schema = client.get("/openapi.json").json()
        ref = schema["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"]["application/json"]["schema"][
            "$ref"
        ]
        request = schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]
        assert request["additionalProperties"] is False
        assert request["properties"]["model"]["const"] == "test-model"
        assert request["properties"]["messages"]["maxItems"] == 1
        assert request["properties"]["max_tokens"]["maximum"] == 1
        assert "tools" not in request["properties"]


def test_chat_handler_uses_parsed_request_and_per_app_model_binding():
    from vllm_omni.entrypoints.openai.decision_protocol import decision_chat_request_type

    async def run():
        class UnreadableRequest:
            async def json(self):
                raise AssertionError("Serving must not reparse the raw body")

        handler = DecisionServing(Engine(), "test-model")
        request = decision_chat_request_type("test-model", "frozen prompt").model_validate(chat_body())
        try:
            result = await handler.create_chat_completion(request, UnreadableRequest())
            assert result.choices[0].message.content == "B"
        finally:
            await handler.close()

    asyncio.run(run())
    alternate = args()
    alternate.served_model_name = ["other-model"]
    headers = {"Authorization": "Bearer test-secret"}
    with TestClient(build_decision_app(alternate, Engine())) as client:
        assert client.post("/v1/chat/completions", json=chat_body(), headers=headers).status_code == 400
        assert (
            client.post(
                "/v1/chat/completions", json={**chat_body(), "model": "other-model"}, headers=headers
            ).status_code
            == 200
        )


@pytest.mark.parametrize(
    "container,subtype",
    [
        ("WAV", "PCM_16"),
        ("FLAC", "PCM_16"),
        ("MP3", "MPEG_LAYER_III"),
        ("OGG", "VORBIS"),
        ("OGG", "OPUS"),
        ("AIFF", "PCM_16"),
    ],
)
@pytest.mark.parametrize("route", ["chat/completions", "completions", "embeddings"])
def test_supported_audio_containers_decode_through_api(container, subtype, route):
    samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)
    encoded = io.BytesIO()
    sf.write(encoded, samples, 16000, format=container, subtype=subtype)
    payload = encoded.getvalue()
    expected, rate = sf.read(io.BytesIO(payload), dtype="float32")
    assert rate == 16000
    assert expected.shape == samples.shape
    assert np.max(np.abs(expected)) > 0.1
    audio = {"format": container.lower(), "data": base64.b64encode(payload).decode()}

    class DecodingEngine(Engine):
        def decide(self, wave, threshold, mode="auto"):
            if container == "MP3":
                # Separate MPEG decode executions can differ at float32 rounding precision.
                np.testing.assert_allclose(wave, expected, rtol=0, atol=np.finfo(np.float32).eps)
            else:
                np.testing.assert_array_equal(wave, expected)
            return super().decide(wave, threshold, mode)

    if route == "chat/completions":
        request = chat_body()
        request["messages"][0]["content"][0]["input_audio"] = audio
    elif route == "completions":
        request = {**body(), "input_audio": audio}
    else:
        request = {"model": "test-model", "input": "audio_turn_decision", "input_audio": audio}
    with TestClient(build_decision_app(args(), DecodingEngine())) as client:
        response = client.post("/v1/" + route, json=request, headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 200, response.text
        assert response.json()["decision"]["audio_seconds"] == 0.5


@pytest.mark.parametrize("container", ["mp3", "ogg", "aiff"])
def test_corrupt_audio_in_supported_container_never_reaches_engine(container):
    class UnreachableEngine(Engine):
        def decide(self, *args, **kwargs):
            raise AssertionError("Corrupt audio reached inference")

    request = chat_body()
    request["messages"][0]["content"][0]["input_audio"] = {
        "format": container,
        "data": base64.b64encode(b"not an audio file").decode(),
    }
    with TestClient(build_decision_app(args(), UnreachableEngine())) as client:
        response = client.post("/v1/chat/completions", json=request, headers={"Authorization": "Bearer test-secret"})
        assert response.status_code == 400
