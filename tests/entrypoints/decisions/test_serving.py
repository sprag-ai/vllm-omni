# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import threading
from dataclasses import dataclass

import pytest
import torch
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from vllm import PoolingParams
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.v1.engine.exceptions import EngineGenerateError

from vllm_omni.entrypoints.openai.serving_decisions import DecisionsServing, mount_decisions
from vllm_omni.inputs.decisions import DecisionsRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def payload():
    return {
        "model": "unrelated-model",
        "input": "evidence",
        "questions": [{"type": "predicate", "instructions": "Is it true?", "name": "condition"}],
    }


class AlternateProcessor:
    supports_decisions = True

    def parse_data(self, data):
        return data

    def pre_process(self, prompt, request_id):
        return {"prompt_token_ids": [9, 8, 7]}

    def merge_pooling_params(self):
        return PoolingParams(task="plugin")

    def prepare_pooling_params(self, prompt):
        return self.merge_pooling_params()

    def post_process(self, output, *, prompt, request_id):
        assert prompt.questions[0].name == "condition"
        return [{"type": "predicate", "name": "condition", "probability": 0.75}]


class Renderer:
    async def render_cmpl_async(self, prompts):
        return prompts


@dataclass
class ModelConfigStub:
    io_processor_plugin: str | None = None


class Engine:
    errored = False
    renderer = Renderer()
    vllm_config = None
    model_config = ModelConfigStub()

    def __init__(self):
        self.calls = 0
        self.failure = None
        self.entered = asyncio.Event()
        self.release = None

    async def get_supported_tasks(self):
        return ("plugin",)

    async def encode(self, prompt, params, request_id):
        self.calls += 1
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.failure is not None:
            raise self.failure
        yield PoolingRequestOutput(
            request_id=request_id,
            outputs=PoolingOutput(torch.tensor([1.0])),
            prompt_token_ids=prompt["prompt_token_ids"],
            num_cached_tokens=0,
            finished=True,
        )


def test_model_independent_processor_uses_encode_and_native_usage():
    async def run():
        engine = Engine()
        serving = DecisionsServing(engine, AlternateProcessor(), ["unrelated-model"], 2)
        result = await serving.evaluate(DecisionsRequest.model_validate(payload()))
        assert result.answers[0].probability == 0.75
        assert result.usage.input_tokens == result.usage.total_tokens == 3
        assert result.usage.output_tokens == 0
        assert engine.calls == 1

    asyncio.run(run())


@pytest.mark.parametrize("modalities", [[], ["audio"], ["image"], ["video"], ["audio", "image", "video"]])
def test_http_usage_preserves_upstream_multimodal_counts(monkeypatch, modalities):
    from vllm.entrypoints.openai.chat_completion.serving import _get_mm_token_counts
    from vllm.multimodal.inputs import PlaceholderRange

    from vllm_omni.entrypoints.openai import serving_decisions

    engine_input = {
        "prompt_token_ids": list(range(64)),
        "mm_placeholders": {
            name: [PlaceholderRange(offset=index * 12, length=3), PlaceholderRange(offset=index * 12 + 3, length=5)]
            for index, name in enumerate(modalities)
        },
    }
    counted = []
    assert serving_decisions._get_mm_token_counts is _get_mm_token_counts

    def count(prompt):
        counted.append(prompt)
        return _get_mm_token_counts(prompt)

    class NativeRenderer(Renderer):
        async def render_cmpl_async(self, prompts):
            return [engine_input]

    monkeypatch.setattr(serving_decisions, "_get_mm_token_counts", count)
    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *args: AlternateProcessor())
    engine = Engine()
    engine.renderer = NativeRenderer()
    app = FastAPI()
    mount_decisions(app, engine, Args())
    with TestClient(app) as client:
        response = client.post("/v1/decisions", json=payload())
        assert response.status_code == 200
        usage = response.json()["usage"]
        assert counted == [engine_input]
        assert usage["input_tokens_details"]["multimodal_tokens"] == (_get_mm_token_counts(engine_input) or None)
        assert usage["input_tokens"] == usage["total_tokens"] == 64
        assert usage["output_tokens"] == 0


@pytest.mark.parametrize("dead,client,status", [(False, True, 422), (False, False, 500), (True, True, 503)])
def test_native_failures_are_redacted_and_do_not_latch(dead, client, status):
    async def run():
        engine = Engine()
        engine.errored = dead
        error = EngineGenerateError()
        error.__cause__ = ValueError("private-media") if client else RuntimeError("private-media")
        engine.failure = error
        serving = DecisionsServing(engine, AlternateProcessor(), ["unrelated-model"], 2)
        with pytest.raises(HTTPException) as exc:
            await serving.evaluate(DecisionsRequest.model_validate(payload()))
        assert exc.value.status_code == status
        assert "private-media" not in str(exc.value)
        engine.errored = False
        engine.failure = None
        await serving.evaluate(DecisionsRequest.model_validate(payload()))

    asyncio.run(run())


def test_disconnected_waiter_keeps_admission_until_work_finishes():
    async def run():
        engine = Engine()
        engine.release = asyncio.Event()
        serving = DecisionsServing(engine, AlternateProcessor(), ["unrelated-model"], 1)
        request = DecisionsRequest.model_validate(payload())
        task = asyncio.create_task(serving.evaluate(request))
        await engine.entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(HTTPException) as exc:
            await serving.evaluate(request)
        assert exc.value.status_code == 429
        engine.release.set()
        await serving.close()
        assert not serving.active

    asyncio.run(run())


@dataclass
class Args:
    model: str = "unrelated-model"
    served_model_name: list[str] | None = None
    decision_max_pending: int = 2


def test_http_route_uses_configured_processor_and_redacts_validation(monkeypatch):
    from vllm_omni.entrypoints.openai import serving_decisions

    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *a: AlternateProcessor())
    app = FastAPI()
    mount_decisions(app, Engine(), Args())
    with TestClient(app) as client:
        response = client.post("/v1/decisions", json=payload())
        assert response.status_code == 200
        assert response.json()["answers"][0]["probability"] == 0.75
        bad = payload()
        bad["questions"][0]["type"] = "private-data"
        response = client.post("/v1/decisions", json=bad)
        assert response.status_code == 422
        assert "private-data" not in response.text
        bad = payload()
        bad["model"] = "other"
        assert client.post("/v1/decisions", json=bad).status_code == 404


def test_ordinary_pooling_processor_does_not_gain_decisions_route(monkeypatch):
    from vllm_omni.entrypoints.openai import serving_decisions

    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *a: None)
    app = FastAPI()
    assert mount_decisions(app, Engine(), Args()) is None
    assert all(getattr(r, "path", None) != "/v1/decisions" for r in app.routes)


def test_load_tracking_counts_work_after_waiter_disconnects():
    from starlette.requests import Request

    async def run():
        app = FastAPI()
        app.state.enable_server_load_tracking = True
        app.state.server_load_metrics = 0
        raw = Request({"type": "http", "app": app})
        engine = Engine()
        engine.release = asyncio.Event()
        serving = DecisionsServing(engine, AlternateProcessor(), ["unrelated-model"], 1)
        task = asyncio.create_task(serving.evaluate(DecisionsRequest.model_validate(payload()), raw))
        await engine.entered.wait()
        assert app.state.server_load_metrics == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert app.state.server_load_metrics == 1
        engine.release.set()
        await serving.close()
        assert app.state.server_load_metrics == 0

    asyncio.run(run())


def test_other_routes_keep_upstream_validation_handler(monkeypatch):
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    from vllm_omni.entrypoints.openai import serving_decisions

    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *a: AlternateProcessor())
    app = FastAPI()

    @app.exception_handler(RequestValidationError)
    async def upstream(request, exc):
        return JSONResponse(status_code=400, content={"error": "upstream"})

    @app.get("/other")
    def other(count: int):
        return count

    mount_decisions(app, Engine(), Args())
    with TestClient(app) as client:
        result = client.get("/other?count=invalid")
        assert result.status_code == 400
        assert result.json() == {"error": "upstream"}


@pytest.mark.parametrize("decoder_error", [None, OSError("private decoder data")])
def test_corrupt_image_http_rejection_and_recovery(monkeypatch, decoder_error):
    from tests.entrypoints.decisions.test_spev import spec
    from vllm_omni.entrypoints.openai import serving_decisions
    from vllm_omni.model_executor.models.spev.processor import SPEVProcessor

    processor = object.__new__(SPEVProcessor)
    processor.config = spec()
    # Invalid media must fail before reaching the tokenizer or model.
    if decoder_error is not None:

        def fail_decode(self, data):
            raise decoder_error

        monkeypatch.setattr("vllm_omni.model_executor.models.spev.processor.ImageMediaIO.load_bytes", fail_decode)
    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *a: processor)
    engine = Engine()
    app = FastAPI()
    serving = mount_decisions(app, engine, Args())
    with TestClient(app) as client:
        bad = payload()
        bad["input"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,cHJpdmF0ZQ==",
                    }
                ],
            }
        ]
        response = client.post("/v1/decisions", json=bad)
        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid or unsupported model input"}
        assert engine.calls == 0
        assert not engine.errored
        serving.processor = AlternateProcessor()
        assert client.post("/v1/decisions", json=payload()).status_code == 200


@pytest.mark.parametrize("format_name,codec", [("mp4", "mpeg4"), ("matroska", "mpeg4"), ("webm", "libvpx-vp9")])
def test_video_http_forwarding_and_native_token_accounting(monkeypatch, video_factory, format_name, codec):
    from types import SimpleNamespace

    import pybase64
    from vllm.multimodal.inputs import PlaceholderRange

    from tests.entrypoints.decisions.test_spev import spec
    from vllm_omni.entrypoints.openai import serving_decisions
    from vllm_omni.model_executor.models.spev.processor import SPEVProcessor

    processor = object.__new__(SPEVProcessor)
    processor.config = spec()
    processor.lock = threading.Lock()
    processor.processor = SimpleNamespace(
        video_processor=SimpleNamespace(sample_frames=lambda metadata, fps: [0, 4]),
        apply_chat_template=lambda *a, **k: "video prompt",
        tokenizer=SimpleNamespace(encode=lambda *a, **k: list(range(12))),
    )

    def post_process(output, *, prompt, request_id):
        assert prompt.request.questions[0].name == "condition"
        return [{"type": "predicate", "name": "condition", "probability": 0.75}]

    processor.post_process = post_process

    class VideoRenderer(Renderer):
        async def render_cmpl_async(self, prompts):
            (prompt,) = prompts
            assert set(prompt["multi_modal_data"]) == {"video"}
            return [{**prompt, "mm_placeholders": {"video": [PlaceholderRange(offset=1, length=4)]}}]

    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *a: processor)
    engine = Engine()
    engine.renderer = VideoRenderer()
    app = FastAPI()
    mount_decisions(app, engine, Args())
    encoded = pybase64.b64encode(video_factory(format_name, codec)).decode()
    with TestClient(app) as client:
        response = client.post(
            "/v1/decisions", json={**payload(), "input_video": {"data": encoded, "format": format_name}}
        )
        assert response.status_code == 200, response.text
        usage = response.json()["usage"]
        assert usage["input_tokens"] == 12
        assert usage["input_tokens_details"]["multimodal_tokens"] == {"video": 4}
        assert engine.calls == 1


@pytest.mark.parametrize("failure", ["corrupt", "audio_only", "ffmpeg", "os", "truncated"])
def test_video_http_rejection_and_recovery(monkeypatch, video_bytes, video_factory, audio_only_mp4, failure):
    from types import SimpleNamespace

    import av
    import pybase64

    from tests.entrypoints.decisions.test_spev import spec
    from vllm_omni.entrypoints.openai import serving_decisions
    from vllm_omni.model_executor.models.spev.processor import SPEVProcessor

    processor = object.__new__(SPEVProcessor)
    processor.config = spec()
    if failure in {"ffmpeg", "os"}:
        error = av.FFmpegError(1, "private decoder data") if failure == "ffmpeg" else OSError("private decoder data")

        def fail(*args, **kwargs):
            raise error

        monkeypatch.setattr("transformers.video_utils.load_video", fail)
    data = b"private" if failure == "corrupt" else audio_only_mp4 if failure == "audio_only" else video_bytes
    if failure == "truncated":
        data = video_factory()
        data = data[: len(data) // 2]
        processor.processor = SimpleNamespace(
            video_processor=SimpleNamespace(sample_frames=lambda metadata, fps: [0, 4])
        )
    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *a: processor)
    engine = Engine()
    app = FastAPI()
    serving = mount_decisions(app, engine, Args())
    with TestClient(app) as client:
        bad = {**payload(), "input_video": {"data": pybase64.b64encode(data).decode(), "format": "mp4"}}
        response = client.post("/v1/decisions", json=bad)
        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid or unsupported model input"}
        assert engine.calls == 0 and not engine.errored
        serving.processor = AlternateProcessor()
        assert client.post("/v1/decisions", json=payload()).status_code == 200


def test_pooling_launcher_enforces_body_limit(monkeypatch):
    from argparse import Namespace
    from contextlib import asynccontextmanager
    from unittest.mock import Mock

    import httpx
    from vllm.entrypoints.launchers import app as upstream_app
    from vllm.entrypoints.launchers import launcher
    from vllm.entrypoints.launchers.api_server import app_state, entry
    from vllm.entrypoints.launchers.utils import server_utils

    from vllm_omni.entrypoints.openai import serving_decisions

    engine = Engine()

    @asynccontextmanager
    async def build_engine(args):
        yield engine

    async def init_state(*args):
        pass

    async def serve(app, **kwargs):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/decisions", content=b"not-json" * 64)
            assert response.status_code == 413
            assert engine.calls == 0
            assert (await client.post("/v1/decisions", json=payload())).status_code == 200

        async def shutdown():
            pass

        return shutdown()

    monkeypatch.setattr(entry, "build_async_engine_client", build_engine)
    monkeypatch.setattr(upstream_app, "build_app", lambda *args: FastAPI())
    monkeypatch.setattr(app_state, "init_app_state", init_state)
    monkeypatch.setattr(server_utils, "get_uvicorn_log_config", lambda args: None)
    monkeypatch.setattr(launcher, "serve_http", serve)
    monkeypatch.setattr(serving_decisions, "get_io_processor", lambda *args: AlternateProcessor())
    args = Namespace(
        runner="pooling",
        model="unrelated-model",
        served_model_name=None,
        decision_max_pending=2,
        max_request_body_bytes=256,
        host="127.0.0.1",
        port=8000,
        enable_ssl_refresh=False,
        uvicorn_log_level="info",
        disable_uvicorn_access_log=True,
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        ssl_ciphers=None,
        h11_max_incomplete_event_size=65536,
        h11_max_header_count=256,
    )
    sock = Mock()
    asyncio.run(serving_decisions.run_pooling_server(args, sock))
    sock.close.assert_called_once()
