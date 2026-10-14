# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
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
