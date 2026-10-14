# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import Message

from vllm_omni.entrypoints.openai.body_limit import BodyLimitMiddleware
from vllm_omni.inputs.decisions import DecisionsRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("headers", [[], [(b"content-length", b"1")], [(b"content-length", b"invalid")]])
def test_chunked_body_limit_before_app_even_with_understated_length(headers):
    async def run():
        called = False
        chunks = iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"56789", "more_body": False},
            ]
        )
        messages = []

        async def app(scope, receive, send):
            nonlocal called
            called = True

        async def receive():
            return next(chunks)

        async def send(message):
            messages.append(message)

        await BodyLimitMiddleware(app, 8)({"type": "http", "method": "POST", "headers": headers}, receive, send)
        assert not called
        assert messages[0]["status"] == 413

    asyncio.run(run())


def test_declared_oversize_is_rejected_without_reading_body():
    async def run():
        async def unexpected(*args):
            pytest.fail("Oversized request reached body parsing")

        sent = []

        async def send(message):
            sent.append(message)

        await BodyLimitMiddleware(unexpected, 8)(
            {"type": "http", "method": "POST", "headers": [(b"content-length", b"9")]}, unexpected, send
        )
        assert sent[0]["status"] == 413

    asyncio.run(run())


def test_exact_limit_replayed_and_disconnect_not_dispatched():
    async def run(disconnect):
        last: Message = {"type": "http.disconnect"}
        if not disconnect:
            last = {"type": "http.request", "body": b"5678", "more_body": False}
        messages: Iterator[Message] = iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                last,
            ]
        )
        called = False

        async def receive():
            return next(messages)

        async def send(message):
            pass

        async def app(scope, receive, send):
            nonlocal called
            called = True
            assert await receive() == {"type": "http.request", "body": b"12345678", "more_body": False}

        await BodyLimitMiddleware(app, 8)({"type": "http", "method": "POST"}, receive, send)
        assert called is not disconnect

    asyncio.run(run(False))
    asyncio.run(run(True))


def test_http_limit_precedes_schema_parsing_and_health_recovers():
    app = FastAPI()

    @app.post("/v1/decisions")
    def decisions(request: DecisionsRequest):
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    with TestClient(BodyLimitMiddleware(app, 256)) as client:
        assert client.post("/v1/decisions", content=b"not-json" * 64).status_code == 413
        assert client.post("/v1/decisions", json={}).status_code == 422
        assert client.get("/health").status_code == 200
        assert (
            client.post(
                "/v1/decisions",
                json={
                    "model": "other",
                    "input": "text",
                    "questions": [{"type": "predicate", "instructions": "True?"}],
                },
            ).status_code
            == 200
        )
