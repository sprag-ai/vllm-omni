# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import pytest
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from vllm_omni.entrypoints.openai.websocket_sessions import WebSocketSessionGauge
from vllm_omni.metrics import definitions as defs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _open_sessions() -> float:
    return REGISTRY.get_sample_value(defs.WEBSOCKET_SESSIONS_ACTIVE)


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        while True:
            try:
                message = await websocket.receive_text()
            except WebSocketDisconnect:
                return
            if message == "fail":
                raise RuntimeError("handler failed")
            await websocket.send_text(str(_open_sessions()))

    @app.get("/health")
    async def health() -> dict:
        return {"open": _open_sessions()}

    app.add_middleware(WebSocketSessionGauge)
    return TestClient(app)


def test_counts_each_open_session(client: TestClient) -> None:
    baseline = _open_sessions()
    with client.websocket_connect("/v1/realtime") as first:
        first.send_text("count")
        assert float(first.receive_text()) == baseline + 1
        with client.websocket_connect("/v1/realtime") as second:
            second.send_text("count")
            assert float(second.receive_text()) == baseline + 2
        first.send_text("count")
        assert float(first.receive_text()) == baseline + 1
    assert _open_sessions() == baseline


def test_releases_a_session_whose_handler_raises(client: TestClient) -> None:
    baseline = _open_sessions()
    with pytest.raises(RuntimeError), client.websocket_connect("/v1/realtime") as ws:
        ws.send_text("fail")
        ws.receive_text()
    assert _open_sessions() == baseline


def test_http_requests_are_not_sessions(client: TestClient) -> None:
    assert client.get("/health").json()["open"] == _open_sessions()
