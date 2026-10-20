# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Open WebSocket session count, exported for shutdown draining.

A realtime or streaming session holds its WebSocket for its whole lifetime, often idle between
turns, so the request gauges read zero while sessions are still open. A pod draining before
shutdown waits on this gauge instead.
"""

from prometheus_client import Gauge
from starlette.types import ASGIApp, Receive, Scope, Send

from vllm_omni.metrics import definitions as defs

WEBSOCKET_SESSIONS = Gauge(
    defs.WEBSOCKET_SESSIONS_ACTIVE,
    "WebSocket connections currently open on this server, across every route.",
    multiprocess_mode="livesum",
)


class WebSocketSessionGauge:
    """Pure-ASGI middleware holding WEBSOCKET_SESSIONS up for each open WebSocket connection."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "websocket":
            await self.app(scope, receive, send)
            return
        WEBSOCKET_SESSIONS.inc()
        try:
            await self.app(scope, receive, send)
        finally:
            WEBSOCKET_SESSIONS.dec()
