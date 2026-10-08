# SPDX-License-Identifier: Apache-2.0
"""Opt-in Choice API using the existing vLLM HTTP server and middleware."""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import create_model
from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.serve.engine.protocol import ModelCard, ModelList

from vllm_omni.entrypoints.audio_choice.contract import ChoiceResponse
from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceChatRequest, ChoiceRequest
from vllm_omni.entrypoints.openai.serving_decision import BodyLimitMiddleware, DecisionServing, parse_audio


class ChoiceServing(DecisionServing):
    async def show_available_models(self):
        return ModelList(data=[ModelCard(id=self.model_name, root=self.model_name, max_model_len=8192)])

    async def evaluate(self, request: ChoiceRequest) -> ChoiceResponse:
        if self.closed or self.errored:
            raise HTTPException(503, "Choice engine unavailable")
        if self.pending >= self.max_pending:
            raise HTTPException(429, "Choice queue is full", headers={"Retry-After": "1"})
        self.pending += 1
        self.loop = asyncio.get_running_loop()

        async def run():
            wave = None
            if request.input_audio is not None:
                wave = await self.loop.run_in_executor(
                    self.pool, parse_audio, {"input_audio": request.input_audio.model_dump()}
                )
            return await self.engine.evaluate(request, wave)

        task = asyncio.create_task(run())
        self.active.add(task)

        def done(future):
            self.pending -= 1
            self.active.discard(future)
            if not future.cancelled():
                failure = future.exception()
                if failure is not None and not isinstance(failure, ValueError):
                    self.errored = True

        task.add_done_callback(done)
        try:
            # Keep admission until decoding/inference actually completes, even if
            # the HTTP waiter disconnects. Shutdown drains these tracked tasks.
            return await asyncio.shield(task)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


def build_choice_app(args, engine):
    names = getattr(args, "served_model_name", None) or ["spev"]
    if len(names) != 1:
        raise ValueError("Choice requires exactly one served model name")
    app = build_app(args, ("generate",))
    keep = {"/health", "/version", "/v1/models", "/load", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") in keep]
    handler = ChoiceServing(engine, names[0], getattr(args, "decision_max_pending", 16))
    request_type = create_model("BoundChoiceRequest", __base__=ChoiceRequest, model=(Literal[names[0]], ...))
    chat_type = create_model("BoundChoiceChatRequest", __base__=ChoiceChatRequest, model=(Literal[names[0]], ...))

    async def system_one(request: request_type) -> ChoiceResponse:
        return await handler.evaluate(request)

    async def chat(request: chat_type):
        result = await handler.evaluate(request.to_choice())
        # Compatibility transport only: native /v1/systemone returns the typed
        # result directly; chat clients receive the same result as JSON content.
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.model_dump_json()},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.usage.input_tokens,
                "completion_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
            },
        }

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Pydantic owns validation. Do not echo uploaded evidence/base64 in errors.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(error["loc"]), "type": error["type"], "msg": error["msg"]} for error in exc.errors()
                ]
            },
        )

    app.add_api_route(
        "/v1/systemone",
        system_one,
        methods=["POST"],
        response_model=ChoiceResponse,
        summary="Evaluate named Choice questions",
    )
    app.add_api_route("/v1/chat/completions", chat, methods=["POST"], summary="Chat compatibility wrapper for Choice")
    app.state.openai_serving_models = handler
    app.state.engine_client = handler
    app.state.log_stats = False
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    upstream_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with upstream_lifespan(application):
                yield
        finally:
            await handler.close()

    app.router.lifespan_context = lifespan
    # Upstream SageMaker bootstrap materializes the middleware stack. Rebuild it
    # with this opt-in app's final handlers before the first request.
    app.middleware_stack = None
    return BodyLimitMiddleware(app)


async def run_choice_server(args, sock, **uvicorn_kwargs):
    import vllm.envs as envs
    from vllm.entrypoints.launchers.launcher import serve_http

    keys = args.explicit_keys
    engine = AsyncChoiceEngine(
        args.model,
        args.choice_bundle,
        args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs if "max_num_seqs" in keys else 8,
        max_num_batched_tokens=args.max_num_batched_tokens if "max_num_batched_tokens" in keys else None,
    )
    app = build_choice_app(args, engine)
    handler = app.state.engine_client
    try:
        task = await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            enable_ssl_refresh=args.enable_ssl_refresh,
            log_level=args.uvicorn_log_level,
            access_log=not args.disable_uvicorn_access_log,
            timeout_keep_alive=envs.VLLM_HTTP_TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            ssl_ciphers=args.ssl_ciphers,
            **uvicorn_kwargs,
        )
        await task
    finally:
        await handler.close()
