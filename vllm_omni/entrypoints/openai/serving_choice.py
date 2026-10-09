# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Choice API using the existing vLLM HTTP server and middleware."""

import asyncio
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import create_model
from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.serve.engine.protocol import ModelCard, ModelList

from vllm_omni.entrypoints.audio_choice.contract import ChoiceResponse
from vllm_omni.entrypoints.audio_choice.decisions import DecisionsRequest, DecisionsResponse
from vllm_omni.entrypoints.audio_choice.decisions_adapter import from_choice, to_choice
from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine
from vllm_omni.entrypoints.audio_choice.errors import ChoiceInputError, log_input_error
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceRequest
from vllm_omni.entrypoints.audio_decision.cli_args import validate_decision_args
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
            if request.input_image is not None or request.input_video is not None:
                visual = await self.loop.run_in_executor(self.pool, self.engine.decode_visual, request)
                return await self.engine.evaluate(request, wave, visual=visual)
            return await self.engine.evaluate(request, wave)

        task = asyncio.create_task(run())
        self.active.add(task)

        def done(future):
            self.pending -= 1
            self.active.discard(future)
            if not future.cancelled():
                # Observe detached failures; engine.errored owns engine health.
                future.exception()

        task.add_done_callback(done)
        try:
            # Keep admission until decoding/inference actually completes, even if
            # the HTTP waiter disconnects. Shutdown drains these tracked tasks.
            return await asyncio.shield(task)
        except ChoiceInputError as exc:
            log_input_error(exc)
            # The original chain is retained and logged above. Prevent upstream
            # HTTP stack logging from serializing media-bearing cause messages.
            status = 503 if self.errored else 422
            message = "Choice engine unavailable" if self.errored else str(exc)
            raise HTTPException(status, message) from None
        except ValueError as exc:
            if self.errored:
                log_input_error(exc)
                raise HTTPException(503, "Choice engine unavailable") from None
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            status = 503 if self.errored else 500
            raise HTTPException(
                status, "Choice engine unavailable" if status == 503 else "Choice request failed"
            ) from exc


def build_choice_app(args, engine):
    names = getattr(args, "served_model_name", None) or ["spev"]
    if len(names) != 1:
        raise ValueError("Choice requires exactly one served model name")
    app = build_app(args, ("generate",))
    keep = {"/health", "/version", "/v1/models", "/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") in keep]
    handler = ChoiceServing(engine, names[0], getattr(args, "decision_max_pending", 16))
    decisions_type = create_model("BoundDecisionsRequest", __base__=DecisionsRequest, model=(Literal[names[0]], ...))

    async def load():
        # Count actual admitted work, including disconnected HTTP waiters.
        return {"server_load": handler.pending}

    app.add_api_route("/load", load, methods=["GET"])

    async def decisions(request: decisions_type) -> DecisionsResponse:
        try:
            native = to_choice(request)
        except ValueError:
            raise HTTPException(422, "Unsupported Decisions input or model limit exceeded") from None
        return from_choice(request, await handler.evaluate(native))

    async def retired_system_one():
        return JSONResponse(
            status_code=410,
            content={
                "error": {
                    "message": "/v1/systemone has been removed. Use POST /v1/decisions "
                    "with input and a questions array; "
                    "use predicate, choices and levels instead of noul and criteria.",
                    "type": "invalid_request_error",
                    "code": "endpoint_removed",
                    "param": None,
                }
            },
        )

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
        "/v1/decisions",
        decisions,
        methods=["POST"],
        response_model=DecisionsResponse,
        summary="Evaluate predicate, choice and score questions",
    )
    app.add_api_route("/v1/systemone", retired_system_one, methods=["POST"], deprecated=True, status_code=410)
    app.state.openai_serving_models = handler
    app.state.engine_client = handler
    app.state.log_stats = False
    app.state.enable_server_load_tracking = False
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
    validate_decision_args(args)
    import vllm.envs as envs
    from vllm.entrypoints.launchers.launcher import serve_http

    keys = args.explicit_keys
    engine = AsyncChoiceEngine(
        args.model,
        args.choice_bundle,
        args.gpu_memory_utilization,
        enable_vision=args.choice_enable_vision,
        max_video_seconds=args.choice_max_video_seconds,
        max_video_frames=args.choice_max_video_frames,
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
