# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decisions transport for model-configured vLLM I/O processors."""

import asyncio
import uuid
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.chat_completion.serving import _get_mm_token_counts
from vllm.entrypoints.serve.utils.api_utils import load_aware_call
from vllm.exceptions import VLLMClientError
from vllm.logger import init_logger
from vllm.plugins.io_processors import get_io_processor
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.v1.engine.exceptions import EngineGenerateError

from vllm_omni.entrypoints.cli.pooling import DEFAULT_MAX_REQUEST_BODY_BYTES, validate_pooling_args
from vllm_omni.entrypoints.openai.body_limit import BodyLimitMiddleware
from vllm_omni.inputs.decisions import (
    DecisionInputTokensDetails,
    DecisionOutputTokensDetails,
    DecisionsRequest,
    DecisionsResponse,
    DecisionUsage,
)

logger = init_logger(__name__)


class DecisionsServing:
    def __init__(self, engine: EngineClient, processor: IOProcessor, model_names: list[str], max_pending: int):
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.engine = engine
        self.processor = processor
        self.model_names = model_names
        self.max_pending = max_pending
        self.active: set[asyncio.Task[DecisionsResponse]] = set()
        self.closed = False

    async def evaluate(self, request: DecisionsRequest, raw_request: Request | None = None) -> DecisionsResponse:
        if request.model not in self.model_names:
            raise HTTPException(404, "Unknown model")
        if self.closed or self.engine.errored:
            raise HTTPException(503, "Decision engine unavailable")
        if len(self.active) >= self.max_pending:
            raise HTTPException(429, "Decision queue is full", headers={"Retry-After": "1"})
        run = self._evaluate if raw_request is None else load_aware_call(self._evaluate)
        task = asyncio.create_task(run(request, raw_request=raw_request))
        self.active.add(task)

        def done(completed: asyncio.Task[DecisionsResponse]) -> None:
            self.active.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(done)
        return await asyncio.shield(task)

    async def _evaluate(self, request: DecisionsRequest, raw_request: Request | None = None) -> DecisionsResponse:
        request_id = "decision-" + uuid.uuid4().hex
        try:

            def prepare():
                parsed = self.processor.parse_data(request)
                return parsed, self.processor.pre_process(parsed, request_id=request_id)

            parsed, prompt = await asyncio.to_thread(prepare)
            (engine_input,) = await self.engine.renderer.render_cmpl_async([prompt])
            modalities = _get_mm_token_counts(engine_input)
            result = None
            async for output in self.engine.encode(engine_input, self.processor.merge_pooling_params(), request_id):
                result = output
        except (ValueError, VLLMClientError, EngineGenerateError) as exc:
            logger.warning("Decision inference failed: %s", type(exc).__name__)
            if self.engine.errored:
                raise HTTPException(503, "Decision engine unavailable") from exc
            if isinstance(exc, (ValueError, VLLMClientError)) or isinstance(exc.__cause__, ValueError):
                raise HTTPException(422, "Invalid or unsupported model input") from exc
            raise HTTPException(500, "Decision inference failed") from exc
        except Exception as exc:
            logger.error("Decision inference failed: %s", type(exc).__name__)
            raise HTTPException(503 if self.engine.errored else 500, "Decision inference failed") from exc

        try:
            if result is None or not result.finished:
                raise RuntimeError("Missing completed pooling result")
            answers = await asyncio.to_thread(
                partial(self.processor.post_process, [result], request_id=request_id, prompt=parsed)
            )
            count = len(result.prompt_token_ids)
            usage = DecisionUsage(
                input_tokens=count,
                output_tokens=0,
                total_tokens=count,
                input_tokens_details=DecisionInputTokensDetails(
                    cached_tokens=result.num_cached_tokens,
                    cache_write_tokens=0,
                    multimodal_tokens=modalities or None,
                ),
                output_tokens_details=DecisionOutputTokensDetails(reasoning_tokens=0),
            )
            return DecisionsResponse(model=request.model, answers=answers, usage=usage)
        except Exception as exc:
            logger.error("Invalid decision model output: %s", type(exc).__name__)
            raise HTTPException(500, "Invalid decision model output") from exc

    async def close(self) -> None:
        self.closed = True
        if self.active:
            await asyncio.gather(*tuple(self.active), return_exceptions=True)


def mount_decisions(app: FastAPI, engine: EngineClient, args: Namespace) -> DecisionsServing | None:
    processor = get_io_processor(engine.vllm_config, engine.renderer, engine.model_config.io_processor_plugin)
    if processor is None or not getattr(processor, "supports_decisions", False):
        return None
    names = args.served_model_name or [args.model]
    serving = DecisionsServing(engine, processor, names, args.decision_max_pending)

    async def decisions(request: DecisionsRequest, raw_request: Request):
        return await serving.evaluate(request, raw_request=raw_request)

    app.add_api_route("/v1/decisions", decisions, methods=["POST"], response_model=DecisionsResponse)

    upstream_validation_handler = app.exception_handlers.get(RequestValidationError)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        if request.url.path != "/v1/decisions" and upstream_validation_handler is not None:
            return await upstream_validation_handler(request, exc)
        return JSONResponse(status_code=422, content={"error": {"message": "Invalid decision request"}})

    upstream_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with upstream_lifespan(application):
                yield
        finally:
            await serving.close()

    app.router.lifespan_context = lifespan
    app.middleware_stack = None
    return serving


async def run_pooling_server(args, sock, **uvicorn_kwargs):
    from vllm import envs
    from vllm.entrypoints.launchers.api_server.app_state import init_app_state
    from vllm.entrypoints.launchers.api_server.entry import build_async_engine_client
    from vllm.entrypoints.launchers.app import build_app
    from vllm.entrypoints.launchers.launcher import serve_http
    from vllm.entrypoints.launchers.utils.server_utils import get_uvicorn_log_config

    validate_pooling_args(args)
    log_config = get_uvicorn_log_config(args)
    if log_config is not None:
        uvicorn_kwargs["log_config"] = log_config
    try:
        async with build_async_engine_client(args) as engine:
            tasks = await engine.get_supported_tasks()
            app = build_app(args, tasks, engine.model_config)
            await init_app_state(engine, app.state, args, tasks)
            mount_decisions(app, engine, args)
            shutdown = await serve_http(
                BodyLimitMiddleware(app, getattr(args, "max_request_body_bytes", DEFAULT_MAX_REQUEST_BODY_BYTES)),
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
                h11_max_incomplete_event_size=args.h11_max_incomplete_event_size,
                h11_max_header_count=args.h11_max_header_count,
                **uvicorn_kwargs,
            )
        await shutdown
    finally:
        sock.close()
