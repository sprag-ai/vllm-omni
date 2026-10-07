# SPDX-License-Identifier: Apache-2.0
"""Native-audio handlers attached to the existing OpenAI API routers.

This opt-in server mode uses the frozen serial decision engine. It deliberately
rejects unsupported generation/pooling options instead of silently ignoring them.
"""

import asyncio
import base64
import binascii
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.completion.protocol import CompletionResponse
from vllm.entrypoints.pooling.embed.protocol import EmbeddingResponse
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, ModelCard, ModelList

from vllm_omni.entrypoints.audio_decision.engine import DecisionEngine
from vllm_omni.entrypoints.audio_decision.server import MAX_BYTES, decode_audio
from vllm_omni.entrypoints.openai.decision_protocol import (
    DecisionChatRequest,
    DecisionInputAudio,
    decision_chat_request_type,
)

MAX_JSON_BYTES = 4 * ((MAX_BYTES + 2) // 3) + 65536


class DecisionCompletionResponse(CompletionResponse):
    decision: dict


class DecisionChatResponse(ChatCompletionResponse):
    decision: dict


class DecisionEmbeddingResponse(EmbeddingResponse):
    decision: dict


def error(message, code=400):
    return ErrorResponse(error={"message": message, "type": "invalid_request_error", "code": code})


def json_error(message, code=400):
    return JSONResponse(error(message, code).model_dump(), status_code=code)


class BodyLimitMiddleware:
    """Bound JSON uploads before the upstream router parses base64 audio."""

    def __init__(self, app):
        self.app = app

    def __getattr__(self, name):
        return getattr(self.app, name)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_JSON_BYTES:
                return await json_error("Audio JSON upload is too large", 413)(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def parse_audio(body):
    audio = DecisionInputAudio.model_validate(body.get("input_audio"))
    try:
        payload = base64.b64decode(audio.data, validate=True)
    except (ValueError, binascii.Error) as e:
        raise ValueError("Invalid base64 audio") from e
    if len(payload) > MAX_BYTES:
        raise ValueError("Audio exceeds 12 MiB")
    # No URL/file-path fetching. decode_audio bounds duration/channels/sample rate.
    try:
        return decode_audio(payload)
    except (ValueError, RuntimeError) as e:
        raise ValueError("Invalid audio: " + str(e)) from e


class DecisionServing:
    def __init__(self, engine, model_name, max_pending=16):
        if max_pending < 1:
            raise ValueError("decision-max-pending must be positive")
        self.engine = engine
        self.model_name = model_name
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision-openai")
        self.pending = 0
        self.max_pending = max_pending
        self.closed = False
        self.errored = False
        self._shutdown_lock = threading.Lock()
        self._stopped = False
        # serve_http's shutdown handler reads this field on the engine client.
        self.vllm_config = SimpleNamespace(shutdown_timeout=10)

    @property
    def is_running(self):
        return not self.closed and not self.errored

    async def check_health(self):
        if not self.is_running:
            raise RuntimeError("Decision engine is shut down")

    async def show_available_models(self):
        return ModelList(data=[ModelCard(id=self.model_name, root=self.model_name, max_model_len=2048)])

    def common(self, body, kind):
        if body.get("model") != self.model_name:
            raise ValueError(f"Unknown model; expected {self.model_name}")
        text = body.get("prompt" if kind == "completion" else "input")
        if not isinstance(text, str) or text not in ("audio_turn_decision", self.engine.config["prompt"]):
            raise ValueError("This frozen audio task accepts audio_turn_decision or its exact bundled prompt only")
        common = {"model", "input_audio", "user"}
        supported = (
            {
                "prompt",
                "max_tokens",
                "temperature",
                "logprobs",
                "allowed_token_ids",
                "n",
                "stream",
                "decision_mode",
                "decision_threshold",
            }
            if kind == "completion"
            else {"input", "encoding_format", "dimensions"}
        )
        unknown = set(body) - common - supported
        if unknown:
            raise ValueError("Unsupported fields for audio decisions: " + ", ".join(sorted(unknown)))
        if kind == "completion":
            if body.get("max_tokens", 1) != 1 or body.get("n", 1) != 1 or body.get("stream", False) is not False:
                raise ValueError("Only max_tokens=1, n=1, stream=false are supported")
            if body.get("temperature", 0) != 0:
                raise ValueError("Only deterministic temperature=0 is supported")
            ids = body.get("allowed_token_ids", self.engine.token_ids)
            if ids != self.engine.token_ids:
                raise ValueError(f"allowed_token_ids must be {self.engine.token_ids} (A/B/C)")
            lp = body.get("logprobs")
            if lp is not None and (isinstance(lp, bool) or not isinstance(lp, int) or not 0 <= lp <= 3):
                raise ValueError("logprobs must be between 0 and 3")
        else:
            if body.get("encoding_format", "float") not in ("float", "base64"):
                raise ValueError("encoding_format must be float or base64")
            if body.get("dimensions") is not None:
                raise ValueError("Raw residual embeddings cannot be truncated with dimensions")

    async def infer(self, body, mode, threshold):
        if self.closed:
            return error("Decision engine unavailable", 503)
        if self.pending >= self.max_pending:
            return error("Decision queue is full", 429)
        self.pending += 1
        try:

            def execute():
                wave = parse_audio(body)
                return self.engine.decide(wave, threshold, mode=mode)

            future = asyncio.get_running_loop().run_in_executor(self.pool, execute)
        except BaseException:
            self.pending -= 1
            raise
        # HTTP cancellation must not admit another request while its GPU job runs.
        future.add_done_callback(lambda _: setattr(self, "pending", self.pending - 1))
        try:
            return await asyncio.shield(future)
        except ValueError as e:
            return error(str(e))
        except Exception:
            self.errored = True
            raise

    async def create_chat_completion(self, request: DecisionChatRequest, raw_request):
        body = {"input_audio": request.input_audio.model_dump()}
        # The chat API is the fixed calibrated policy, not the raw readout experiment API.
        result = await self.infer(body, "auto", 0.95)
        if isinstance(result, ErrorResponse):
            return result
        labels = ["A", "B", "C"]
        index = self.engine.config["actions"].index(result["action"])
        logprobs = None
        if request.logprobs:
            lp = result["label_logprobs"]
            ranked = sorted(range(3), key=lambda i: lp[i], reverse=True)

            def token(i):
                return {"token": labels[i], "logprob": lp[i], "bytes": list(labels[i].encode())}

            logprobs = {
                "content": [{**token(index), "top_logprobs": [token(i) for i in ranked[: request.top_logprobs]]}]
            }
        tokens = result["input_tokens"]
        return DecisionChatResponse(
            model=self.model_name,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": labels[index]},
                    "finish_reason": "length",
                    "logprobs": logprobs,
                }
            ],
            usage={"prompt_tokens": tokens, "completion_tokens": 1, "total_tokens": tokens + 1},
            decision={**result, "mode": "auto"},
        )

    async def create_completion(self, request, raw_request):
        body = await raw_request.json()
        try:
            self.common(body, "completion")
            mode = body.get("decision_mode", "raw")
            if mode not in ("raw", "auto", "head", "full"):
                raise ValueError("decision_mode must be raw, auto, head or full")
            threshold = body.get("decision_threshold", 0.95)
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError("decision_threshold must be finite and between 0 and 1")
        except ValueError as e:
            return error(str(e))
        result = await self.infer(body, mode, threshold)
        if isinstance(result, ErrorResponse):
            return result
        labels = ["A", "B", "C"]
        index = self.engine.config["actions"].index(result["action"])
        logprobs = None
        if body.get("logprobs") is not None:
            lp = result["label_logprobs"]
            ranked = sorted(range(3), key=lambda i: lp[i], reverse=True)
            top = {labels[i]: lp[i] for i in ranked[: body["logprobs"]]}
            # OpenAI includes the sampled token even when top_logprobs=0.
            top[labels[index]] = lp[index]
            logprobs = {
                "tokens": [labels[index]],
                "token_logprobs": [lp[index]],
                "top_logprobs": [top],
                "text_offset": [len(body["prompt"])],
            }
        tokens = result["input_tokens"]
        return DecisionCompletionResponse(
            model=self.model_name,
            choices=[{"index": 0, "text": labels[index], "finish_reason": "length", "logprobs": logprobs}],
            usage={"prompt_tokens": tokens, "completion_tokens": 1, "total_tokens": tokens + 1},
            decision={**result, "mode": mode},
        )

    async def __call__(self, request, raw_request):
        body = await raw_request.json()
        try:
            self.common(body, "embedding")
        except ValueError as e:
            return json_error(str(e))
        result = await self.infer(body, "embedding", 0.95)
        if isinstance(result, ErrorResponse):
            return JSONResponse(result.model_dump(), status_code=result.error.code)
        vector = result.pop("embedding")
        encoded = vector
        if body.get("encoding_format", "float") == "base64":
            encoded = base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")
        # Do not present this as normalized embeddings or a decision distribution.
        metadata = {
            k: result[k]
            for k in ("decoder_depth", "audio_seconds", "input_tokens", "audio_encoder_calls", "elapsed_ms")
        }
        metadata.update(normalized=False, representation="last_token_post_block_residual", internal_sample_tokens=1)
        response = DecisionEmbeddingResponse(
            model=self.model_name,
            data=[{"index": 0, "embedding": encoded}],
            usage={"prompt_tokens": result["input_tokens"], "total_tokens": result["input_tokens"]},
            decision=metadata,
        )
        return JSONResponse(response.model_dump())

    def shutdown(self, timeout=10):
        # Called in the upstream launcher's executor. Stop admission immediately,
        # then drain this service's worker before disposing of engine resources.
        # The container stop grace period is the outer bound for a stalled GPU.
        with self._shutdown_lock:
            if self._stopped:
                return
            self.closed = True
            self.pool.shutdown(wait=True, cancel_futures=False)
            llm = getattr(self.engine, "llm", None)
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
            self._stopped = True

    async def close(self):
        await asyncio.to_thread(self.shutdown)


def build_decision_app(args, engine):
    """Keep upstream handlers and middleware with a strict decision chat schema."""
    app = build_app(args, ("generate", "embed"))
    # Upstream chooses a single runner router family; explicitly attach its
    # embedding router for this dual-readout engine, without replacing routes.
    if not any(getattr(r, "path", None) == "/v1/embeddings" for r in app.routes):
        from vllm.entrypoints.pooling.embed.api_router import router as embed_router

        app.include_router(embed_router)
    names = getattr(args, "served_model_name", None) or ["native-audio-decision"]
    if len(names) != 1:
        raise ValueError("Decision serving requires exactly one served model name")
    # Bind the request schema before FastAPI parses the body. Delegate transport,
    # cancellation and load accounting to the upstream chat route handler.
    from vllm.entrypoints.openai.chat_completion.api_router import create_chat_completion

    request_type = decision_chat_request_type(names[0], engine.config["prompt"])

    async def decision_chat(request: request_type, raw_request: Request):
        return await create_chat_completion(request, raw_request)

    chat_route = next(r for r in app.routes if getattr(r, "path", None) == "/v1/chat/completions")
    app.router.routes.remove(chat_route)
    app.add_api_route(
        "/v1/chat/completions",
        decision_chat,
        methods=["POST"],
        dependencies=chat_route.dependencies,
        responses=chat_route.responses,
    )
    handler = DecisionServing(engine, names[0], getattr(args, "decision_max_pending", 16))
    app.state.openai_serving_completion = handler
    app.state.openai_serving_chat = handler
    app.state.serving_embedding = handler  # vLLM 0.30 uses this name in embed.api_router.
    app.state.openai_serving_models = handler
    app.state.engine_client = handler
    app.state.log_stats = False
    app.state.enable_server_load_tracking = True
    app.state.server_load_metrics = 0
    # Unsupported routes should be absent, not advertise handlers that aren't loaded.
    allowed = {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/models",
        "/health",
        "/version",
        "/load",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", "") in allowed]
    upstream_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        try:
            async with upstream_lifespan(application):
                yield
        finally:
            await handler.close()

    app.router.lifespan_context = lifespan
    return BodyLimitMiddleware(app)


async def run_decision_server(args, sock, **uvicorn_kwargs):
    import vllm.envs as envs
    from vllm.entrypoints.launchers.launcher import serve_http

    # The current model's side-channel is deliberately single-request; fail early.
    if getattr(args, "tensor_parallel_size", 1) != 1 or getattr(args, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Decision mode supports TP=1 and PP=1 only")
    engine = DecisionEngine(args.model, args.decision_bundle, args.gpu_memory_utilization)
    app = build_decision_app(args, engine)
    shutdown_task = await serve_http(
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
    await shutdown_task
