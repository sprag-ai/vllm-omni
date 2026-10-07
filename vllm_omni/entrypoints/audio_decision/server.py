# SPDX-License-Identifier: Apache-2.0
"""Bounded single-worker HTTP frontend for the native decision engine."""

import argparse
import asyncio
import hmac
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from scipy.signal import resample_poly
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media.audio import load_audio

from .engine import DecisionEngine

MAX_BYTES = 12 * 1024 * 1024
MAX_SECONDS = 30


def decode_audio(data):
    if len(data) > MAX_BYTES:
        raise ValueError("Audio exceeds 12 MiB")
    # Use the same auto decoder/fallback path as vLLM's AudioMediaIO. Keep
    # source-rate decoding so the frozen model's resampling stays unchanged.
    try:
        wave, rate = load_audio(
            io.BytesIO(data),
            sr=None,
            mono=True,
            max_duration_s=MAX_SECONDS,
            max_decode_bytes=MAX_SECONDS * 192000 * 8 * np.dtype(np.float32).itemsize,
        )
    except VLLMValidationError as e:
        raise ValueError(str(e)) from e
    if wave.size == 0 or not 8000 <= rate <= 192000 or wave.size / rate > MAX_SECONDS:
        raise ValueError("Audio must contain 0–30 seconds at 8–192 kHz")
    if not np.isfinite(wave).all():
        raise ValueError("Audio contains nonfinite samples")
    if rate != 16000:
        g = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // g, rate // g).astype(np.float32)
    return wave


def create_app(model, bundle, api_key=None, max_pending=16):
    @asynccontextmanager
    async def lifespan(app):
        app.state.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision")
        # Initialize on the main thread (vLLM installs process-level handlers).
        app.state.engine = DecisionEngine(model, bundle)
        app.state.pending = 0
        app.state.ready = True
        yield
        app.state.ready = False
        app.state.pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Native audio decision API", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": getattr(app.state, "ready", False), "backend": "vllm-omni-0.30.0"}

    @app.post("/v1/decide")
    async def decide(request: Request, threshold: float):
        if api_key and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + api_key):
            raise HTTPException(401, "Invalid API key")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise HTTPException(422, "threshold must be finite and between 0 and 1")
        if app.state.pending >= max_pending:
            raise HTTPException(429, "Decision queue is full", headers={"Retry-After": "1"})
        app.state.pending += 1
        submitted = False
        start = time.perf_counter()
        try:
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise HTTPException(413, "Audio upload exceeds 12 MiB")

            def process():
                try:
                    wave = decode_audio(data)
                except (ValueError, RuntimeError) as e:
                    raise ValueError("Invalid audio: " + str(e)) from e
                answer = app.state.engine.decide(wave, threshold)
                return {**answer, "server_elapsed_ms": (time.perf_counter() - start) * 1000}

            future = asyncio.get_running_loop().run_in_executor(app.state.pool, process)
            # Keep the admission slot until work actually finishes, even on disconnect.
            future.add_done_callback(lambda _: setattr(app.state, "pending", app.state.pending - 1))
            submitted = True
            return await asyncio.shield(future)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        finally:
            if not submitted:
                app.state.pending -= 1

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/models/qwen3-omni")
    parser.add_argument("--bundle", default="/models/decision")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.model, args.bundle, os.environ.get("DECISION_API_KEY")),
        host=args.host,
        port=args.port,
        timeout_keep_alive=10,
    )


if __name__ == "__main__":
    main()
