# SPDX-License-Identifier: Apache-2.0
"""Async vLLM admission with scheduler batching and keyed worker readouts."""

import asyncio
import os
import time
import uuid
from pathlib import Path

from vllm_omni.entrypoints.audio_decision.engine import validate_wave, verify_bundle


class AsyncDecisionEngine:
    is_async = True

    def __init__(self, model, bundle, gpu_memory_utilization=0.90, max_num_seqs=8, max_num_batched_tokens=None):
        if os.environ.get("VLLM_BATCH_INVARIANT", "1") != "1":
            raise ValueError("Decision batches require VLLM_BATCH_INVARIANT=1 for stable per-request readouts")
        os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "0") != "0":
            raise ValueError("Decision batches require VLLM_USE_V2_MODEL_RUNNER=0")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

        from transformers import Qwen3OmniMoeProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        if not 1 <= max_num_seqs <= 32:
            raise ValueError("Decision max-num-seqs must be between 1 and 32")
        if max_num_batched_tokens is None:
            max_num_batched_tokens = 2048 * max_num_seqs
        if not 2048 <= max_num_batched_tokens <= 65536:
            raise ValueError("Decision max-num-batched-tokens must be between 2048 and 65536")
        self.bundle = Path(bundle).resolve()
        self.config = verify_bundle(self.bundle)
        processor = Qwen3OmniMoeProcessor.from_pretrained(model, local_files_only=True)
        tokenizer = processor.tokenizer
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B", "C")]
        if any(len(v) != 1 for v in ids):
            raise ValueError("Decision IDs must be single tokens")
        self.token_ids = [v[0] for v in ids]
        self.rendered = processor.apply_chat_template(
            [
                {"role": "system", "content": "Follow the task instructions using only the supplied evidence."},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": "provided-array"},
                        {"type": "text", "text": self.config["prompt"]},
                    ],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prefix = tokenizer.encode(self.rendered, add_special_tokens=False)
        for letter, token in zip(("A", "B", "C"), self.token_ids):
            if tokenizer.encode(self.rendered + letter, add_special_tokens=False) != prefix + [token]:
                raise RuntimeError("Action token boundary changed")
        args = AsyncEngineArgs(
            model=model,
            tokenizer=model,
            dtype="bfloat16",
            hf_overrides={
                "architectures": ["Qwen3OmniBatchedDecisionThinker"],
                "decision_bundle": str(self.bundle),
                "decision_token_ids": self.token_ids,
                "decision_batched": True,
            },
            worker_cls="vllm_omni.entrypoints.audio_decision.worker.DecisionWorker",
            enforce_eager=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_num_seqs=max_num_seqs,
            max_model_len=2048,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=False,
            async_scheduling=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
            mm_processor_cache_gb=0,
            seed=17,
            disable_log_stats=True,
        )
        # AsyncLLM handles concurrent admission/output. The scheduler's separate
        # CPU/GPU pipelining option stays off for this one-prefill custom worker.
        self.llm = AsyncLLM.from_engine_args(args)

    async def decide(self, wave, threshold, sample_rate=16000, mode="auto"):
        from vllm import SamplingParams

        wave = validate_wave(wave, threshold, sample_rate)
        if mode not in ("auto", "head", "full", "raw", "embedding"):
            raise ValueError("Unknown execution mode")
        start = time.perf_counter()
        request_id = uuid.uuid4().hex
        params = SamplingParams(
            temperature=0,
            max_tokens=1,
            ignore_eos=True,
            extra_args={"audio_decision": {"request_id": request_id, "threshold": threshold, "mode": mode}},
        )
        consumed = False
        try:
            output = None
            async for update in self.llm.generate(
                {
                    "prompt": self.rendered,
                    "multi_modal_data": {"audio": (wave, 16000)},
                    "multi_modal_uuids": {"audio": [request_id]},
                },
                params,
                request_id,
            ):
                output = update
            results = await self.llm.collective_rpc("take_decision_result", args=(request_id,))
            result = results[0]
            consumed = True
            if not result or result["request_id"] != request_id:
                raise RuntimeError("Worker decision did not match the request")
            expected = self.token_ids[self.config["actions"].index(result["action"])]
            if output is None or list(output.outputs[0].token_ids) != [expected]:
                raise RuntimeError("Sampler and decision head disagree")
            result.pop("request_id")
            return {
                **result,
                "model": self.config["model"],
                "backend": "vllm-omni-0.30.0-async",
                "audio_seconds": len(wave) / 16000,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            }
        finally:
            if not consumed:
                # Abort through the engine before discarding the keyed readout.
                # Shield cleanup so client cancellation cannot leak completed rows.
                async def cleanup():
                    await self.llm.abort(request_id)
                    await self.llm.collective_rpc("take_decision_result", args=(request_id,))

                await asyncio.shield(cleanup())

    @property
    def errored(self):
        return self.llm.errored

    def shutdown(self):
        self.llm.shutdown()
