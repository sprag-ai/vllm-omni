# SPDX-License-Identifier: Apache-2.0
"""Named Choice likelihoods through AsyncLLM, with no fixed action head."""

import asyncio
import hashlib
import json
import math
import os
import uuid
from pathlib import Path

from vllm_omni.entrypoints.audio_choice.contract import ChoiceResponse, Usage, answer, render_question, target


def verify_bundle(root):
    root = Path(root).resolve()
    config = json.loads((root / "choice.json").read_text())
    required = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors", "calibration.json"}
    if set(config["files"]) != required:
        raise ValueError("Choice bundle must hash exactly its adapter and calibration files")
    for name, digest in config["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Choice bundle hash mismatch: " + name)
    calibration = json.loads((root / "calibration.json").read_text())
    if calibration["adapter_sha256"] != config["files"]["adapter/adapter_model.safetensors"]:
        raise ValueError("Calibration belongs to a different adapter")
    temperature = calibration["temperature"]
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid Choice calibration temperature")
    if calibration["scorer"] != "vllm-full-sequence-choice-v1":
        raise ValueError("Choice calibration must match the serving scorer")
    return config, temperature


class AsyncChoiceEngine:
    is_async = True

    def __init__(self, model, bundle, gpu_memory_utilization=0.9, max_num_seqs=8, max_num_batched_tokens=None):
        from transformers import Qwen3OmniMoeProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        if not 1 <= max_num_seqs <= 32:
            raise ValueError("Choice max-num-seqs must be between 1 and 32")
        max_num_batched_tokens = 16384 if max_num_batched_tokens is None else max_num_batched_tokens
        if not 8192 <= max_num_batched_tokens <= 262144:
            raise ValueError("Choice token budget must be between 8192 and 262144")
        for key, value in {"VLLM_BATCH_INVARIANT": "1", "VLLM_USE_V2_MODEL_RUNNER": "0"}.items():
            if os.environ.get(key, value) != value:
                raise ValueError(f"Choice serving requires {key}={value}")
            os.environ.setdefault(key, value)
        self.bundle = Path(bundle).resolve()
        self.config, self.temperature = verify_bundle(self.bundle)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.slots = asyncio.Semaphore(max_num_seqs)
        self.llm = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model,
                tokenizer=model,
                dtype="bfloat16",
                hf_overrides={"architectures": ["Qwen3OmniChoiceThinker"], "choice_bundle": str(self.bundle)},
                enforce_eager=True,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                max_num_seqs=max_num_seqs,
                max_model_len=8192,
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
        )

    def prompt(self, question, state, has_audio):
        content = [{"type": "audio", "audio": "provided-array"}] if has_audio else []
        content.append({"type": "text", "text": render_question(question, state)})
        return self.processor.apply_chat_template(
            [
                {"role": "system", "content": "Follow the task instructions using only the supplied evidence."},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    async def score(self, rendered, key, wave):
        from vllm import SamplingParams

        text = target(key)
        prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.tokenizer.encode(rendered + text, add_special_tokens=False) != prefix + ids:
            raise ValueError("Choice target token boundary changed")
        ids.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))
        prompt = {"prompt": rendered + text + "<|im_end|>"}
        request_id = "choice-" + uuid.uuid4().hex
        if wave is not None:
            prompt.update(multi_modal_data={"audio": (wave, 16000)}, multi_modal_uuids={"audio": [request_id]})
        async with self.slots:
            finished = False
            try:
                output = None
                async for update in self.llm.generate(
                    prompt, SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, prompt_logprobs=0), request_id
                ):
                    output = update
                if output is None or output.prompt_logprobs is None:
                    raise RuntimeError("Missing Choice prompt likelihoods")
                if list(output.prompt_token_ids[-len(ids) :]) != ids:
                    raise RuntimeError("Scored token suffix does not match the named Choice target")
                terms = [entry[token].logprob for entry, token in zip(output.prompt_logprobs[-len(ids) :], ids)]
                if not all(math.isfinite(x) for x in terms):
                    raise RuntimeError("Nonfinite Choice likelihood")
                finished = True
                return sum(terms), len(output.prompt_token_ids) - len(ids)
            finally:
                if not finished:
                    await asyncio.shield(self.llm.abort(request_id))

    async def evaluate(self, request, wave=None):
        answers = {}
        input_tokens = 0
        for question_id, question in request.questions.items():
            rendered = self.prompt(question, request.state, wave is not None)
            tasks = [asyncio.create_task(self.score(rendered, key, wave)) for key in question.criteria]
            try:
                scores = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            answers[question_id] = answer(
                question, dict(zip(question.criteria, [s[0] for s in scores])), self.temperature
            )
            # Logical input usage: one shared question prompt; targets are scored, not generated.
            input_tokens += scores[0][1]
        return ChoiceResponse(
            model=self.config["model"], answers=answers, usage=Usage(input_tokens=input_tokens, output_tokens=0)
        )

    @property
    def errored(self):
        return self.llm.errored

    def shutdown(self):
        self.llm.shutdown()
