# SPDX-License-Identifier: Apache-2.0
"""Choice, Noul and Score likelihoods through AsyncLLM."""

import asyncio
import hashlib
import json
import math
import os
import threading
import uuid
from pathlib import Path

from vllm_omni.entrypoints.audio_choice.contract import (
    ChoiceResponse,
    Usage,
    primitive_answer,
    render_question,
    scoring_question,
    target,
)
from vllm_omni.entrypoints.audio_choice.errors import ChoiceInputError

MAX_MODEL_LEN = 8192
MAX_PROMPT_CHARS = 131072


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

    def __init__(
        self,
        model,
        bundle,
        gpu_memory_utilization=0.9,
        max_num_seqs=8,
        max_num_batched_tokens=None,
        enable_vision=False,
        max_video_seconds=60,
        max_video_frames=1800,
    ):
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
        if not math.isfinite(max_video_seconds) or max_video_seconds <= 0 or max_video_frames < 1:
            raise ValueError("Choice video limits must be positive and finite")
        self.enable_vision = enable_vision
        self.max_video_seconds = max_video_seconds
        self.max_video_frames = max_video_frames
        self.visual_lock = threading.Lock()
        self.bundle = Path(bundle).resolve()
        self.config, self.temperature = verify_bundle(self.bundle)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.slots = asyncio.Semaphore(max_num_seqs)
        self.prepare_lock = threading.Lock()
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
                max_model_len=MAX_MODEL_LEN,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=False,
                async_scheduling=False,
                enable_prefix_caching=False,
                gpu_memory_utilization=gpu_memory_utilization,
                limit_mm_per_prompt={"audio": 1, "image": int(enable_vision), "video": int(enable_vision)},
                mm_processor_cache_gb=0,
                seed=17,
                disable_log_stats=True,
            )
        )

    def decode_visual(self, request):
        from vllm_omni.entrypoints.audio_choice.media import decode_visual

        if not self.enable_vision:
            raise ValueError("Visual inputs require --choice-enable-vision")
        return decode_visual(request, self.processor, self.visual_lock, self.max_video_seconds, self.max_video_frames)

    def prompt(self, question, state, has_audio, visual=None):
        content = [{"type": "audio", "audio": "provided-array"}] if has_audio else []
        if visual is not None:
            content.extend(visual.content())
        content.append({"type": "text", "text": render_question(question, state)})
        return self.processor.apply_chat_template(
            [
                {"role": "system", "content": "Follow the task instructions using only the supplied evidence."},
                {"role": "user", "content": content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def audio_tokens(self, wave):
        from transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe import _get_feat_extract_output_lengths

        features = self.processor.feature_extractor(
            wave, sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True, return_tensors="np"
        )
        return int(_get_feat_extract_output_lengths(features["attention_mask"].sum(-1))[0])

    def prepare(self, question, state, wave, visual=None):
        # Called off the event loop. Protect the shared HF processor/tokenizer
        # from concurrent mutation of its internal padding/truncation settings.
        with self.prepare_lock:
            rendered = self.prompt(question, state, wave is not None, visual)
            if len(rendered) > MAX_PROMPT_CHARS:
                raise ValueError("Choice prompt exceeds the text size limit")
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(prefix) >= MAX_MODEL_LEN:
                raise ValueError("Choice prompt exceeds the 8192-token context")
            expanded_length = len(prefix)
            audio_id = self.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
            if prefix.count(audio_id) != int(wave is not None):
                raise ValueError("Choice audio placeholder count mismatch")
            if wave is not None:
                # Count the same feature-mask expansion used by the native
                # processor, before submitting any candidate to the scheduler.
                expanded_length += self.audio_tokens(wave) - 1
            if visual is not None:
                for modality in ("image", "video"):
                    token_id = self.tokenizer.convert_tokens_to_ids(f"<|{modality}_pad|>")
                    if prefix.count(token_id) != int(getattr(visual, modality) is not None):
                        raise ValueError(f"Choice {modality} placeholder count mismatch")
                expanded_length += visual.expansion
            end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            candidates = {}
            for key in question.criteria:
                ids = self.tokenizer.encode(target(key), add_special_tokens=False) + [end_id]
                # Reserve one token for the discarded internal generation step.
                if expanded_length + len(ids) + 1 > MAX_MODEL_LEN:
                    raise ValueError("Choice prompt and target exceed the 8192-token context")
                candidates[key] = ids
            return prefix, candidates

    async def score(self, prefix, ids, wave, visual=None):
        from vllm import SamplingParams
        from vllm.exceptions import VLLMClientError
        from vllm.v1.engine.exceptions import EngineGenerateError

        # Caller content was escaped before tokenization. Submit IDs directly so
        # vLLM does not tokenize the shared text again for every candidate.
        prompt = {"prompt_token_ids": prefix + ids}
        request_id = "choice-" + uuid.uuid4().hex
        mm_data = visual.multimodal_data() if visual is not None else {}
        if wave is not None:
            mm_data["audio"] = (wave, 16000)
        if mm_data:
            prompt.update(multi_modal_data=mm_data, multi_modal_uuids={key: [request_id + key] for key in mm_data})
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
            except (EngineGenerateError, VLLMClientError) as exc:
                if self.llm.errored:
                    raise
                if isinstance(exc, VLLMClientError) or isinstance(exc.__cause__, ValueError):
                    raise ChoiceInputError("Invalid or unsupported model input") from exc
                raise
            finally:
                if not finished:
                    await asyncio.shield(self.llm.abort(request_id))

    async def evaluate(self, request, wave=None, visual=None):
        answers = {}
        input_tokens = 0
        for question_id, question in request.questions.items():
            candidate_question = scoring_question(question)
            prefix, candidates = await asyncio.to_thread(self.prepare, candidate_question, request.state, wave, visual)
            tasks = [asyncio.create_task(self.score(prefix, ids, wave, visual)) for ids in candidates.values()]
            try:
                scores = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            # The fitted temperature covers Choice only; new primitives start at identity.
            temperature = self.temperature if question.type == "choice" and visual is None else 1.0
            answers[question_id] = primitive_answer(
                question, dict(zip(candidate_question.criteria, [s[0] for s in scores])), temperature
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
