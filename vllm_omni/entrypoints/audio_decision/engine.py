# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import math
import threading
import time
import uuid
from pathlib import Path

import numpy as np


def verify_bundle(root):
    root = Path(root)
    config = json.loads((root / "decision.json").read_text())
    for name, expected in config["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Bundle paths must be relative and contained")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle hash mismatch: {name}")
    if config["actions"] != ["keep_listening", "respond", "insufficient_evidence"]:
        raise ValueError("Unsupported action schema")
    with np.load(root / "head.npz", allow_pickle=False) as h:
        for k in ("mean", "scale", "weight", "bias"):
            if not np.isfinite(h[k]).all():
                raise ValueError("Nonfinite head")
        if np.any(h["scale"] <= 0) or h["weight"].shape != (h["mean"].size, 3):
            raise ValueError("Invalid head dimensions or scales")
    return config


def validate_wave(wave, threshold, sample_rate=16000, max_seconds=30):
    if sample_rate != 16000:
        raise ValueError("Waveform input must be 16000 Hz")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a number")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between 0 and 1")
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not 0 < len(wave) <= sample_rate * max_seconds:
        raise ValueError(f"Expected nonempty mono audio, at most {max_seconds} seconds")
    if not np.isfinite(wave).all():
        raise ValueError("Audio contains nonfinite samples")
    return wave


class DecisionEngine:
    """Persistent vLLM engine; admission is deliberately serial in this first port."""

    def __init__(self, model, bundle, gpu_memory_utilization=0.90):
        from transformers import Qwen3OmniMoeProcessor
        from vllm import LLM, SamplingParams
        from vllm.model_executor.models import ModelRegistry

        from vllm_omni.entrypoints.audio_decision.model import arm_worker, take_worker

        self.bundle = Path(bundle).resolve()
        self.config = verify_bundle(self.bundle)
        self.lock = threading.Lock()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B", "C")]
        if any(len(v) != 1 for v in ids):
            raise ValueError("Decision IDs must be single tokens")
        self.token_ids = [v[0] for v in ids]
        ModelRegistry.register_model(
            "Qwen3OmniDecisionThinker",
            "vllm_omni.entrypoints.audio_decision.model:Qwen3OmniDecisionThinker",
        )
        self.llm = LLM(
            model=model,
            tokenizer=model,
            dtype="bfloat16",
            hf_overrides={
                "architectures": ["Qwen3OmniDecisionThinker"],
                "decision_bundle": str(self.bundle),
                "decision_token_ids": self.token_ids,
            },
            enforce_eager=True,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_num_seqs=1,
            max_model_len=2048,
            max_num_batched_tokens=2048,
            enable_chunked_prefill=False,
            async_scheduling=False,
            enable_prefix_caching=False,
            gpu_memory_utilization=gpu_memory_utilization,
            limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
            mm_processor_cache_gb=0,
            seed=17,
        )
        self.sampling = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
        self.arm_worker, self.take_worker = arm_worker, take_worker

    def decide(self, wave, threshold, sample_rate=16000, mode="auto"):
        wave = validate_wave(wave, threshold, sample_rate)
        if mode not in ("auto", "head", "full"):
            raise ValueError("Unknown execution mode")
        with self.lock:
            start = time.perf_counter()
            content = [{"type": "audio", "audio": "provided-array"}, {"type": "text", "text": self.config["prompt"]}]
            rendered = self.processor.apply_chat_template(
                [
                    {"role": "system", "content": "Follow the task instructions using only the supplied evidence."},
                    {"role": "user", "content": content},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            prefix = self.tokenizer.encode(rendered, add_special_tokens=False)
            for letter, token in zip(("A", "B", "C"), self.token_ids):
                if self.tokenizer.encode(rendered + letter, add_special_tokens=False) != prefix + [token]:
                    raise RuntimeError("Action token boundary changed")
            request_id = uuid.uuid4().hex
            self.llm.collective_rpc(self.arm_worker, args=(threshold, request_id, mode))
            result = None
            try:
                outputs = self.llm.generate(
                    {
                        "prompt": rendered,
                        "multi_modal_data": {"audio": (wave, 16000)},
                        "multi_modal_uuids": {"audio": [request_id]},
                    },
                    self.sampling,
                    use_tqdm=False,
                )
                result = self.llm.collective_rpc(self.take_worker)[0]
                if not result or result["request_id"] != request_id:
                    raise RuntimeError("Worker decision did not match the request")
                if result["audio_encoder_calls"] != 1:
                    raise RuntimeError("Expected exactly one fresh audio encoder execution")
                token = outputs[0].outputs[0].token_ids
                expected = self.token_ids[self.config["actions"].index(result["action"])]
                if list(token) != [expected]:
                    raise RuntimeError("Sampler and decision head disagree")
            finally:
                if result is None:
                    self.llm.collective_rpc(self.take_worker)
            result.pop("request_id")
            return {
                **result,
                "model": self.config["model"],
                "backend": "vllm-omni-0.30.0",
                "audio_seconds": len(wave) / 16000,
                "elapsed_ms": (time.perf_counter() - start) * 1000,
            }
