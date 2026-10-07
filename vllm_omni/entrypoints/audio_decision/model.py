# SPDX-License-Identifier: Apache-2.0
"""Single-request eager decision model. No Transformers model execution.

The vLLM fused decoder keeps the residual separate until the next norm. The
trained head sees their BF16 sum, matching the logical post-block residual.
Early acceptance returns before the remaining layers execute. Rejection keeps
those same tensors and continues in the same forward pass.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)


def apply_qv_adapter(output, inputs, qa, qb, va, vb, scale):
    result, bias = output
    x = inputs.float()
    qn, vn = qb.shape[0], vb.shape[0]
    if result.shape[-1] != qn + 2 * vn:
        raise RuntimeError("Unexpected QKV layout; only TP=1 is supported")
    result[..., :qn] += F.linear(F.linear(x, qa), qb) * scale
    result[..., qn + vn :] += F.linear(F.linear(x, va), vb) * scale
    return result, bias


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniDecisionThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        scheduler = vllm_config.scheduler_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Decision model supports TP=1 and PP=1 only")
        if scheduler.enable_chunked_prefill or (
            scheduler.max_num_seqs != 1 and not getattr(vllm_config.model_config.hf_config, "decision_batched", False)
        ):
            raise ValueError("Decision model requires one sequence and unchunked prefill")
        if vllm_config.cache_config.enable_prefix_caching or not vllm_config.model_config.enforce_eager:
            raise ValueError("Decision model requires eager execution without prefix caching")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        root = Path(vllm_config.model_config.hf_config.decision_bundle)
        # This endpoint accepts audio only; no visual deepstack buffers are used.
        self.use_deepstack = False
        self.deepstack_input_embeds = []
        self.decision_root = root
        self.decision_spec = json.loads((root / "decision.json").read_text())
        with np.load(root / "head.npz", allow_pickle=False) as data:
            self.decision_head = {
                k: torch.as_tensor(data[k], device="cpu", dtype=torch.float64)
                for k in ("mean", "scale", "weight", "bias")
            }
        self.decision_ids = list(vllm_config.model_config.hf_config.decision_token_ids)
        self.decision_request = None
        self.decision_result = None
        self.decision_scores = None
        self.decision_encoder_calls = 0

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        # Preserve FP32 PEFT arithmetic: native LoRA kernels require BF16 adapters.
        # These persistent hooks alter only Q/V, after the native fused base GEMM.
        config = json.loads((self.decision_root / "adapter/adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        device = next(self.language_model.parameters()).device
        self.decision_head = {k: v.to(device) for k, v in self.decision_head.items()}
        tensors = load_file(str(self.decision_root / "adapter/adapter_model.safetensors"), device=str(device))
        self._decision_lora = tensors
        self._decision_lora_handles = []
        consumed = set()
        for i, layer in enumerate(self.language_model.model.layers):
            keys = [
                f"base_model.model.thinker.model.layers.{i}.self_attn.{proj}.lora_{which}.weight"
                for proj in ("q_proj", "v_proj")
                for which in ("A", "B")
            ]
            qa, qb, va, vb = [tensors[k] for k in keys]
            consumed.update(keys)
            if any(t.dtype != torch.float32 for t in (qa, qb, va, vb)):
                raise ValueError("Decision adapter must retain FP32 tensors")

            def apply_adapter(module, args, output, qa=qa, qb=qb, va=va, vb=vb):
                return apply_qv_adapter(output, args[0], qa, qb, va, vb, scale)

            self._decision_lora_handles.append(layer.self_attn.qkv_proj.register_forward_hook(apply_adapter))
        if consumed != set(tensors):
            raise ValueError("Adapter has unbound tensors")
        return loaded

    def _process_audio_input(self, audio_input):
        self.decision_encoder_calls += 1
        return super()._process_audio_input(audio_input)

    def arm_decision(self, threshold, request_id, mode="auto"):
        if self.decision_request is not None:
            raise RuntimeError("Previous decision has not been consumed")
        if mode not in ("auto", "head", "full", "raw", "embedding"):
            raise ValueError("Unknown execution mode")
        self.decision_encoder_calls = 0
        self.decision_request = (float(threshold), request_id, mode)
        self.decision_result = None
        self.decision_scores = None

    def take_decision(self):
        result = self.decision_result
        self.decision_request = None
        self.decision_result = None
        self.decision_scores = None
        return result

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        # vLLM profiling/warmup retains the standard full-model path.
        if self.decision_request is None:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if intermediate_tensors is not None:
            raise RuntimeError("Decision serving requires pipeline_parallel_size=1")
        if self.decision_result is not None:
            raise RuntimeError("Decision serving requires one prefill and max_tokens=1")
        threshold, request_id, mode = self.decision_request
        decoder = self.language_model.model
        hidden = inputs_embeds if inputs_embeds is not None else decoder.embed_input_ids(input_ids)
        # Eager, one sequence, unchunked prefill: last row is the decision token.
        if hidden.ndim != 2:
            raise RuntimeError("Unexpected decoder input shape")
        residual = None
        executed = 0
        fallback = True
        head_confidence = None
        self.decision_scores = None
        embedding = None
        for index, layer in enumerate(decoder.layers):
            hidden, residual = layer(positions, hidden, residual)
            executed += 1
            if index + 1 == self.decision_spec["depth"]:
                logical = hidden[-1] if residual is None else hidden[-1] + residual[-1]
                if mode == "embedding":
                    embedding = logical.float().cpu().tolist()
                h = self.decision_head
                scores = ((logical.double() - h["mean"]) / h["scale"]) @ h["weight"] + h["bias"]
                scores = scores / self.decision_spec["head_temperature"]
                head_confidence = float(torch.softmax(scores, dim=-1).max())
                fallback = mode in ("full", "raw") or (mode == "auto" and head_confidence < threshold)
                if not fallback:
                    self.decision_scores = scores
                    break
        if fallback:
            hidden, _ = decoder.norm(hidden, residual)
        self.decision_result = {
            "request_id": request_id,
            "used_full_decoder": fallback,
            "decoder_depth": executed,
            "head_confidence": head_confidence,
            "threshold": threshold,
            "input_tokens": int(hidden.shape[0]),
            "audio_encoder_calls": self.decision_encoder_calls,
        }
        if embedding is not None:
            self.decision_result["embedding"] = embedding
        if inputs_embeds is not None:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden

    def compute_logits(self, hidden_states):
        if self.decision_request is None:
            return super().compute_logits(hidden_states)
        if hidden_states.shape[0] != 1 or self.decision_result is None:
            raise RuntimeError("Decision serving requires exactly one complete request")
        if self.decision_scores is None:
            stock = super().compute_logits(hidden_states)
            scores = stock[0, self.decision_ids].double()
            if self.decision_request[2] != "raw":
                scores = scores / self.decision_spec["full_temperature"]
        else:
            scores = self.decision_scores
        probabilities = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Nonfinite decision probabilities")
        p = probabilities.cpu().tolist()
        actions = self.decision_spec["actions"]
        self.decision_result.update(
            action=actions[int(probabilities.argmax())],
            probabilities=dict(zip(actions, p)),
            confidence=max(p),
            label_logprobs=torch.log_softmax(scores, dim=-1).cpu().tolist(),
        )
        logits = torch.full(
            (1, self.config.text_config.vocab_size),
            -float("inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        logits[0, self.decision_ids] = scores.float()
        return logits


def arm_worker(worker, threshold, request_id, mode):
    worker.model_runner.model.arm_decision(threshold, request_id, mode)


def take_worker(worker):
    return worker.model_runner.model.take_decision()
