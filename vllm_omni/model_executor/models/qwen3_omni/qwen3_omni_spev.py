# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SPEV FP32 residual adapter and contextual candidate head."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)


class OptionHead(nn.Module):
    def __init__(self, hidden_size, width):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width), nn.GELU(), nn.Linear(width, 1)
        )

    def forward(self, features):
        return self.scorer(features.float()).squeeze(-1)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniSPEVThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("SPEV supports TP=1 and PP=1")
        if (
            vllm_config.scheduler_config.enable_chunked_prefill
            or vllm_config.cache_config.enable_prefix_caching
            or not vllm_config.model_config.enforce_eager
        ):
            raise ValueError("SPEV requires complete eager prefills without prefix caching")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.spev_root = Path(vllm_config.model_config.hf_config.choice_bundle)
        self.spec = json.loads((self.spev_root / "spev_config.json").read_text())
        self.batch_provider = None
        self.completed_scores = {}
        # Match the reference scorer's CPU FP32 head, independent of GPU batch shape.
        self.spev_head = OptionHead(self.spec["hidden_size"], self.spec["width"]).float().cpu().eval()

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self.spev_head.load_state_dict(load_file(str(self.spev_root / "head.safetensors")))
        self.spev_head.requires_grad_(False)
        device = next(self.language_model.parameters()).device
        tensors = load_file(str(self.spev_root / "adapter.safetensors"), device=str(device))
        scale = self.spec["alpha"] / self.spec["rank"]
        self._spev_adapter = tensors
        self._spev_handles = []
        consumed = set()
        for i, layer in enumerate(self.language_model.model.layers):
            keys = [
                f"model.layers.{i}.self_attn.{proj}.{which}" for proj in ("q_proj", "v_proj") for which in ("a", "b")
            ]
            qa, qb, va, vb = [tensors[k] for k in keys]
            if any(t.dtype != torch.float32 for t in (qa, qb, va, vb)):
                raise ValueError("SPEV adapter requires FP32 tensors")
            consumed.update(keys)

            def apply(module, args, output, qa=qa, qb=qb, va=va, vb=vb):
                result, bias = output
                x = args[0].float()
                qn, vn = qb.shape[0], vb.shape[0]
                if result.shape[-1] != qn + 2 * vn:
                    raise RuntimeError("Unexpected SPEV QKV layout")
                for start, end, a, b in ((0, qn, qa, qb), (qn + vn, qn + 2 * vn, va, vb)):
                    residual = torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * scale
                    result[..., start:end] += residual.to(result.dtype)
                return result, bias

            self._spev_handles.append(layer.self_attn.qkv_proj.register_forward_hook(apply))
        if consumed != set(tensors):
            raise ValueError("SPEV adapter has unbound tensors")
        return loaded | {"spev_head." + name for name, _ in self.spev_head.named_parameters()}

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if self.batch_provider is None:
            return hidden
        requests = self.batch_provider()
        if not requests or hidden.ndim != 2 or sum(r["input_tokens"] for r in requests) != hidden.shape[0]:
            raise RuntimeError("SPEV rows do not match scheduler prefills")
        offset = 0
        for request in requests:
            key = request["request_id"]
            if key in self.completed_scores:
                raise RuntimeError("Duplicate SPEV result")
            anchors = [offset + p for p in request["positions"]]
            scores = self.spev_head(hidden[anchors].float().cpu())
            if not torch.isfinite(scores).all():
                raise RuntimeError("Nonfinite SPEV head scores")
            self.completed_scores[key] = {"request_id": key, "scores": scores.tolist(), "batch_size": len(requests)}
            offset += request["input_tokens"]
        return hidden

    def compute_logits(self, hidden_states):
        if self.batch_provider is None:
            return super().compute_logits(hidden_states)
        # The scheduler completes each prefill with one discarded token.
        logits = torch.full(
            (hidden_states.shape[0], self.config.text_config.vocab_size),
            -float("inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        logits[:, 0] = 0
        return logits
