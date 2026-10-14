# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SPEV FP32 residual adapter and contextual candidate head."""

import torch
from safetensors.torch import load_file
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.spev.config import SPEVConfig
from vllm_omni.model_executor.models.spev.pooler import CandidatePooler, OptionHead


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class SPEVForDecisions(Qwen3OmniMoeThinkerForConditionalGeneration):
    is_pooling_model = True

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
        self.spev_config = SPEVConfig.from_hf(vllm_config.model_config.hf_config)
        self.spev_model_config = vllm_config.model_config
        head = OptionHead(self.spev_config.hidden_size, self.spev_config.width).float().cpu().eval()
        self.pooler = CandidatePooler(head, self.spev_config.marker_token_ids)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self.pooler.head.load_state_dict(
            load_file(str(self.spev_config.asset(self.spev_model_config, "head.safetensors")))
        )
        self.pooler.head.requires_grad_(False)
        device = next(self.language_model.parameters()).device
        tensors = load_file(
            str(self.spev_config.asset(self.spev_model_config, "adapter.safetensors")), device=str(device)
        )
        scale = self.spev_config.alpha / self.spev_config.rank
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
        return loaded | {"pooler.head." + name for name, _ in self.pooler.head.named_parameters()}
