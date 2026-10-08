# SPDX-License-Identifier: Apache-2.0
"""Full-depth native vLLM Thinker with the trained FP32 Q/V adapter."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_decision_base import apply_qv_adapter
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniChoiceThinker(Qwen3OmniMoeThinkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        if (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError("Choice FP32 adapter requires TP=1 and PP=1")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.choice_root = Path(vllm_config.model_config.hf_config.choice_bundle)
        self.use_deepstack = False
        self.deepstack_input_embeds = []

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        root = self.choice_root / "adapter"
        config = json.loads((root / "adapter_config.json").read_text())
        scale = config["lora_alpha"] / config["r"]
        device = next(self.language_model.parameters()).device
        tensors = load_file(str(root / "adapter_model.safetensors"), device=str(device))
        self._choice_lora = tensors
        self._choice_handles = []
        consumed = set()
        for i, layer in enumerate(self.language_model.model.layers):
            keys = [
                f"base_model.model.model.layers.{i}.self_attn.{proj}.lora_{which}.weight"
                for proj in ("q_proj", "v_proj")
                for which in ("A", "B")
            ]
            qa, qb, va, vb = [tensors[k] for k in keys]
            if any(t.dtype != torch.float32 for t in (qa, qb, va, vb)):
                raise ValueError("Choice adapter must preserve FP32 tensors")
            consumed.update(keys)

            def apply(module, args, output, qa=qa, qb=qb, va=va, vb=vb):
                return apply_qv_adapter(output, args[0], qa, qb, va, vb, scale)

            self._choice_handles.append(layer.self_attn.qkv_proj.register_forward_hook(apply))
        if consumed != set(tensors):
            raise ValueError("Choice adapter has unbound tensors")
        return loaded
