# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SPEV FP32 residual adapter and contextual candidate head."""

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder as TransformersAudioEncoder,
)
from vllm import envs
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)
from vllm_omni.model_executor.models.spev.attention_backend import checkpoint_attention
from vllm_omni.model_executor.models.spev.checkpoint_ops import (
    CheckpointDecoderLayer,
    CheckpointExperts,
    CheckpointLinear,
    CheckpointQKV,
    CheckpointRotary,
)
from vllm_omni.model_executor.models.spev.config import SPEVConfig
from vllm_omni.model_executor.models.spev.decision_head import IsolatedDecisionHead
from vllm_omni.model_executor.models.spev.isolated_attention import IsolatedAttention, isolated_batch
from vllm_omni.model_executor.models.spev.normalization import CheckpointRMSNorm
from vllm_omni.model_executor.models.spev.pooler import (
    CandidatePooler,
    IsolatedCandidatePooler,
    OptionHead,
    decision_layouts,
)
from vllm_omni.model_executor.models.spev.vision import CheckpointVisionEncoder

MISSING_POOLING_METADATA = object()


class SPEVProcessingInfo(Qwen3OmniMoeThinkerProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config().thinker_config


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=SPEVProcessingInfo,
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
        hf_config = vllm_config.model_config.hf_config
        self.standalone_checkpoint = hf_config.model_type == "spev"
        tokenizer = None
        if self.standalone_checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(
                vllm_config.model_config.tokenizer,
                revision=vllm_config.model_config.tokenizer_revision,
                trust_remote_code=vllm_config.model_config.trust_remote_code,
                local_files_only=True,
            )
            self.language_model.lm_head = torch.nn.Identity()
        self.spev_config = SPEVConfig.from_hf(hf_config, tokenizer)
        self.spev_model_config = vllm_config.model_config
        self.supports_pooling_metadata = self.spev_config.architecture == "isolated_evidence_v1"
        if self.supports_pooling_metadata:
            config = self.spev_config
            if envs.VLLM_BATCH_INVARIANT:
                raise ValueError("SPEV isolated checkpoints require VLLM_BATCH_INVARIANT=0")
            if vllm_config.model_config.dtype != torch.bfloat16 or vllm_config.quant_config is not None:
                raise ValueError("SPEV isolated checkpoints require unquantized BF16 weights")
            if envs.VLLM_FLOAT32_MATMUL_PRECISION != "highest":
                raise ValueError("SPEV isolated checkpoints require highest FP32 matmul precision")
            thinker_config = hf_config.thinker_config if self.standalone_checkpoint else hf_config
            thinker_config.audio_config._attn_implementation = "sdpa"
            self.audio_tower = TransformersAudioEncoder(thinker_config.audio_config)
            thinker_config.vision_config._attn_implementation = "sdpa"
            self.visual = CheckpointVisionEncoder(thinker_config.vision_config)
            head = IsolatedDecisionHead(
                config.hidden_size, config.width, config.head_heads, config.head_ff, config.head_depth
            )
            self.pooler = IsolatedCandidatePooler(head.float().eval(), self._embed_criteria)
            for layer in self.language_model.model.layers:
                layer.input_layernorm = CheckpointRMSNorm(layer.input_layernorm)
                layer.post_attention_layernorm = CheckpointRMSNorm(layer.post_attention_layernorm)
                layer.self_attn.q_norm = CheckpointRMSNorm(layer.self_attn.q_norm)
                layer.self_attn.k_norm = CheckpointRMSNorm(layer.self_attn.k_norm)
                layer.self_attn = IsolatedAttention(layer.self_attn)
                attention = layer.self_attn
                attention.qkv_proj = CheckpointQKV(attention.qkv_proj, attention.q_size, attention.kv_size)
                attention.o_proj = CheckpointLinear(attention.o_proj)
                attention.rotary_emb = CheckpointRotary(
                    thinker_config.text_config, attention.qkv_proj.weight.device, attention.head_dim
                )
                layer.mlp = CheckpointExperts(layer.mlp, thinker_config.text_config)
            for index in range(len(thinker_config.vision_config.deepstack_visual_indexes)):
                self.language_model.model.layers[index] = CheckpointDecoderLayer(
                    self.language_model.model.layers[index]
                )
            self.language_model.model.norm = CheckpointRMSNorm(self.language_model.model.norm)
        else:
            head = OptionHead(self.spev_config.hidden_size, self.spev_config.width).float().cpu().eval()
            self.pooler = CandidatePooler(head, self.spev_config.marker_token_ids)

    def _embed_criteria(self, ids):
        return self.language_model.model.embed_tokens(ids)

    def _process_audio_input(self, audio_input):
        if not self.supports_pooling_metadata:
            return super()._process_audio_input(audio_input)
        features = audio_input["input_features"].to(self.audio_tower.dtype)
        lengths = audio_input["audio_feature_lengths"].to(features.device)
        outputs, offset = [], 0
        with checkpoint_attention():
            for index, length in enumerate(lengths.tolist()):
                outputs.append(
                    self.audio_tower(
                        features[:, offset : offset + length], feature_lens=lengths[index : index + 1]
                    ).last_hidden_state
                )
                offset += length
        if offset != features.shape[-1]:
            raise ValueError("SPEV audio features do not match their sequence lengths")
        return tuple(outputs)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        pooling_metadata=MISSING_POOLING_METADATA,
        **kwargs,
    ):
        if not self.supports_pooling_metadata:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if pooling_metadata is MISSING_POOLING_METADATA:
            raise RuntimeError("This SPEV checkpoint requires vLLM pooling metadata input support")
        if pooling_metadata is None:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        layouts = decision_layouts(pooling_metadata)
        if sum(layout.tree.length for layout in layouts) != positions.shape[-1]:
            raise ValueError("SPEV requires complete prefills for every batched request")
        positions = positions.clone()
        offset = 0
        for layout in layouts:
            tree = layout.tree
            prefix = positions[:, offset : offset + tree.prefix_length][:, None, :]
            positions[:, offset : offset + tree.length] = tree.position_ids(prefix)[:, 0]
            offset += tree.length
        with isolated_batch(layouts), checkpoint_attention():
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)

    def load_weights(self, weights):
        device = next(self.language_model.parameters()).device
        if self.standalone_checkpoint:
            tensors, head = {}, {}

            def backbone_weights():
                for name, tensor in weights:
                    if name.startswith("head."):
                        if tensor.dtype != torch.float32:
                            raise ValueError("SPEV head requires FP32 tensors")
                        head[name.removeprefix("head.")] = tensor
                    elif name.startswith("backbone."):
                        name = name.removeprefix("backbone.")
                        if name.endswith((".q_proj.a", ".q_proj.b", ".v_proj.a", ".v_proj.b")):
                            tensors[name] = tensor.to(device=device)
                        else:
                            name = name.replace(".q_proj.base.", ".q_proj.").replace(".v_proj.base.", ".v_proj.")
                            yield "thinker." + name, tensor
                    else:
                        raise ValueError(f"Unexpected SPEV checkpoint tensor: {name}")

            loaded = super().load_weights(backbone_weights())
        else:
            loaded = super().load_weights(weights)
            head = load_file(str(self.spev_config.asset(self.spev_model_config, "head.safetensors")))
            tensors = load_file(
                str(self.spev_config.asset(self.spev_model_config, "adapter.safetensors")), device=str(device)
            )
        self.pooler.head.load_state_dict(head)
        self.pooler.head.requires_grad_(False)
        if self.spev_config.architecture == "isolated_evidence_v1":
            self.pooler.head.to(device=device, dtype=torch.float32)
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
            if self.spev_config.architecture == "isolated_evidence_v1":
                layer.self_attn.qkv_proj.bind_adapters(qa, qb, va, vb, scale)
                continue

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
