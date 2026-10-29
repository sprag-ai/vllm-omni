# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batched audio encoding whose per-recording features ignore the rest of the batch."""

import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import audio_conv2d_batch_invariant

from .invariant_ops import invariant_linear


class InvariantLinear(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.bias = source.bias

    def forward(self, value):
        if value.is_cuda:
            return invariant_linear(value, self.weight, self.bias), None
        return F.linear(value, self.weight, self.bias), None


class InvariantConv2d(nn.Conv2d):
    @classmethod
    def from_conv(cls, source):
        conv = cls(
            source.in_channels,
            source.out_channels,
            source.kernel_size,
            source.stride,
            source.padding,
            device="meta",
        )
        conv.weight = source.weight
        conv.bias = source.bias
        return conv

    def forward(self, value):
        if value.is_cuda:
            return audio_conv2d_batch_invariant(value, self)
        return super().forward(value)


class InvariantEncoderAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version

        self.source = source
        self.scale = source.scale
        self.version = get_flash_attn_version() or 2

    def forward(self, query, key, value, cu_seqlens, max_seqlen):
        if not query.is_cuda:
            return self.source(query=query, key=key, value=value, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        longest = int(max_seqlen)
        return flash_attn_varlen_func(
            query[0],
            key[0],
            value[0],
            max_seqlen_q=longest,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=longest,
            cu_seqlens_k=cu_seqlens,
            softmax_scale=self.scale,
            num_splits=1,
            fa_version=self.version,
        )


def install_invariant_audio(tower):
    """Replace the audio tower's projections, convolutions and attention with batch-invariant kernels."""
    for name in ("conv2d1", "conv2d2", "conv2d3"):
        setattr(tower, name, InvariantConv2d.from_conv(getattr(tower, name)))
    for name in ("conv_out", "proj1", "proj2"):
        setattr(tower, name, InvariantLinear(getattr(tower, name)))
    for layer in tower.layers:
        layer.self_attn.qkv_proj = InvariantLinear(layer.self_attn.qkv_proj)
        layer.self_attn.out_proj = InvariantLinear(layer.self_attn.out_proj)
        layer.self_attn.attn = InvariantEncoderAttention(layer.self_attn.attn)
        layer.fc1 = InvariantLinear(layer.fc1)
        layer.fc2 = InvariantLinear(layer.fc2)


def encode_audio(tower, features, feature_lens, aftercnn_lens):
    """Encode recordings in groups that share their standalone chunk padding.

    Returns:
        One feature tensor per recording, in input order.
    """
    lengths = feature_lens.tolist()
    groups = {}
    for index, length in enumerate(lengths):
        groups.setdefault(min(length, tower.n_window * 2), []).append(index)
    recordings = features.split(lengths, dim=-1)
    outputs = [None] * len(lengths)
    for indices in groups.values():
        encoded = tower._forward_same_padding(
            torch.cat([recordings[i] for i in indices], dim=-1), feature_lens[indices], aftercnn_lens[indices]
        )
        for index, output in zip(indices, encoded.split(aftercnn_lens[indices].tolist())):
            outputs[index] = output
    return tuple(outputs)
