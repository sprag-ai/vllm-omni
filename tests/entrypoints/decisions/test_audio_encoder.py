# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Checkpoint-faithful audio encoding without cross-request batching effects."""

import pytest
import torch
from transformers import Qwen3OmniMoeAudioEncoderConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from vllm.model_executor.models.qwen3_omni_moe_thinker import _get_feat_extract_output_lengths
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.spev.model import SPEVForDecisions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def encoder():
    config = Qwen3OmniMoeAudioEncoderConfig(
        encoder_layers=1, d_model=64, encoder_attention_heads=4, encoder_ffn_dim=128, output_dim=64
    )
    config._attn_implementation = "sdpa"
    return Qwen3OmniMoeAudioEncoder(config).eval()


def test_native_loader_binds_unfused_audio_checkpoint_tensors():
    torch.manual_seed(78)
    source = encoder()
    model = torch.nn.Module()
    model.audio_tower = encoder()
    values = {"audio_tower." + name: value for name, value in source.state_dict().items()}
    loaded = AutoWeightsLoader(model).load_weights(iter(values.items()))
    assert set(dict(model.named_parameters())) <= loaded
    for name, value in model.state_dict().items():
        assert torch.equal(value, values[name])


def test_each_audio_is_encoded_once_and_matches_standalone():
    torch.manual_seed(78)
    model = object.__new__(SPEVForDecisions)
    torch.nn.Module.__init__(model)
    model.supports_pooling_metadata = True
    model.audio_tower = encoder()
    lengths = torch.tensor([80, 160])
    features = torch.randn(128, 240)
    expected = []
    offset = 0
    with torch.inference_mode():
        for index, length in enumerate(lengths.tolist()):
            lens = lengths[index : index + 1]
            expected.append(
                model.audio_tower(
                    features[:, offset : offset + length],
                    feature_lens=lens,
                    aftercnn_lens=_get_feat_extract_output_lengths(lens),
                ).last_hidden_state
            )
            offset += length
    calls = []
    model.audio_tower.register_forward_hook(lambda *args: calls.append(1))
    with torch.inference_mode():
        actual = model._process_audio_input({"input_features": features, "audio_feature_lengths": lengths})
    assert len(calls) == len(actual) == len(expected) == 2
    for measured, reference in zip(actual, expected):
        torch.testing.assert_close(measured, reference, atol=0, rtol=0)
