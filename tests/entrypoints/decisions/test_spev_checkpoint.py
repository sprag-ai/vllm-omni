# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Standalone Transformers checkpoint loading through the native SPEV model."""

import os
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm_omni.model_executor.models.registry import _OMNI_MODELS
from vllm_omni.model_executor.models.spev.config import SPEVConfig
from vllm_omni.model_executor.models.spev.model import SPEVForDecisions, SPEVProcessingInfo
from vllm_omni.model_executor.models.spev.pooler import OptionHead
from vllm_omni.model_executor.models.spev.processor import SPEVProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def checkpoint_config():
    return SimpleNamespace(
        model_type="spev",
        thinker_config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=4)),
        head_width=3,
        adapter_rank=2,
        adapter_alpha=4,
        temperature=1.3,
    )


def test_checkpoint_config_uses_saved_dimensions_and_tokenizer():
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [3, 4])
    cfg = SPEVConfig.from_hf(checkpoint_config(), tokenizer)
    assert (cfg.hidden_size, cfg.width, cfg.rank, cfg.alpha, cfg.temperature) == (4, 3, 2, 4, 1.3)
    assert cfg.marker_token_ids == [3, 4]
    with pytest.raises(ValueError, match="tokenizer"):
        SPEVConfig.from_hf(checkpoint_config())
    assert _OMNI_MODELS["SPEVModel"] == ("spev", "model", "SPEVForDecisions")
    info = object.__new__(SPEVProcessingInfo)
    config = checkpoint_config()
    info.ctx = SimpleNamespace(get_hf_config=lambda: config)
    assert info.get_hf_config() is config.thinker_config


def loader_model(monkeypatch):
    model = object.__new__(SPEVForDecisions)
    torch.nn.Module.__init__(model)
    model.standalone_checkpoint = True
    model.spev_config = SPEVConfig.from_hf(checkpoint_config(), SimpleNamespace(encode=lambda *args, **kwargs: [3, 4]))
    model.pooler = torch.nn.Module()
    model.pooler.head = OptionHead(4, 3)
    qkv = torch.nn.Linear(4, 12, bias=False)
    qkv.forward = lambda x: (torch.nn.functional.linear(x, qkv.weight), None)
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = qkv
    model.language_model = torch.nn.Module()
    model.language_model.model = torch.nn.Module()
    model.language_model.model.layers = torch.nn.ModuleList([layer])
    forwarded = {}

    def load(self, weights):
        forwarded.update(weights)
        return {"language_model.model.layers.0.self_attn.qkv_proj.weight"}

    monkeypatch.setattr(Qwen3OmniMoeThinkerForConditionalGeneration, "load_weights", load)
    head = OptionHead(4, 3).state_dict()
    tensors = {"head." + name: tensor for name, tensor in head.items()}
    for proj in ["q_proj", "v_proj"]:
        prefix = "backbone.model.layers.0.self_attn." + proj
        tensors[prefix + ".a"] = torch.randn(2, 4)
        tensors[prefix + ".b"] = torch.randn(4, 2)
        tensors[prefix + ".base.weight"] = torch.randn(4, 4)
    tensors["backbone.visual.test.weight"] = torch.randn(2, 2)
    return model, tensors, head, forwarded


def test_native_loader_preserves_backbone_head_and_fp32_residuals(monkeypatch):
    model, tensors, head, forwarded = loader_model(monkeypatch)
    loaded = model.load_weights(iter(tensors.items()))
    assert set(forwarded) == {
        "thinker.model.layers.0.self_attn.q_proj.weight",
        "thinker.model.layers.0.self_attn.v_proj.weight",
        "thinker.visual.test.weight",
    }
    for name, value in head.items():
        assert torch.equal(model.pooler.head.state_dict()[name], value)
    assert "pooler.head.scorer.0.weight" in loaded
    qkv = model.language_model.model.layers[0].self_attn.qkv_proj
    x = torch.randn(2, 4)
    expected = torch.nn.functional.linear(x, qkv.weight)
    for projection, begin in [("q_proj", 0), ("v_proj", 8)]:
        prefix = "backbone.model.layers.0.self_attn." + projection
        a, b = tensors[prefix + ".a"], tensors[prefix + ".b"]
        expected[:, begin : begin + 4] += torch.nn.functional.linear(torch.nn.functional.linear(x, a), b) * 2
    assert torch.equal(qkv(x)[0], expected)
    assert all(t.dtype == torch.float32 for t in model._spev_adapter.values())


@pytest.mark.parametrize("kind", ["head_dtype", "adapter_dtype", "missing_head", "unknown"])
def test_native_loader_rejects_invalid_checkpoint(monkeypatch, kind):
    model, tensors, _, _ = loader_model(monkeypatch)
    if kind == "head_dtype":
        tensors["head.scorer.0.weight"] = tensors["head.scorer.0.weight"].bfloat16()
    elif kind == "adapter_dtype":
        key = "backbone.model.layers.0.self_attn.q_proj.a"
        tensors[key] = tensors[key].bfloat16()
    elif kind == "missing_head":
        del tensors["head.scorer.0.weight"]
    else:
        tensors["unrelated.weight"] = torch.zeros(1)
    with pytest.raises((ValueError, RuntimeError)):
        model.load_weights(iter(tensors.items()))


def test_published_checkpoint_processor_loads_without_sidecar_assets():
    path = os.environ.get("SPEV_TEST_CHECKPOINT")
    if not path:
        pytest.skip("Set SPEV_TEST_CHECKPOINT for standalone checkpoint validation")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    spec = SPEVConfig.from_hf(config, tokenizer)
    assert (spec.hidden_size, spec.rank, spec.width) == (2048, 32, 256)
    model_config = SimpleNamespace(hf_config=config, tokenizer=path, tokenizer_revision=None, trust_remote_code=True)
    processor = SPEVProcessor(SimpleNamespace(model_config=model_config), None)
    assert processor.config.marker_token_ids == tokenizer.encode("<spev_score>\n", add_special_tokens=False)
    assert type(processor.processor).__name__ == "Qwen3OmniMoeProcessor"
    assert processor.config.video_fps == 1.0


def test_published_checkpoint_resolves_native_pooling_model():
    path = os.environ.get("SPEV_TEST_CHECKPOINT")
    if not path:
        pytest.skip("Set SPEV_TEST_CHECKPOINT for standalone checkpoint validation")
    from vllm.config import ModelConfig

    config = ModelConfig(
        model=path,
        trust_remote_code=True,
        runner="pooling",
        io_processor_plugin="spev_decisions",
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
    )
    assert config.architecture == "SPEVModel"
    assert config.runner_type == "pooling"
    assert config.hf_text_config.hidden_size == 2048

    from vllm.multimodal import MULTIMODAL_REGISTRY

    processor = MULTIMODAL_REGISTRY.create_processor(config)
    assert processor.info.get_hf_config().text_config.hidden_size == 2048
    assert type(processor.info.get_hf_processor()).__name__ == "Qwen3OmniMoeProcessor"
