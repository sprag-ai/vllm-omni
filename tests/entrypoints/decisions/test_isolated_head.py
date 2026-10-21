# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decision layout isolation across questions and engine request batches."""

import os

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoConfig
from vllm import PoolingParams
from vllm.config import ModelConfig, VllmConfig
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from vllm_omni.inputs.decisions import DecisionsRequest
from vllm_omni.model_executor.models.spev.candidate_tree import CandidateTree
from vllm_omni.model_executor.models.spev.decision_head import IsolatedDecisionHead
from vllm_omni.model_executor.models.spev.decision_layout import DecisionLayout
from vllm_omni.model_executor.models.spev.isolated_attention import IsolatedAttention, isolated_batch
from vllm_omni.model_executor.models.spev.pooler import IsolatedCandidatePooler, decision_layouts
from vllm_omni.model_executor.models.spev.processor import SPEVProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def layout():
    return DecisionLayout(CandidateTree(3, (3, 3)), (5, 8), ((3,), (6,)), ((20, 21), (22, 23)), (0, 1), (0, 1))


def metadata(layouts):
    lengths = torch.tensor([item.tree.length for item in layouts])
    ids = torch.ones(len(layouts), lengths.max(), dtype=torch.long)
    params = [PoolingParams(task="plugin", extra_kwargs={"spev_layout": item.to_dict()}) for item in layouts]
    result = PoolingMetadata(lengths, ids, ids, params, [PoolingStates() for _ in layouts])
    counts = np.array(lengths.tolist())
    starts = torch.tensor(np.concatenate(([0], counts.cumsum())))
    result.build_pooling_cursor(counts, lengths, torch.device("cpu"), query_start_loc_gpu=starts)
    return result


class TupleLinear(nn.Linear):
    def forward(self, value):
        return super().forward(value), None


class IdentityRope(nn.Module):
    def forward(self, positions, query, key):
        return query, key


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = TupleLinear(8, 16, bias=False)
        self.o_proj = TupleLinear(8, 8, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.rotary_emb = IdentityRope()
        self.attn = nn.Identity()
        self.head_dim, self.q_size, self.kv_size = 4, 8, 4
        self.num_heads, self.num_kv_heads, self.scaling = 2, 1, 0.5


def attention():
    return IsolatedAttention(TinyAttention())


def test_attention_batch_and_question_isolation():
    torch.manual_seed(78)
    model = attention().eval()
    item = layout()
    first, second = torch.randn(9, 8), torch.randn(9, 8)
    with isolated_batch([item, item]):
        joint = model(torch.arange(18), torch.cat([first, second]))
    with isolated_batch([item]):
        alone = model(torch.arange(9), first)
        modified = first.clone()
        modified[6:] += 100
        changed = model(torch.arange(9), modified)
    torch.testing.assert_close(joint[:9], alone, atol=1e-6, rtol=0)
    torch.testing.assert_close(changed[:6], alone[:6], atol=0, rtol=0)
    assert not torch.equal(changed[6:], alone[6:])


def test_batched_pooler_calls_head_once_and_preserves_request_order():
    torch.manual_seed(78)
    head = IsolatedDecisionHead(8, 8, 2, 16, 2).eval()
    embedding = nn.Embedding(32, 8)
    pooler = IsolatedCandidatePooler(head, lambda ids: embedding(ids))
    first, second = torch.randn(9, 8), torch.randn(9, 8)
    count = []
    handle = head.register_forward_hook(lambda *args: count.append(1))
    joint = pooler(torch.cat([first, second]), metadata([layout(), layout()]))
    assert len(count) == 1
    handle.remove()
    alone = pooler(first, metadata([layout()]))
    reversed_batch = pooler(torch.cat([second, first]), metadata([layout(), layout()]))
    torch.testing.assert_close(joint[0], alone[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(joint[0], reversed_batch[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(joint[1], reversed_batch[0], atol=1e-6, rtol=0)


def test_layout_rebases_multimodal_expansion_and_requires_metadata():
    original = layout()
    expanded = original.rebase(original.tree.length + 100)
    assert expanded.tree.prefix_length == original.tree.prefix_length + 100
    assert expanded.anchors == tuple(index + 100 for index in original.anchors)
    assert expanded.lexical_ids == original.lexical_ids
    data = metadata([original])
    data.pooling_params[0].extra_kwargs = None
    with pytest.raises(ValueError, match="processor-generated"):
        decision_layouts(data)


def test_checkpoint_processor_keeps_metadata_on_its_own_request():
    path = os.environ.get("SPEV_TEST_CHECKPOINT")
    if not path:
        pytest.skip("Set SPEV_TEST_CHECKPOINT for isolated checkpoint validation")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    if config.head_architecture != "isolated_evidence_v1":
        pytest.skip("Requires an isolated SPEV head checkpoint")
    model_config = object.__new__(ModelConfig)
    model_config.hf_config = config
    model_config.tokenizer = path
    model_config.tokenizer_revision = None
    model_config.trust_remote_code = True
    vllm_config = object.__new__(VllmConfig)
    vllm_config.model_config = model_config
    processor = SPEVProcessor(vllm_config, None)
    requests = [
        DecisionsRequest.model_validate(
            dict(
                model="spev",
                input="red",
                questions=[
                    dict(type="choice", instructions="What color?", choices=[dict(value="red"), dict(value="blue")])
                ],
            )
        ),
        DecisionsRequest.model_validate(
            dict(model="spev", input="blue", questions=[dict(type="predicate", instructions="Is it blue?")])
        ),
    ]
    parsed = [processor.parse_data(request) for request in requests]
    with pytest.raises(ValueError, match="preprocessed"):
        processor.prepare_pooling_params(parsed[0])
    inputs = [processor.pre_process(prompt) for prompt in parsed]
    params = [processor.prepare_pooling_params(prompt) for prompt in parsed]
    assert params[0] is not params[1]
    assert params[0].extra_kwargs["spev_layout"]["question_types"] == (1,)
    assert params[1].extra_kwargs["spev_layout"]["question_types"] == (0,)
    for prompt, param in zip(inputs, params):
        value = DecisionLayout.from_dict(param.extra_kwargs["spev_layout"])
        value.validate(len(prompt["prompt_token_ids"]))
