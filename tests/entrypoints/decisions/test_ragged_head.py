# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ragged evidence projection preserves question-local scores and gradients."""

import copy

import pytest
import torch

from vllm_omni.model_executor.models.spev.decision_head import (
    IsolatedDecisionHead,
    QuestionFeatures,
    score_questions,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def records():
    torch.manual_seed(943)
    return [
        QuestionFeatures(
            torch.randn(length, 8, requires_grad=True),
            torch.randn(8, requires_grad=True),
            torch.randn(options, 8, requires_grad=True),
            torch.randn(options, 8, requires_grad=True),
            torch.randn(8, requires_grad=True),
            kind,
        )
        for length, options, kind in [(3, 2, 0), (129, 3, 1), (5, 2, 2)]
    ]


def head():
    torch.manual_seed(944)
    model = IsolatedDecisionHead(8, 8, 2, 16, 2).eval()
    with torch.no_grad():
        model.norm.bias.uniform_(-0.5, 0.5)
    return model


def test_memory_projection_excludes_padding_and_calls_head_once():
    model, inputs, shapes, calls = head(), records(), [], []
    model.memory.register_forward_pre_hook(lambda _, args: shapes.append(tuple(args[0].shape)))
    model.register_forward_hook(lambda *_: calls.append(1))
    actual = score_questions(model, inputs)
    assert shapes == [(137, 8)]
    assert calls == [1]
    assert [len(value) for value in actual] == [2, 3, 2]


def test_ragged_batch_matches_independent_scores_and_parameter_gradients():
    batched, independent, inputs = head(), head(), records()
    single_inputs = copy.deepcopy(inputs)
    joint = score_questions(batched, inputs)
    alone = [score_questions(independent, [record])[0] for record in single_inputs]
    for actual, expected in zip(joint, alone):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)
    sum(value.square().sum() for value in joint).backward()
    sum(value.square().sum() for value in alone).backward()
    for actual, expected in zip(batched.parameters(), independent.parameters()):
        assert (actual.grad is None) == (expected.grad is None)
        if actual.grad is not None:
            torch.testing.assert_close(actual.grad, expected.grad, atol=2e-5, rtol=1e-5)
    for actual, expected in zip(inputs, single_inputs):
        torch.testing.assert_close(actual.memory.grad, expected.memory.grad, atol=2e-6, rtol=1e-5)


def test_ragged_memory_has_no_other_question_gradient():
    model, inputs = head(), records()
    result = score_questions(model, inputs)
    result[0].sum().backward()
    assert inputs[0].memory.grad.abs().sum() > 0
    for record in inputs[1:]:
        assert record.memory.grad is None or torch.count_nonzero(record.memory.grad) == 0
