# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import copy

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.audio_choice.contract import (
    ChoiceResponse,
    InputTokensDetails,
    MultimodalTokens,
    Usage,
    primitive_answer,
    render_question,
    scoring_question,
    target,
)
from vllm_omni.entrypoints.audio_choice.decisions import DecisionsRequest
from vllm_omni.entrypoints.audio_choice.decisions_adapter import choice_keys, from_choice, to_choice

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def payload():
    return {
        "model": "spev",
        "input": "invoice",
        "questions": [
            {
                "name": "route",
                "type": "choice",
                "instructions": "Choose a department",
                "choices": [{"value": "billing", "description": "Payments"}, {"value": "technical"}],
            }
        ],
    }


def evaluate(request, modalities=None):
    native = to_choice(request)
    return from_choice(
        request,
        ChoiceResponse(
            model="spev",
            answers={
                key: primitive_answer(q, {name: -float(i) for i, name in enumerate(scoring_question(q).criteria)})
                for key, q in native.questions.items()
            },
            usage=Usage(
                input_tokens=42,
                output_tokens=0,
                input_tokens_details=InputTokensDetails(
                    multimodal_tokens=MultimodalTokens(**modalities) if modalities else None,
                ),
            ),
        ),
    )


def test_named_targets_prompt_and_question_names():
    request = DecisionsRequest.model_validate(payload())
    native = to_choice(request)
    assert native.state == "invoice"
    assert native.questions["0"].criteria == {"billing": "Payments", "technical": None}
    changed = copy.deepcopy(request)
    changed.questions[0].name = "PRIVATE NAME"
    assert to_choice(changed).model_dump() == native.model_dump()
    assert "PRIVATE NAME" not in render_question(native.questions["0"], native.state)
    assert target("billing") == '{"choice":"billing"}'


def test_ordered_mixed_answers_and_typed_boolean_collision():
    data = payload()
    data["questions"] = [
        {"type": "predicate", "instructions": "Urgent?"},
        {
            "name": "route",
            "type": "choice",
            "instructions": "Choose",
            "choices": [
                {"value": True},
                {"value": "true"},
                {"value": "_true"},
                {"value": False},
                {"value": "false"},
            ],
        },
        {
            "type": "score",
            "instructions": "Rate",
            "levels": [
                {"label": "low", "description": "routine"},
                {"label": "high", "description": "urgent"},
            ],
        },
    ]
    request = DecisionsRequest.model_validate(data)
    assert choice_keys(request.questions[1]) == ["__true", "true", "_true", "_false", "false"]
    answers = evaluate(request).model_dump()["answers"]
    assert [a["type"] for a in answers] == ["predicate", "choice", "score"]
    assert [a["name"] for a in answers] == [None, "route", None]
    assert answers[1]["choice"] is True
    assert [p["value"] for p in answers[1]["probabilities"]] == [True, "true", "_true", False, "false"]
    assert answers[0]["probability"] > 0.5
    assert [p["label"] for p in answers[2]["probabilities"]] == ["low", "high"]
    assert answers[2]["score"] == pytest.approx(answers[2]["probabilities"][1]["probability"])


@pytest.mark.parametrize("modalities", [None, {"audio": 10}, {"image": 4}])
def test_canonical_usage(modalities):
    result = evaluate(DecisionsRequest.model_validate(payload()), modalities).model_dump()
    assert result["usage"] == {
        "input_tokens": 42,
        "output_tokens": 0,
        "total_tokens": 42,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0, "multimodal_tokens": modalities},
        "output_tokens_details": {"reasoning_tokens": 0},
    }


@pytest.mark.parametrize(
    "change",
    [
        {"questions": []},
        {"questions": {}},
        {"input": 5},
        {"stream": True},
        {"questions": [{"type": "predicate", "name": "x", "instructions": "Check"}] * 2},
        {"questions": [{"type": "choice", "instructions": "Pick", "choices": [{"value": True}, {"value": True}]}]},
        {"questions": [{"type": "choice", "instructions": "Pick", "choices": [{"value": 1}, {"value": "a"}]}]},
        {"questions": [{"type": "score", "instructions": "Rate", "levels": [{"label": "one"}]}]},
        {"input": [{"role": "assistant", "content": "test"}]},
        {"input_audio": {"data": "PRIVATE AUDIO", "format": "wav"}},
    ],
)
def test_invalid_contract(change):
    with pytest.raises(ValidationError):
        DecisionsRequest.model_validate({**payload(), **change})


def test_audio_extension_and_text_message_order():
    data = payload()
    data["input_audio"] = {"data": "QUJD", "format": "wav"}
    data["input"] = [
        {"role": "user", "content": "first"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "second"},
                {"type": "input_text", "text": "third"},
            ],
        },
    ]
    native = to_choice(DecisionsRequest.model_validate(data))
    assert native.input_audio.data == "QUJD"
    assert native.state == data["input"]


def test_inline_image_is_removed_from_text_prompt():
    data = payload()
    data["input"] = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Look"},
                {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
            ],
        }
    ]
    native = to_choice(DecisionsRequest.model_validate(data))
    assert native.input_image.data == "QUJD" and native.input_image.format == "png"
    assert native.state[0]["content"] == [{"type": "input_text", "text": "Look"}, {"type": "input_image"}]


@pytest.mark.parametrize(
    "parts",
    [
        [{"type": "input_image", "image_url": "https://example.com/a.png"}],
        [{"type": "input_image", "image_url": "data:image/png;base64,QUJD", "detail": "high"}],
        [{"type": "input_image", "image_url": "data:image/png;base64,QUJD"}] * 2,
    ],
)
def test_unsupported_images_are_explicitly_rejected(parts):
    with pytest.raises(ValueError):
        to_choice(DecisionsRequest.model_validate({**payload(), "input": [{"role": "user", "content": parts}]}))


@pytest.mark.parametrize(
    "change",
    [
        {"input": "x" * 65537},
        {"questions": [{"type": "predicate", "instructions": "Check"}] * 17},
        {"questions": [{"type": "score", "instructions": "Rate", "levels": [{"label": "x"}] * 11}]},
    ],
)
def test_native_work_limits_still_apply(change):
    with pytest.raises(ValidationError):
        to_choice(DecisionsRequest.model_validate({**payload(), **change}))


@pytest.mark.parametrize("length", [64, 65, 128, 129])
def test_safety_identifier_limit(length):
    data = {**payload(), "safety_identifier": "x" * length}
    if length > 128:
        with pytest.raises(ValidationError):
            DecisionsRequest.model_validate(data)
    else:
        request = DecisionsRequest.model_validate(data)
        assert request.safety_identifier == "x" * length
        assert to_choice(request).model_dump() == to_choice(DecisionsRequest.model_validate(payload())).model_dump()
