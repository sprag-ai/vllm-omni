# SPDX-License-Identifier: Apache-2.0
import math

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.audio_choice.contract import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    primitive_answer,
    score_confidence,
    scoring_question,
)
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceChatRequest, ChoiceRequest


def test_noul_is_probability_not_boolean_or_generated_number():
    question = NoulQuestion(instructions="Is the customer asking for a refund?")
    result = primitive_answer(question, {"true": math.log(0.8), "false": math.log(0.2)})
    assert result.model_dump() == {"type": "noul", "noul": 0.8}
    assert scoring_question(question).criteria == {"true": "The assertion is true.", "false": "The assertion is false."}


def test_noul_preserves_descriptions():
    question = NoulQuestion(instructions="test", criteria={"true": {"a": 1}, "false": ["absent"]})
    assert scoring_question(question).criteria == question.criteria


@pytest.mark.parametrize("criteria", [{"true": "yes"}, {"yes": "y", "no": "n"}, {}])
def test_noul_requires_both_descriptions(criteria):
    with pytest.raises(ValidationError):
        NoulQuestion(instructions="test", criteria=criteria)


@pytest.mark.parametrize(
    "p,expected", [([0, 0.5, 0.5], 0.25), ([0.5, 0, 0.5], 0), ([0, 0.57, 0.43], 0.355), ([0, 1, 0], 1)]
)
def test_score_reference_confidence(p, expected):
    assert score_confidence(p) == pytest.approx(expected)


def test_score_expected_level_legend_and_numeric_order():
    question = ScoreQuestion(instructions="Satisfaction", criteria=["bad", {"neutral": True}, ["good"]])
    result = primitive_answer(question, {"2": math.log(0.6), "0": math.log(0.1), "1": math.log(0.3)})
    assert result.score == pytest.approx(1.5)
    assert result.legend == {"0": "bad", "1": {"neutral": True}, "2": ["good"]}
    assert result.probabilities == pytest.approx({"0": 0.1, "1": 0.3, "2": 0.6})
    assert result.confidence == pytest.approx(0.25)


@pytest.mark.parametrize("criteria", [["one"], ["x"] * 11, {"0": "a", "1": "b"}])
def test_score_contract_rejects_invalid_scales(criteria):
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="test", criteria=criteria)


def test_choice_conversion_is_identity():
    question = ChoiceQuestion(instructions="test", criteria={"a": None, "b": "B"})
    assert scoring_question(question) is question


def test_mixed_questions_and_media_round_trip_chat():
    import json

    questions = {
        "a": {"type": "choice", "instructions": "name", "criteria": {"red": None, "blue": None}},
        "b": {"type": "noul", "instructions": "red?"},
        "c": {"type": "score", "instructions": "redness", "criteria": ["none", "some", "all"]},
    }
    payload = {"state": {}, "questions": questions}
    media = {
        kind: {"format": fmt, "data": "QUJD"}
        for kind, fmt in (("input_audio", "wav"), ("input_image", "png"), ("input_video", "mp4"))
    }
    native = ChoiceRequest(model="spev", **payload, **media)
    chat = ChoiceChatRequest(
        model="spev",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": json.dumps(payload)},
                    *[{"type": k, k: v} for k, v in media.items()],
                ],
            }
        ],
    )
    assert chat.to_choice() == native


@pytest.mark.parametrize("kind", ["input_image", "input_video"])
def test_media_rejects_urls_and_invalid_base64(kind):
    payload = dict(model="spev", state="test", questions={"q": {"type": "noul", "instructions": "yes?"}})
    for invalid in ({"url": "file:///etc/passwd"}, {"data": "not base64", "format": "png"}):
        with pytest.raises(ValidationError):
            ChoiceRequest(**payload, **{kind: invalid})
