# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bound the internal named-choice scoring payload."""

import base64
import binascii
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from vllm_omni.entrypoints.audio_choice.contract import (
    ChoiceQuestion,
    Description,
    NoulQuestion,
    ScoreQuestion,
    scoring_question,
)
from vllm_omni.entrypoints.openai.decision_protocol import (
    DecisionInputAudio,
    DecisionRequestModel,
)


class PublicChoiceQuestion(ChoiceQuestion):
    type: Literal["choice"]


class PublicNoulQuestion(NoulQuestion):
    type: Literal["noul"]


class PublicScoreQuestion(ScoreQuestion):
    type: Literal["score"]


PublicQuestion = Annotated[PublicChoiceQuestion | PublicNoulQuestion | PublicScoreQuestion, Field(discriminator="type")]


class ChoiceInputAudio(DecisionInputAudio):
    """SPEV extension; the upstream TypeSafe reference has no audio transport."""

    @field_validator("data")
    @classmethod
    def valid_base64(cls, data):
        try:
            if not base64.b64decode(data, validate=True):
                raise ValueError("Empty audio payload")
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Expected nonempty base64 audio bytes") from exc
        return data


class ChoiceInputMedia(DecisionRequestModel):
    data: str = Field(min_length=1, max_length=16 * 1024 * 1024)
    format: str = Field(min_length=1)

    @field_validator("data")
    @classmethod
    def valid_base64(cls, data):
        try:
            if not base64.b64decode(data, validate=True):
                raise ValueError("Empty media payload")
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Expected nonempty base64 media bytes") from exc
        return data


class ChoicePayload(DecisionRequestModel):
    state: Description
    questions: dict[str, PublicQuestion] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def bound_work(self):
        if sum(len(scoring_question(q).criteria) for q in self.questions.values()) > 255:
            raise ValueError("At most 255 total criteria per request")
        # Bound textual work separately from the larger base64-audio body limit.
        pending = [self.state]
        for key, question in self.questions.items():
            pending.extend((key, question.instructions, question.criteria))
        characters = nodes = 0
        while pending:
            value = pending.pop()
            nodes += 1
            if isinstance(value, str):
                characters += len(value)
            elif isinstance(value, dict):
                if len(value) > 4096:
                    raise ValueError("Choice text structure exceeds 4096 nodes")
                pending.extend(value.keys())
                pending.extend(value.values())
            elif isinstance(value, list):
                if len(value) > 4096:
                    raise ValueError("Choice text structure exceeds 4096 nodes")
                pending.extend(value)
            if nodes > 4096 or characters > 65536:
                raise ValueError("Choice text exceeds 65536 characters or 4096 structure nodes")
        return self


class ChoiceRequest(ChoicePayload):
    model: str = Field(min_length=1)
    input_audio: ChoiceInputAudio | None = None
    input_image: ChoiceInputMedia | None = None
    input_video: ChoiceInputMedia | None = None
