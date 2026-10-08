# SPDX-License-Identifier: Apache-2.0
"""Validate Choice requests at the HTTP edge, including the chat envelope."""

import base64
import binascii
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from vllm_omni.entrypoints.audio_choice.contract import ChoiceQuestion, Description
from vllm_omni.entrypoints.openai.decision_protocol import (
    DecisionInputAudio,
    DecisionRequestModel,
    DecisionTextPart,
)


class PublicChoiceQuestion(ChoiceQuestion):
    type: Literal["choice"]


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


class ChoiceAudioPart(DecisionRequestModel):
    type: Literal["input_audio"]
    input_audio: ChoiceInputAudio


class ChoicePayload(DecisionRequestModel):
    state: Description
    questions: dict[str, PublicChoiceQuestion] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def bound_work(self):
        if sum(len(q.criteria) for q in self.questions.values()) > 255:
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


class ChoiceMessage(DecisionRequestModel):
    role: Literal["user"]
    content: list[Annotated[ChoiceAudioPart | DecisionTextPart, Field(discriminator="type")]] = Field(
        min_length=1, max_length=2
    )

    @model_validator(mode="after")
    def validate_parts(self):
        texts = [p for p in self.content if isinstance(p, DecisionTextPart)]
        if len(texts) != 1:
            raise ValueError("Exactly one text part containing a ChoicePayload JSON object is required")
        ChoicePayload.model_validate_json(texts[0].text)
        return self


class ChoiceChatRequest(DecisionRequestModel):
    model: str = Field(min_length=1)
    messages: list[ChoiceMessage] = Field(min_length=1, max_length=1)
    stream: Literal[False] = False
    temperature: float = Field(default=0, ge=0, le=0)
    n: int = Field(default=1, ge=1, le=1)
    # No sampled text is generated; generation limits are rejected.
    max_tokens: Literal[None] = None
    max_completion_tokens: Literal[None] = None
    user: str | None = None
    cache_salt: str = ""

    def to_choice(self):
        parts = self.messages[0].content
        payload = ChoicePayload.model_validate_json(next(p.text for p in parts if isinstance(p, DecisionTextPart)))
        audio = next((p.input_audio for p in parts if isinstance(p, ChoiceAudioPart)), None)
        return ChoiceRequest(
            model=self.model, input_audio=audio.model_dump() if audio is not None else None, **payload.model_dump()
        )
