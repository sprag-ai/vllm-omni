# SPDX-License-Identifier: Apache-2.0
"""Request schemas for the frozen native-audio decision chat API."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


class DecisionRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DecisionInputAudio(DecisionRequestModel):
    data: str = Field(max_length=4 * ((12 * 1024 * 1024 + 2) // 3))
    format: Literal["wav", "flac", "mp3", "ogg", "aiff"]


class DecisionAudioPart(DecisionRequestModel):
    type: Literal["input_audio"]
    input_audio: DecisionInputAudio


class DecisionTextPart(DecisionRequestModel):
    type: Literal["text"]
    text: str


class DecisionUserMessage(DecisionRequestModel):
    role: Literal["user"]
    content: list[Annotated[DecisionAudioPart | DecisionTextPart, Field(discriminator="type")]] = Field(
        min_length=1, max_length=2
    )

    @model_validator(mode="after")
    def require_one_audio(self):
        if sum(isinstance(part, DecisionAudioPart) for part in self.content) != 1:
            raise ValueError("Expected one audio clip and at most one task text part")
        return self


class DecisionChatRequest(DecisionRequestModel):
    model: str
    messages: list[DecisionUserMessage] = Field(min_length=1, max_length=1)
    max_tokens: int = Field(default=1, ge=1, le=1)
    max_completion_tokens: int = Field(default=1, ge=1, le=1)
    n: int = Field(default=1, ge=1, le=1)
    temperature: float = Field(default=0, ge=0, le=0)
    stream: StrictBool = False
    modalities: list[Literal["text"]] = Field(default_factory=lambda: ["text"], min_length=1, max_length=1)
    logprobs: StrictBool = False
    top_logprobs: int = Field(default=0, ge=0, le=3)
    user: str | None = None
    # Gateway metadata; this engine disables cross-request caches.
    cache_salt: str = ""

    @field_validator("stream")
    @classmethod
    def reject_streaming(cls, value):
        if value:
            raise ValueError("Only stream=false is supported")
        return value

    @model_validator(mode="after")
    def require_logprobs(self):
        if self.top_logprobs and not self.logprobs:
            raise ValueError("top_logprobs requires logprobs=true")
        return self

    @property
    def input_audio(self) -> DecisionInputAudio:
        return next(part.input_audio for part in self.messages[0].content if isinstance(part, DecisionAudioPart))


def decision_chat_request_type(model_name: str, prompt: str) -> type[DecisionChatRequest]:
    """Bind serving configuration to the edge schema without global model state."""

    class BoundDecisionChatRequest(DecisionChatRequest):
        model: Literal[model_name]

        @field_validator("messages")
        @classmethod
        def require_frozen_task(cls, messages):
            for part in messages[0].content:
                if isinstance(part, DecisionTextPart) and part.text not in ("audio_turn_decision", prompt):
                    raise ValueError(
                        "This frozen audio task accepts audio_turn_decision or its exact bundled prompt only"
                    )
            return messages

    return BoundDecisionChatRequest
