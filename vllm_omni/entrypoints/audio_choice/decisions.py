# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The OpenAI Decisions wire contract with optional native audio evidence."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from vllm_omni.entrypoints.audio_choice.protocol import ChoiceInputAudio


class DecisionObject(BaseModel):
    """A strictly typed Decisions object."""

    model_config = ConfigDict(extra="forbid", strict=True)


class DecisionChoiceOption(DecisionObject):
    """One typed choice and its meaning."""

    value: str | bool = Field(description="distinct string or boolean choice value")
    description: str | None = Field(default=None, description="when this choice applies")


class DecisionLevel(DecisionObject):
    """One level in an ascending rubric."""

    label: str = Field(description="level label")
    description: str | None = Field(default=None, description="criteria for this level")


class DecisionPredicateQuestion(DecisionObject):
    """A condition to evaluate against the evidence."""

    type: Literal["predicate"] = Field(description="question primitive")
    name: str | None = Field(default=None, description="optional caller question name")
    instructions: str = Field(description="condition to evaluate")


class DecisionChoiceQuestion(DecisionObject):
    """A selection from distinct typed values."""

    type: Literal["choice"] = Field(description="question primitive")
    name: str | None = Field(default=None, description="optional caller question name")
    instructions: str = Field(description="selection instructions")
    choices: list[DecisionChoiceOption] = Field(min_length=2, max_length=255, description="allowed values and meanings")

    @model_validator(mode="after")
    def unique_choices(self) -> Self:
        """Reject duplicate typed choice values."""
        keys = [(type(option.value), option.value) for option in self.choices]
        if len(set(keys)) != len(keys):
            raise ValueError("Choice values must be unique")
        return self


class DecisionScoreQuestion(DecisionObject):
    """A rating against ordered rubric levels."""

    type: Literal["score"] = Field(description="question primitive")
    name: str | None = Field(default=None, description="optional caller question name")
    instructions: str = Field(description="rating instructions")
    levels: list[DecisionLevel] = Field(min_length=2, description="levels in ascending order starting at zero")


DecisionQuestion = Annotated[
    DecisionPredicateQuestion | DecisionChoiceQuestion | DecisionScoreQuestion, Field(discriminator="type")
]


class DecisionInputText(DecisionObject):
    """A text evidence part."""

    type: Literal["input_text"] = Field(description="input part type")
    text: str = Field(description="text evidence")


class DecisionInputImage(DecisionObject):
    """An image evidence part."""

    type: Literal["input_image"] = Field(description="input part type")
    image_url: str = Field(description="image data URL")
    detail: Literal["low", "high", "auto", "original"] | None = Field(default=None, description="image detail hint")


class DecisionInputMessage(DecisionObject):
    """An ordered user evidence message."""

    role: Literal["user"] = Field(description="evidence role")
    type: Literal["message"] | None = Field(default=None, description="message item type")
    content: str | list[Annotated[DecisionInputText | DecisionInputImage, Field(discriminator="type")]] = Field(
        description="text or ordered evidence parts"
    )


DecisionInput = str | list[DecisionInputMessage]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class DecisionPredicateAnswer(DecisionObject):
    """The estimated probability of a condition being true."""

    type: Literal["predicate"] = Field(description="answer primitive")
    name: str | None = Field(description="caller name or null")
    probability: Probability = Field(description="probability that the condition is true")


class DecisionChoiceProbability(DecisionObject):
    """A probability associated with a typed choice."""

    value: str | bool = Field(description="supplied choice value")
    probability: Probability = Field(description="normalized candidate probability")


class DecisionChoiceAnswer(DecisionObject):
    """A selected value and the distribution over supplied choices."""

    type: Literal["choice"] = Field(description="answer primitive")
    name: str | None = Field(description="caller name or null")
    choice: str | bool = Field(description="selected supplied value")
    probabilities: list[DecisionChoiceProbability] = Field(min_length=2, description="probabilities in choice order")
    confidence: Probability = Field(description="confidence in the selection")

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        """Validate distinct values, normalization and the selected maximum."""
        keys = [(type(p.value), p.value) for p in self.probabilities]
        selected = (type(self.choice), self.choice)
        if len(set(keys)) != len(keys) or selected not in keys:
            raise ValueError("Invalid choice distribution values")
        values = [p.probability for p in self.probabilities]
        if abs(sum(values) - 1) > 1e-7 or values[keys.index(selected)] != max(values):
            raise ValueError("Invalid choice distribution probabilities")
        return self


class DecisionScoreProbability(DecisionObject):
    """A probability associated with an ordered level."""

    value: int = Field(ge=0, description="zero-based level index")
    label: str = Field(description="supplied level label")
    probability: Probability = Field(description="normalized level probability")


class DecisionScoreAnswer(DecisionObject):
    """The probability-weighted mean of the level indices."""

    type: Literal["score"] = Field(description="answer primitive")
    name: str | None = Field(description="caller name or null")
    score: float = Field(ge=0, allow_inf_nan=False, description="expected zero-based level index")
    probabilities: list[DecisionScoreProbability] = Field(min_length=2, description="probabilities in level order")
    confidence: Probability = Field(description="confidence in the rating")

    @model_validator(mode="after")
    def valid_distribution(self) -> Self:
        """Validate consecutive levels, normalization and the expected score."""
        if [p.value for p in self.probabilities] != list(range(len(self.probabilities))):
            raise ValueError("Score levels must be consecutive from zero")
        if abs(sum(p.probability for p in self.probabilities) - 1) > 1e-7:
            raise ValueError("Score probabilities must sum to one")
        if abs(self.score - sum(p.value * p.probability for p in self.probabilities)) > 1e-7:
            raise ValueError("Score must equal the expected level")
        return self


class DecisionRefusal(DecisionObject):
    """A question the model declined to answer."""

    type: Literal["refusal"] = Field(description="answer primitive")
    name: str | None = Field(description="caller name or null")


DecisionAnswer = Annotated[
    DecisionPredicateAnswer | DecisionChoiceAnswer | DecisionScoreAnswer | DecisionRefusal, Field(discriminator="type")
]


class DecisionInputTokensDetails(DecisionObject):
    """Input cache accounting and optional native modality counts."""

    cached_tokens: int = Field(ge=0, description="input tokens read from cache")
    cache_write_tokens: int = Field(ge=0, description="input tokens written to cache")
    multimodal_tokens: dict[Literal["audio", "image", "video"], Annotated[int, Field(ge=0)]] | None = Field(
        default=None, description="native modality token counts, when supplied"
    )


class DecisionOutputTokensDetails(DecisionObject):
    """Output reasoning accounting."""

    reasoning_tokens: int = Field(ge=0, description="reasoning output tokens")


class DecisionUsage(DecisionObject):
    """The backend's logical token accounting."""

    input_tokens: int = Field(ge=0, description="total logical input tokens")
    input_tokens_details: DecisionInputTokensDetails = Field(description="input cache and modality breakdown")
    output_tokens: int = Field(ge=0, description="generated output tokens")
    output_tokens_details: DecisionOutputTokensDetails = Field(description="output token breakdown")
    total_tokens: int = Field(ge=0, description="sum of input and output tokens")

    @model_validator(mode="after")
    def valid_totals(self) -> Self:
        """Reject inconsistent token totals and attribution."""
        details = self.input_tokens_details
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Inconsistent token total")
        if details.cached_tokens + details.cache_write_tokens > self.input_tokens:
            raise ValueError("Cache counts exceed input tokens")
        if sum((details.multimodal_tokens or {}).values()) > self.input_tokens:
            raise ValueError("Modality counts exceed input tokens")
        if self.output_tokens_details.reasoning_tokens > self.output_tokens:
            raise ValueError("Reasoning count exceeds output tokens")
        return self


class DecisionsResponse(DecisionObject):
    """Ordered answers and token usage for a Decisions request."""

    model: str = Field(description="backend model identifier")
    answers: list[DecisionAnswer] = Field(min_length=1, description="answers in request order")
    usage: DecisionUsage = Field(description="backend logical token usage")


class DecisionsRequest(DecisionObject):
    """Shared evidence and ordered decision questions."""

    model: str = Field(min_length=1, description="model identifier or alias")
    input: DecisionInput = Field(description="shared text or user evidence messages")
    questions: list[DecisionQuestion] = Field(min_length=1, description="questions in answer order")
    safety_identifier: str | None = Field(default=None, max_length=128, description="opaque end-user safety identifier")
    input_audio: ChoiceInputAudio | None = Field(
        default=None, description="Sprag extension: inline native audio evidence"
    )

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        """Reject duplicate non-null question names."""
        names = [q.name for q in self.questions if q.name is not None]
        if len(set(names)) != len(names):
            raise ValueError("Question names must be unique when supplied")
        return self

    @property
    def carries_media(self) -> bool:
        """Whether the evidence includes native audio or an image part."""
        return self.input_audio is not None or (
            isinstance(self.input, list)
            and any(
                isinstance(message.content, list)
                and any(isinstance(part, DecisionInputImage) for part in message.content)
                for message in self.input
            )
        )
