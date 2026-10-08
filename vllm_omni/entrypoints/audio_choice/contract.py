"""Typed Choice boundary; model scores supply probabilities, never generated numbers."""

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Description = str | dict[str, JsonValue] | list[JsonValue]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice"] = "choice"
    instructions: Description
    criteria: dict[str, Description | None] = Field(min_length=2, max_length=255)


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability] = Field(min_length=2, max_length=255)
    confidence: Probability

    @model_validator(mode="after")
    def valid_distribution(self):
        p = self.probabilities
        if abs(sum(p.values()) - 1) > 1e-7:
            raise ValueError("Probabilities must sum to one")
        if self.choice not in p or p[self.choice] != max(p.values()):
            raise ValueError("Choice must name a maximum-probability criterion")
        expected = (max(p.values()) - 1 / len(p)) / (1 - 1 / len(p))
        if abs(self.confidence - expected) > 1e-7:
            raise ValueError("Confidence must follow the documented Choice formula")
        return self


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ChoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    answers: dict[str, ChoiceAnswer] = Field(min_length=1)
    usage: Usage


def literal_json(value):
    # Qwen control and multimodal tokens use angle brackets. JSON Unicode
    # escapes preserve caller strings/keys without letting the tokenizer turn
    # their literal spelling into template or audio control tokens.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e")


def render_question(question: ChoiceQuestion, state=None):
    # Question ID is intentionally not an argument and never reaches the model.
    payload = {"instructions": question.instructions, "criteria": question.criteria}
    if state is not None:
        payload["state"] = state
    return (
        "Evaluate the supplied evidence for this Choice question. "
        "Select exactly one named criterion. Return only a JSON object with "
        'the key "choice" and its exact criterion name as the value, then stop. '
        "Do not return letter aliases, explanations or invented probabilities.\n" + literal_json(payload)
    )


def target(choice: str):
    return literal_json({"choice": choice})


def answer(question: ChoiceQuestion, log_scores: dict[str, float], temperature=1.0):
    if set(log_scores) != set(question.criteria):
        raise ValueError("Scores must cover exactly the supplied criteria")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    if not all(math.isfinite(x) for x in log_scores.values()):
        raise ValueError("Nonfinite score")
    z = {k: v / temperature for k, v in log_scores.items()}
    peak = max(z.values())
    weights = {k: math.exp(v - peak) for k, v in z.items()}
    denom = sum(weights.values())
    probabilities = {k: v / denom for k, v in weights.items()}
    # Stable tie handling independent of the caller's map insertion order.
    choice = min(probabilities, key=lambda k: (-probabilities[k], k))
    n = len(probabilities)
    confidence = max(0.0, min(1.0, (probabilities[choice] - 1 / n) / (1 - 1 / n)))
    return ChoiceAnswer(choice=choice, probabilities=probabilities, confidence=confidence)
