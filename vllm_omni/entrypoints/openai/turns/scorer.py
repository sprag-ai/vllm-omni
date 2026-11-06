# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-of-turn readings through the served classification model."""

from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from vllm.entrypoints.pooling.classify.protocol import ClassificationChatRequest

from vllm_omni.entrypoints.openai.turns.protocol import SAMPLE_RATE_HZ

TURN_PROMPT_FILE = "turn_prompt.json"


class TurnReadingError(RuntimeError):
    """A reading failed inside the classification serving path."""


class TurnPrompt(BaseModel):
    """The model's end-of-turn question, shipped with its weights as ``turn_prompt.json``."""

    question: str = Field(description="question text after the context, including the lettered options")
    context_prefix: str = Field(description="label that introduces the agent's last utterance")
    window_seconds: float = Field(gt=0, le=30, description="seconds of caller audio each reading sees")
    max_context_chars: int = Field(gt=0, description="trailing characters of the agent's utterance kept")

    def text(self, context: str) -> str:
        """Render the text part of a reading; an empty context leaves the context line out."""
        context = context[-self.max_context_chars :]
        if not context:
            return self.question
        literal = json.dumps(context, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
        return self.context_prefix + literal + "\n" + self.question


def wav_base64(samples: np.ndarray) -> str:
    """Encode float32 16 kHz mono samples in [-1, 1] as a base64 PCM16 WAV file."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE_HZ)
        out.writeframes(pcm)
    return base64.b64encode(buffer.getvalue()).decode()


@dataclass(slots=True)
class TurnScorer:
    """Score end-of-turn probability for one audio window and context.

    Attributes:
        classification: The server's ``ServingClassification`` handler.
        model: Served model name sent with every reading.
        prompt: The model's question.
    """

    classification: Any
    model: str
    prompt: TurnPrompt

    def request(self, samples: np.ndarray, context: str, request_id: str) -> ClassificationChatRequest:
        """Build the classification request for one reading."""
        return ClassificationChatRequest(
            model=self.model,
            request_id=request_id,
            add_generation_prompt=True,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt.text(context)},
                        {"type": "input_audio", "input_audio": {"data": wav_base64(samples), "format": "wav"}},
                    ],
                }
            ],
        )

    async def score(self, samples: np.ndarray, context: str, request_id: str) -> float:
        """Return the probability that the turn has ended.

        Raises:
            TurnReadingError: The classification handler returned an error or an unexpected shape.
        """
        response = await self.classification(self.request(samples, context, request_id))
        body = json.loads(bytes(response.body))
        try:
            return float(body["data"][0]["probs"][0])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            message = body.get("error", {}).get("message") if isinstance(body, dict) else None
            raise TurnReadingError(message or "classification returned no probability") from exc
