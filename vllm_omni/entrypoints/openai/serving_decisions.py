# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decisions answered by a chat model's next-token scores over each question's lettered options.

Each question becomes one chat request: the question and its lettered options, then the evidence, then the assistant
turn. The answer distribution is the softmax of the model's raw log-probabilities for the option letters.
"""

import asyncio
import json
import math
import string
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import load_aware_call, with_cancellation
from vllm.logger import init_logger

from vllm_omni.inputs.decisions import (
    DecisionAnswer,
    DecisionChoiceAnswer,
    DecisionChoiceQuestion,
    DecisionInputImage,
    DecisionInputText,
    DecisionInputTokensDetails,
    DecisionOutputTokensDetails,
    DecisionPredicateAnswer,
    DecisionPredicateQuestion,
    DecisionQuestion,
    DecisionScoreAnswer,
    DecisionsRequest,
    DecisionsResponse,
    DecisionUsage,
)

logger = init_logger(__name__)

LETTERS = string.ascii_uppercase
ANSWER_INSTRUCTION = "\nAnswer with the option's letter from the given choices directly."
TOKEN_ID_PREFIX = "token_id:"


def literal(value: object) -> str:
    """Serialize a value as JSON with angle brackets escaped so it cannot open a chat-template tag."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


def options(question: DecisionQuestion) -> list[str]:
    """Return the option texts of a question in answer order."""
    if isinstance(question, DecisionPredicateQuestion):
        return ["The assertion is true.", "The assertion is false."]
    if isinstance(question, DecisionChoiceQuestion):
        return [
            choice.description
            if choice.description is not None
            else choice.value
            if isinstance(choice.value, str)
            else literal(choice.value)
            for choice in question.choices
        ]
    return [
        level.label if level.description is None else f"{level.label}: {level.description}" for level in question.levels
    ]


def state(request: DecisionsRequest) -> object:
    """Return the request's text evidence, with each image replaced by a placeholder part."""
    if isinstance(request.input, str):
        return request.input
    return [
        {
            "role": message.role,
            "content": message.content
            if isinstance(message.content, str)
            else [
                part.model_dump() if isinstance(part, DecisionInputText) else {"type": "input_image"}
                for part in message.content
            ],
        }
        for message in request.input
    ]


def question_text(request: DecisionsRequest, question: DecisionQuestion) -> str:
    """Render the text that precedes the evidence: optional state, the question and its lettered options.

    Raises:
        ValueError: The question has more options than there are letters.
    """
    texts = options(question)
    if len(texts) > len(LETTERS):
        raise ValueError(f"A question accepts at most {len(LETTERS)} options")
    evidence = state(request)
    prefix = "State: " + literal(evidence) + "\n" if evidence else ""
    lettered = "\n".join(f"{letter}. {text}" for letter, text in zip(LETTERS, texts))
    return prefix + question.instructions + "\n" + lettered + ANSWER_INSTRUCTION


def media_parts(request: DecisionsRequest) -> list[dict]:
    """Return the request's audio, image and video evidence as chat content parts, in that order."""
    parts: list[dict] = []
    if request.input_audio is not None:
        parts.append(
            {
                "type": "input_audio",
                "input_audio": {"data": request.input_audio.data, "format": request.input_audio.format},
            }
        )
    if isinstance(request.input, list):
        parts += [
            {"type": "image_url", "image_url": {"url": part.image_url}}
            for message in request.input
            if not isinstance(message.content, str)
            for part in message.content
            if isinstance(part, DecisionInputImage)
        ]
    if request.input_video is not None:
        url = f"data:video/{request.input_video.format};base64,{request.input_video.data}"
        parts.append({"type": "video_url", "video_url": {"url": url}})
    return parts


def chat_request(
    request: DecisionsRequest, question: DecisionQuestion, letter_ids: list[int], request_id: str
) -> ChatCompletionRequest:
    """Build the one-token chat request whose letter log-probabilities answer ``question``."""
    text = question_text(request, question)
    return ChatCompletionRequest(
        model=request.model,
        messages=[{"role": "user", "content": [{"type": "text", "text": text}, *media_parts(request)]}],
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        logprob_token_ids=letter_ids[: len(options(question))],
        return_tokens_as_token_ids=True,
        request_id=request_id,
    )


def letter_probabilities(response: ChatCompletionResponse, letter_ids: list[int]) -> list[float]:
    """Return the softmax over the given letters' log-probabilities at the first generated position.

    Raises:
        RuntimeError: The response lacks a log-probability for one of the letters.
    """
    content = response.choices[0].logprobs.content if response.choices[0].logprobs is not None else None
    if not content:
        raise RuntimeError("The chat response carries no log-probabilities")
    scores = {
        int(entry.token.removeprefix(TOKEN_ID_PREFIX)): entry.logprob
        for entry in content[0].top_logprobs
        if entry.token.startswith(TOKEN_ID_PREFIX)
    }
    if any(letter not in scores for letter in letter_ids):
        raise RuntimeError("The chat response lacks a letter log-probability")
    values = [scores[letter] for letter in letter_ids]
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    return [weight / sum(weights) for weight in weights]


def answer(question: DecisionQuestion, probabilities: list[float]) -> DecisionAnswer:
    """Build the typed answer for a question from its option distribution."""
    n = len(probabilities)
    if isinstance(question, DecisionPredicateQuestion):
        return DecisionPredicateAnswer(type="predicate", name=question.name, probability=probabilities[0])
    if isinstance(question, DecisionChoiceQuestion):
        peak = max(probabilities)
        selected = min(
            (i for i, p in enumerate(probabilities) if p == peak),
            key=lambda i: (type(question.choices[i].value).__name__, str(question.choices[i].value)),
        )
        return DecisionChoiceAnswer(
            type="choice",
            name=question.name,
            choice=question.choices[selected].value,
            probabilities=[{"value": c.value, "probability": p} for c, p in zip(question.choices, probabilities)],
            confidence=max(0.0, min(1.0, (peak - 1 / n) / (1 - 1 / n))),
        )
    peak = probabilities.index(max(probabilities))
    spread = sum(p * abs(i - peak) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return DecisionScoreAnswer(
        type="score",
        name=question.name,
        score=sum(i * p for i, p in enumerate(probabilities)),
        confidence=max(0.0, 1 - spread / even_spread),
        probabilities=[
            {"value": i, "label": level.label, "probability": probabilities[i]}
            for i, level in enumerate(question.levels)
        ],
    )


def usage(responses: list[ChatCompletionResponse]) -> DecisionUsage:
    """Sum the prompt usage of every question's chat request; the generated letter is not counted as output."""
    input_tokens = sum(r.usage.prompt_tokens for r in responses)
    cached = 0
    modalities: dict[str, int] = {}
    for response in responses:
        details = response.usage.prompt_tokens_details
        if details is None:
            continue
        cached += details.cached_tokens or 0
        for modality, count in (details.multimodal_tokens or {}).items():
            modalities[modality] = modalities.get(modality, 0) + count
    return DecisionUsage(
        input_tokens=input_tokens,
        output_tokens=0,
        total_tokens=input_tokens,
        input_tokens_details=DecisionInputTokensDetails(
            cached_tokens=cached, cache_write_tokens=0, multimodal_tokens=modalities or None
        ),
        output_tokens_details=DecisionOutputTokensDetails(reasoning_tokens=0),
    )


class OmniServingDecisions:
    """Serves Decisions requests through the chat completion path, one chat request per question."""

    def __init__(self, chat, letter_ids: list[int]):
        if len(letter_ids) != len(LETTERS) or len(set(letter_ids)) != len(LETTERS):
            raise ValueError("Each option letter must be one distinct token")
        self.chat = chat
        self.letter_ids = letter_ids

    async def create_decisions(self, request: DecisionsRequest, request_id: str) -> DecisionsResponse:
        """Answer every question of ``request`` concurrently.

        Raises:
            HTTPException: 404 for an unknown model, 422 for input the model cannot take, 503 when the engine is
                dead, 500 for any other failure.
        """
        try:
            requests = [
                chat_request(request, question, self.letter_ids, f"{request_id}-{index}")
                for index, question in enumerate(request.questions)
            ]
        except (ValueError, ValidationError) as exc:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "Invalid or unsupported model input") from exc
        tasks = [asyncio.create_task(self.chat.create_chat_completion(r)) for r in requests]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException as exc:
            for task in tasks:
                task.cancel()
            if isinstance(exc, Exception) and not isinstance(exc, HTTPException):
                logger.error("Decision inference failed: %s", type(exc).__name__)
                status = (
                    HTTPStatus.SERVICE_UNAVAILABLE
                    if self.chat.engine_client.errored
                    else HTTPStatus.INTERNAL_SERVER_ERROR
                )
                raise HTTPException(status, "Decision inference failed") from exc
            raise
        for result in results:
            if isinstance(result, ErrorResponse):
                code = result.error.code
                if code == HTTPStatus.NOT_FOUND:
                    raise HTTPException(HTTPStatus.NOT_FOUND, "Unknown model")
                if HTTPStatus.BAD_REQUEST <= code < HTTPStatus.INTERNAL_SERVER_ERROR:
                    raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "Invalid or unsupported model input")
                raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "Decision inference failed")
            if not isinstance(result, ChatCompletionResponse):
                raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "Decision inference failed")
        try:
            answers = [
                answer(question, letter_probabilities(result, self.letter_ids[: len(options(question))]))
                for question, result in zip(request.questions, results)
            ]
            return DecisionsResponse(model=request.model, answers=answers, usage=usage(results))
        except (RuntimeError, ValidationError) as exc:
            logger.error("Invalid decision model output: %s", type(exc).__name__)
            raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "Invalid decision model output") from exc


async def letter_token_ids(chat) -> list[int]:
    """Return the token id of each option letter in the chat model's tokenizer.

    Raises:
        ValueError: A letter does not encode as exactly one token.
    """
    tokenizer = await chat.engine_client.get_tokenizer()
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError("Each option letter must encode as one token")
    return [i[0] for i in ids]


async def mount_decisions(app: FastAPI) -> OmniServingDecisions:
    """Register ``POST /v1/decisions`` on an app whose chat serving is initialized.

    Raises:
        ValueError: The server does not serve chat completions or does not report prompt token details.
    """
    chat = getattr(app.state, "openai_serving_chat", None)
    if chat is None:
        raise ValueError("--decisions requires a model served through chat completions")
    if not chat.enable_prompt_tokens_details:
        raise ValueError("--decisions requires --enable-prompt-tokens-details so usage reports cached and media tokens")
    serving = OmniServingDecisions(chat, await letter_token_ids(chat))

    @with_cancellation
    @load_aware_call
    async def create_decisions(body: dict[str, Any], raw_request: Request):
        try:
            request = DecisionsRequest.model_validate(body)
        except ValidationError:
            return JSONResponse(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY, content={"error": {"message": "Invalid decision request"}}
            )
        request_id = "decision-" + chat._base_request_id(raw_request)
        response = await serving.create_decisions(request, request_id)
        return JSONResponse(content=response.model_dump(mode="json"))

    app.add_api_route("/v1/decisions", create_decisions, methods=["POST"], response_model=DecisionsResponse)
    return serving
