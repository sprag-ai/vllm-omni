# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Adapt Decisions requests to the named-choice scorer without changing its targets."""

from vllm_omni.entrypoints.audio_choice.contract import ChoiceResponse
from vllm_omni.entrypoints.audio_choice.decisions import (
    DecisionChoiceQuestion,
    DecisionInputImage,
    DecisionInputText,
    DecisionPredicateQuestion,
    DecisionsRequest,
    DecisionsResponse,
)
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceInputMedia, ChoiceRequest


def choice_keys(question: DecisionChoiceQuestion) -> list[str]:
    """Preserve string targets and assign collision-free names to boolean targets."""
    used = {option.value for option in question.choices if isinstance(option.value, str)}
    keys = []
    for option in question.choices:
        if isinstance(option.value, str):
            key = option.value
        else:
            key = "true" if option.value else "false"
            while key in used:
                key = "_" + key
            used.add(key)
        keys.append(key)
    return keys


def to_choice(request: DecisionsRequest) -> ChoiceRequest:
    """Validate model-specific limits and construct a native scoring request."""
    questions = {}
    for index, question in enumerate(request.questions):
        if isinstance(question, DecisionChoiceQuestion):
            criteria = {}
            for key, option in zip(choice_keys(question), question.choices):
                description = option.description
                if isinstance(option.value, bool):
                    description = {"value": option.value, "description": description}
                criteria[key] = description
            primitive = {"type": "choice", "criteria": criteria}
        elif isinstance(question, DecisionPredicateQuestion):
            primitive = {"type": "noul"}
        else:
            primitive = {
                "type": "score",
                "criteria": [level.model_dump(exclude_none=True) for level in question.levels],
            }
        questions[str(index)] = {**primitive, "instructions": question.instructions}

    image = None
    state = request.input
    if isinstance(request.input, list):
        state = []
        for message in request.input:
            if isinstance(message.content, str):
                state.append({"role": "user", "content": message.content})
                continue
            content = []
            for part in message.content:
                if isinstance(part, DecisionInputText):
                    content.append(part.model_dump())
                elif isinstance(part, DecisionInputImage):
                    if image is not None:
                        raise ValueError("This model supports at most one image per request")
                    if part.detail not in (None, "auto"):
                        raise ValueError("This model supports only automatic image detail")
                    header, separator, data = part.image_url.partition(",")
                    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
                        raise ValueError("This model requires an inline base64 image data URL")
                    image = ChoiceInputMedia(data=data, format=header[11:-7])
                    content.append({"type": "input_image"})
            state.append({"role": "user", "content": content})
    return ChoiceRequest(
        model=request.model,
        state=state,
        questions=questions,
        input_audio=request.input_audio,
        input_image=image,
    )


def from_choice(request: DecisionsRequest, response: ChoiceResponse) -> DecisionsResponse:
    """Restore ordered typed answers and canonical usage field names."""
    answers = []
    for index, question in enumerate(request.questions):
        result = response.answers[str(index)]
        answer = {"type": question.type, "name": question.name}
        if isinstance(question, DecisionPredicateQuestion):
            answer["probability"] = result.noul
        elif isinstance(question, DecisionChoiceQuestion):
            keys = choice_keys(question)
            answer.update(
                choice=next(option.value for key, option in zip(keys, question.choices) if key == result.choice),
                confidence=result.confidence,
                probabilities=[
                    {"value": option.value, "probability": result.probabilities[key]}
                    for key, option in zip(keys, question.choices)
                ],
            )
        else:
            answer.update(
                score=result.score,
                confidence=result.confidence,
                probabilities=[
                    {"value": i, "label": level.label, "probability": result.probabilities[str(i)]}
                    for i, level in enumerate(question.levels)
                ],
            )
        answers.append(answer)
    usage = response.usage.model_dump()
    details = usage["input_tokens_details"]
    if details["multimodal_tokens"] is not None:
        details["multimodal_tokens"] = {k: v for k, v in details["multimodal_tokens"].items() if v is not None}
    details["cache_write_tokens"] = 0
    usage["output_tokens_details"] = {"reasoning_tokens": 0}
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return DecisionsResponse(model=response.model, answers=answers, usage=usage)
