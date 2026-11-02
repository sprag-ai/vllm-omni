# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Decisions served through per-question chat requests."""

import asyncio
import math
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
)
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse, PromptTokenUsageInfo, UsageInfo

from vllm_omni.entrypoints.openai.serving_decisions import (
    LETTERS,
    OmniServingDecisions,
    chat_request,
    letter_probabilities,
    media_parts,
    question_text,
    usage,
)
from vllm_omni.inputs.decisions import DecisionsRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LETTER_IDS = list(range(32, 42))
SUFFIX = "\nAnswer with the option's letter from the given choices directly."


def decisions_request(questions, **kwargs):
    return DecisionsRequest.model_validate({"model": "spev", "input": "", "questions": questions, **kwargs})


def predicate(instructions="Has the speaker finished?", name=None):
    return {"type": "predicate", "name": name, "instructions": instructions}


def chat_response(letter_logprobs, prompt_tokens=100, cached=0, modalities=None):
    top = [ChatCompletionLogProb(token=f"token_id:{i}", logprob=lp) for i, lp in letter_logprobs.items()]
    content = ChatCompletionLogProbsContent(token=top[0].token, logprob=top[0].logprob, top_logprobs=top)
    return ChatCompletionResponse(
        model="spev",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="A"),
                logprobs=ChatCompletionLogProbs(content=[content]),
                finish_reason="length",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=1,
            total_tokens=prompt_tokens + 1,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=cached, multimodal_tokens=modalities),
        ),
    )


class FakeChat:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []
        self.engine_client = SimpleNamespace(errored=False)

    async def create_chat_completion(self, request):
        self.requests.append(request)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if callable(result):
            return await result()
        return result


def test_predicate_prompt_matches_the_trained_layout():
    request = decisions_request([predicate()])

    assert question_text(request, request.questions[0]) == (
        "Has the speaker finished?\nA. The assertion is true.\nB. The assertion is false." + SUFFIX
    )


def test_text_input_becomes_an_escaped_state_line():
    request = decisions_request([predicate()], input="caller said <hello>")

    assert question_text(request, request.questions[0]).startswith('State: "caller said \\u003chello\\u003e"\n')


def test_message_input_state_replaces_images_with_placeholders():
    image = "data:image/png;base64,AAAA"
    request = decisions_request(
        [predicate()],
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}, {"type": "input_image", "image_url": image}],
            }
        ],
    )

    text = question_text(request, request.questions[0])

    assert text.startswith(
        'State: [{"role": "user", "content": [{"type": "input_text", "text": "hi"}, {"type": "input_image"}]}]\n'
    )
    assert media_parts(request) == [{"type": "image_url", "image_url": {"url": image}}]


def test_choice_and_score_options_render_descriptions_values_and_levels():
    request = decisions_request(
        [
            {
                "type": "choice",
                "instructions": "Who answered?",
                "choices": [
                    {"value": "person", "description": "A live person."},
                    {"value": "voicemail"},
                    {"value": True},
                ],
            },
            {
                "type": "score",
                "instructions": "How loud?",
                "levels": [{"label": "quiet"}, {"label": "loud", "description": "Clipping or shouting."}],
            },
        ]
    )

    choice, score = (question_text(request, q) for q in request.questions)

    assert choice == "Who answered?\nA. A live person.\nB. voicemail\nC. true" + SUFFIX
    assert score == "How loud?\nA. quiet\nB. loud: Clipping or shouting." + SUFFIX


def test_more_options_than_letters_is_refused():
    choices = [{"value": str(i)} for i in range(len(LETTERS) + 1)]
    request = decisions_request([{"type": "choice", "instructions": "Pick", "choices": choices}])

    with pytest.raises(ValueError, match="at most 10 options"):
        question_text(request, request.questions[0])


def test_chat_request_reads_the_question_letters_after_the_evidence():
    request = decisions_request(
        [predicate()], input_audio={"data": "UklGRg==", "format": "wav"}, input_video={"data": "AAAA", "format": "mp4"}
    )

    built = chat_request(request, request.questions[0], LETTER_IDS, "decision-x-0")

    content = built.messages[0]["content"]
    assert [part["type"] for part in content] == ["text", "input_audio", "video_url"]
    assert content[1]["input_audio"] == {"data": "UklGRg==", "format": "wav"}
    assert content[2]["video_url"]["url"] == "data:video/mp4;base64,AAAA"
    assert (built.max_tokens, built.temperature, built.logprobs) == (1, 0.0, True)
    assert built.logprob_token_ids == [32, 33]
    assert built.return_tokens_as_token_ids is True
    assert built.request_id == "decision-x-0"


def test_letter_probabilities_renormalize_raw_letter_logprobs():
    response = chat_response({32: math.log(0.3), 33: math.log(0.1)})

    probabilities = letter_probabilities(response, [32, 33])

    assert probabilities == pytest.approx([0.75, 0.25])


def test_missing_letter_logprob_is_an_error():
    with pytest.raises(RuntimeError, match="lacks a letter"):
        letter_probabilities(chat_response({32: -0.1}), [32, 33])


def test_usage_sums_every_question_and_counts_no_output():
    responses = [
        chat_response({32: 0.0, 33: 0.0}, prompt_tokens=200, cached=40, modalities={"audio": 130}),
        chat_response({32: 0.0, 33: 0.0}, prompt_tokens=210, cached=40, modalities={"audio": 130}),
    ]

    result = usage(responses)

    assert (result.input_tokens, result.output_tokens, result.total_tokens) == (410, 0, 410)
    assert result.input_tokens_details.cached_tokens == 80
    assert result.input_tokens_details.multimodal_tokens == {"audio": 260}


def test_create_decisions_answers_each_question_in_order():
    request = decisions_request(
        [
            predicate(name="done"),
            {"type": "choice", "name": "who", "instructions": "Who?", "choices": [{"value": "a"}, {"value": "b"}]},
            {"type": "score", "name": "level", "instructions": "Level?", "levels": [{"label": "0"}, {"label": "1"}]},
        ]
    )
    chat = FakeChat(
        [
            chat_response({32: math.log(0.8), 33: math.log(0.2)}),
            chat_response({32: math.log(0.5), 33: math.log(0.5)}),
            chat_response({32: math.log(0.25), 33: math.log(0.75)}),
        ]
    )

    response = asyncio.run(OmniServingDecisions(chat, LETTER_IDS).create_decisions(request, "decision-r"))

    done, who, level = response.answers
    assert (done.name, done.probability) == ("done", pytest.approx(0.8))
    assert (who.choice, who.confidence) == ("a", 0.0)
    assert level.score == pytest.approx(0.75)
    assert [r.request_id for r in chat.requests] == ["decision-r-0", "decision-r-1", "decision-r-2"]
    assert response.usage.input_tokens == 300


@pytest.mark.parametrize(
    ("code", "status"),
    [(404, 404), (400, 422), (500, 500)],
    ids=["unknown-model", "bad-input", "server-error"],
)
def test_chat_errors_map_to_decision_statuses(code, status):
    error = ErrorResponse(error=ErrorInfo(message="x", type="x", code=code))
    chat = FakeChat([error])

    with pytest.raises(HTTPException) as caught:
        asyncio.run(OmniServingDecisions(chat, LETTER_IDS).create_decisions(decisions_request([predicate()]), "d"))

    assert caught.value.status_code == status


def test_a_failed_question_cancels_the_others():
    cancelled = asyncio.Event()

    async def slow():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def run():
        chat = FakeChat([slow, RuntimeError("boom")])
        with pytest.raises(HTTPException) as caught:
            await OmniServingDecisions(chat, LETTER_IDS).create_decisions(
                decisions_request([predicate(name="a"), predicate(name="b")]), "d"
            )
        await asyncio.sleep(0)
        return caught.value.status_code

    assert asyncio.run(run()) == 500
    assert cancelled.is_set()


def test_letters_must_be_distinct_single_tokens():
    with pytest.raises(ValueError, match="distinct token"):
        OmniServingDecisions(FakeChat([]), [32] * len(LETTERS))
