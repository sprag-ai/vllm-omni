# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chat usage: the text stage's tokens whatever the request returns, built in one overridable place."""

from unittest.mock import MagicMock

import pytest
from vllm.entrypoints.serve.engine.protocol import UsageInfo

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_request,
    make_text_omni_output,
    parse_sse_chunks,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT_TOKENS = 3
TEXT_USAGE = {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": 3, "total_tokens": PROMPT_TOKENS + 3}
STAGE_METRICS = {"stage_metrics": {"0": {"num_tokens_out": 3}, "1": {"num_tokens_out": 221}}}


def audio_output(*, finished: bool, metrics: dict | None = None) -> OmniRequestOutput:
    audio = make_text_omni_output(token_ids=[], finish_reason="stop" if finished else None, final_output_type="audio")
    if metrics is not None:
        audio.metrics = metrics
    return audio


def streamed_outputs():
    return [
        make_text_omni_output(token_ids=[10, 11]),
        make_text_omni_output(token_ids=[12], finish_reason="stop"),
        audio_output(finished=False),
        audio_output(finished=True, metrics=STAGE_METRICS),
    ]


def full_outputs():
    return [
        make_text_omni_output(token_ids=[10, 11, 12], finish_reason="stop"),
        audio_output(finished=True, metrics=STAGE_METRICS),
    ]


async def results(outputs):
    for output in outputs:
        yield output


async def streamed_usage(serving_chat, modalities):
    lines = await collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=make_request(modalities=modalities, include_usage=True),
            result_generator=results(streamed_outputs()),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )
    (final,) = [chunk for chunk in parse_sse_chunks(lines) if chunk.get("usage") and not chunk.get("choices")]
    return final["usage"]


async def full_usage(serving_chat, modalities):
    serving_chat._create_audio_choice = MagicMock(return_value=[])
    response = await serving_chat.chat_completion_full_generator(
        request=make_request(modalities=modalities, stream=False),
        result_generator=results(full_outputs()),
        request_id="test-req",
        model_name="test-model",
        conversation=[],
        tokenizer=MagicMock(),
        request_metadata=MagicMock(),
    )
    return response.usage.model_dump(exclude_none=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [["audio"], ["text", "audio"]])
async def test_streamed_usage_counts_the_text_stage(modalities):
    assert await streamed_usage(build_serving_chat(), modalities) == TEXT_USAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [["audio"], ["text", "audio"]])
async def test_full_usage_counts_the_text_stage(modalities):
    assert await full_usage(build_serving_chat(), modalities) == TEXT_USAGE


@pytest.mark.asyncio
async def test_streamed_audio_only_request_metadata_counts_the_text_stage():
    metadata = MagicMock()
    await collect_stream(
        build_serving_chat().chat_completion_stream_generator(
            request=make_request(modalities=["audio"]),
            result_generator=results(streamed_outputs()),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=metadata,
        )
    )
    assert metadata.final_usage_info.model_dump(exclude_none=True) == TEXT_USAGE


class StageTokenUsage(OmniOpenAIServingChat):
    """Reports the second stage's generated tokens, read off the metrics ``_build_usage`` receives."""

    def _build_usage(self, request, **counts) -> UsageInfo:
        usage = super()._build_usage(request, **counts)
        stages = (counts["metrics"] or {}).get("stage_metrics", {})
        if "1" in stages:
            usage.stage_one_tokens = stages["1"]["num_tokens_out"]
        return usage


@pytest.mark.asyncio
@pytest.mark.parametrize("modalities", [["audio"], ["text", "audio"]])
async def test_a_subclass_builds_streamed_and_full_usage_in_one_place(modalities):
    serving_chat = build_serving_chat(chat_cls=StageTokenUsage)
    expected = {**TEXT_USAGE, "stage_one_tokens": 221}
    assert await streamed_usage(serving_chat, modalities) == expected
    assert await full_usage(serving_chat, modalities) == expected
