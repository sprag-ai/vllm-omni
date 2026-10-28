# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
from vllm.config import SpeechToTextConfig
from vllm.multimodal.inputs import PlaceholderRange

from vllm_omni.entrypoints.openai.serving_speech_to_text import (
    OmniOpenAIServingTranscription,
    _AudioDurationBound,
    _StreamedPromptUsage,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNKED = SpeechToTextConfig(max_audio_clip_s=90, min_energy_split_window_size=1600)
UNCHUNKED = SpeechToTextConfig(max_audio_clip_s=90, min_energy_split_window_size=None)


class FakeServing:
    """Stands in for upstream's speech-to-text serving: reports a fixed duration."""

    def __init__(self, asr_config: SpeechToTextConfig, duration: float):
        self.asr_config = asr_config
        self.duration = duration

    async def _preprocess_speech_to_text(self, request, audio_data, request_id):
        return ["engine-input"], self.duration, [0.0]


class BoundedServing(_AudioDurationBound, FakeServing):
    pass


async def bound_for(asr_config, duration, asked=None):
    request = SimpleNamespace(max_completion_tokens=asked)
    await BoundedServing(asr_config, duration)._preprocess_speech_to_text(request, b"", "req-1")
    return request.max_completion_tokens


async def test_unchunked_bound_scales_with_whole_clip():
    assert await bound_for(UNCHUNKED, 3600) == 128 + 8 * 3600


async def test_chunked_bound_scales_with_one_window():
    # Each window is a separate generation sharing these sampling params, so the
    # budget must cover one window, not the whole clip.
    assert await bound_for(CHUNKED, 3600) == 128 + 8 * 90


async def test_chunked_clip_shorter_than_window_uses_its_duration():
    assert await bound_for(CHUNKED, 7) == 128 + 8 * 7


async def test_caller_limit_below_bound_is_kept():
    assert await bound_for(CHUNKED, 3600, asked=50) == 50


async def test_caller_limit_above_bound_is_narrowed():
    assert await bound_for(CHUNKED, 3600, asked=100_000) == 128 + 8 * 90


DELTA = 'data: {"id":"t","object":"transcription.chunk","choices":[{"delta":{"content":"Hi"}}]}\n\n'
DONE = "data: [DONE]\n\n"


def usage_chunk(prompt_tokens, completion_tokens=20):
    usage = {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens + completion_tokens}
    usage["completion_tokens"] = completion_tokens
    return f"data: {json.dumps({'id': 't', 'object': 'transcription.chunk', 'choices': [], 'usage': usage})}\n\n"


def engine_input(prompt_tokens, *audio_lengths):
    placeholders = [PlaceholderRange(offset=3, length=length) for length in audio_lengths]
    return {"prompt_token_ids": [0] * prompt_tokens, "mm_placeholders": {"audio": placeholders}}


class FakeStreamServing:
    """Stands in for upstream's speech-to-text serving: preprocesses fixed engine inputs, then streams or not."""

    def __init__(self, engine_inputs, events=None):
        self.engine_inputs = engine_inputs
        self.events = events

    async def _preprocess_speech_to_text(self, request, audio_data, request_id):
        return self.engine_inputs, 5.12, [0.0]

    async def _create_speech_to_text(self, *args, **kwargs):
        await self._preprocess_speech_to_text(request=None, audio_data=b"", request_id="req-1")
        if self.events is None:
            return {"text": "Hi"}

        async def stream():
            for event in self.events:
                yield event

        return stream()


class PromptUsageServing(_StreamedPromptUsage, FakeStreamServing):
    pass


async def serve(engine_inputs, events=None, enabled=True):
    serving = PromptUsageServing(engine_inputs, events, enable_prompt_tokens_details=enabled)
    result = await serving._create_speech_to_text()
    if events is None:
        return result
    return [event async for event in result]


def usage_of(event):
    return json.loads(event.removeprefix("data: "))["usage"]


async def test_final_usage_chunk_carries_audio_prompt_tokens():
    _, usage, _ = await serve([engine_input(88, 78)], [DELTA, usage_chunk(88), DONE])
    assert usage_of(usage) == {
        "prompt_tokens": 88,
        "total_tokens": 108,
        "completion_tokens": 20,
        "prompt_tokens_details": {"multimodal_tokens": {"audio": 78}},
    }


async def test_chunked_request_counts_every_chunk_in_prompt_and_audio_tokens():
    inputs = [engine_input(90, 40, 26), engine_input(85, 75), engine_input(40, 30)]
    _, usage, _ = await serve(inputs, [DELTA, usage_chunk(40), DONE])
    assert usage_of(usage) == {
        "prompt_tokens": 215,
        "total_tokens": 235,
        "completion_tokens": 20,
        "prompt_tokens_details": {"multimodal_tokens": {"audio": 171}},
    }


async def test_chunked_request_counts_every_chunk_when_details_are_disabled():
    inputs = [engine_input(90, 66), engine_input(40, 30)]
    _, usage, _ = await serve(inputs, [DELTA, usage_chunk(40), DONE], enabled=False)
    assert usage_of(usage) == {"prompt_tokens": 130, "total_tokens": 150, "completion_tokens": 20}


async def test_other_events_pass_through_unchanged():
    error = 'data: {"error":{"message":"boom"}}\n\n'
    assert await serve([engine_input(88, 78)], [DELTA, error, DONE]) == [DELTA, error, DONE]


async def test_single_chunk_stream_is_unchanged_when_details_are_disabled():
    events = [DELTA, usage_chunk(88), DONE]
    assert await serve([engine_input(88, 78)], events, enabled=False) == events


async def test_non_streamed_result_is_unchanged():
    assert await serve([engine_input(88, 78), engine_input(40, 30)]) == {"text": "Hi"}


def test_transcription_serving_streams_through_the_prompt_usage():
    assert OmniOpenAIServingTranscription._create_speech_to_text is _StreamedPromptUsage._create_speech_to_text
