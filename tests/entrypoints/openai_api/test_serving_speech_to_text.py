# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
from vllm.config import SpeechToTextConfig

from vllm_omni.entrypoints.openai.serving_speech_to_text import (
    OmniOpenAIServingTranscription,
    _AudioDurationBound,
    _StreamedDurationUsage,
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
USAGE = (
    'data: {"id":"t","object":"transcription.chunk","choices":[],'
    '"usage":{"prompt_tokens":125,"total_tokens":145,"completion_tokens":20}}\n\n'
)
DONE = "data: [DONE]\n\n"


class FakeStream:
    """Stands in for upstream's stream generator: yields fixed events."""

    def __init__(self, events):
        self.events = events

    async def _speech_to_text_stream_generator(self, *, audio_duration_s, **kwargs):
        for event in self.events:
            yield event


class DurationStream(_StreamedDurationUsage, FakeStream):
    pass


async def stream(events, duration):
    serving = DurationStream(events)
    return [event async for event in serving._speech_to_text_stream_generator(audio_duration_s=duration, request=None)]


async def test_final_usage_chunk_reports_duration_rounded_up():
    _, usage, _ = await stream([DELTA, USAGE, DONE], 5.12)
    assert json.loads(usage.removeprefix("data: "))["usage"] == {
        "prompt_tokens": 125,
        "total_tokens": 145,
        "completion_tokens": 20,
        "type": "duration",
        "seconds": 6,
    }


async def test_whole_second_duration_is_not_rounded_up():
    _, usage, _ = await stream([DELTA, USAGE, DONE], 6.0)
    assert json.loads(usage.removeprefix("data: "))["usage"]["seconds"] == 6


async def test_other_events_pass_through_unchanged():
    error = 'data: {"error":{"message":"boom"}}\n\n'
    assert await stream([DELTA, error, DONE], 5.12) == [DELTA, error, DONE]


async def test_stream_without_usage_chunk_is_unchanged():
    assert await stream([DELTA, DONE], 5.12) == [DELTA, DONE]


def test_transcription_serving_streams_through_the_duration_usage():
    assert (
        OmniOpenAIServingTranscription._speech_to_text_stream_generator
        is _StreamedDurationUsage._speech_to_text_stream_generator
    )
