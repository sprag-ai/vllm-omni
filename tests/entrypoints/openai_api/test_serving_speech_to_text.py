# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import math
from types import SimpleNamespace

import pytest
from vllm.config import SpeechToTextConfig
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponseStreamChoice,
    TranscriptionStreamResponse,
)
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


DURATION_S = 750.0
PIECES = ("Hi", " there")


class PassThrough:
    """Stands in for a model's streaming post-processor: returns each delta unchanged."""

    def process_delta(self, text, finished):
        return text


def engine_input(prompt_tokens, *audio_lengths):
    placeholders = [PlaceholderRange(offset=3, length=length) for length in audio_lengths]
    return {"prompt_token_ids": [0] * prompt_tokens, "mm_placeholders": {"audio": placeholders}}


async def engine_outputs(prompt_tokens, fail=False):
    """Yield one chunk's engine outputs as the engine streams them: the prompt on the first, the finish on the last."""
    for index, piece in enumerate(PIECES):
        await asyncio.sleep(0)
        if fail and index:
            raise RuntimeError("engine died")
        finished = index == len(PIECES) - 1
        output = SimpleNamespace(
            text=piece, token_ids=[7], finish_reason="stop" if finished else None, stop_reason=None
        )
        yield SimpleNamespace(prompt_token_ids=[0] * prompt_tokens if index == 0 else None, outputs=[output])


class FakeUpstream:
    """Upstream's speech-to-text serving with fixed engine inputs, streaming through upstream's own generator."""

    task_type = "transcribe"
    enable_force_include_usage = False
    asr_config = SpeechToTextConfig(max_audio_clip_s=600)
    model_config = None
    streaming_post_processor_cls = PassThrough

    def __init__(self, engine_inputs, audio_tokens_per_second=None, fail=False):
        self.engine_inputs = engine_inputs
        self.fail = fail
        rate = audio_tokens_per_second
        self.model_cls = SimpleNamespace(
            get_num_audio_tokens=lambda duration, config, model: None if rate is None else math.ceil(duration * rate)
        )

    def create_streaming_error_response(self, exc):
        return json.dumps({"error": {"message": str(exc)}})

    async def _preprocess_speech_to_text(self, request, audio_data, request_id):
        await asyncio.sleep(0)
        return self.engine_inputs, DURATION_S, [0.0] * len(self.engine_inputs)

    async def _create_speech_to_text(self, request):
        engine_inputs, duration, _ = await self._preprocess_speech_to_text(
            request=request, audio_data=b"", request_id="req"
        )
        if not request.stream:
            return {"text": "Hi there"}
        generators = [engine_outputs(len(item["prompt_token_ids"]), self.fail) for item in engine_inputs]
        return SpeechToTextBaseServing._speech_to_text_stream_generator(
            self,
            request=request,
            list_result_generator=generators,
            request_id="req",
            request_metadata=RequestResponseMetadata(request_id="req"),
            audio_duration_s=duration,
            chunk_object_type="transcription.chunk",
            response_stream_choice_class=TranscriptionResponseStreamChoice,
            stream_response_class=TranscriptionStreamResponse,
            separator=" ",
        )


class PromptUsageServing(_StreamedPromptUsage, FakeUpstream):
    pass


def stt_request(stream=True, include_usage=True, continuous=False, model="rhythm"):
    return SimpleNamespace(
        model=model,
        stream=stream,
        stream_include_usage=include_usage,
        stream_continuous_usage_stats=continuous,
    )


async def collect(serving, request):
    result = await serving._create_speech_to_text(request)
    if not request.stream:
        return result
    return [event async for event in result]


async def serve(engine_inputs, request=None, enabled=True, **upstream):
    serving = PromptUsageServing(engine_inputs, enable_prompt_tokens_details=enabled, **upstream)
    return await collect(serving, request or stt_request())


async def upstream_stream(engine_inputs, request=None, **upstream):
    return await collect(FakeUpstream(engine_inputs, **upstream), request or stt_request())


def chunk_of(event):
    return json.loads(event.removeprefix("data: "))


def final_usage(events):
    finals = [
        chunk_of(event)["usage"]
        for event in events
        if event.startswith("data: {") and chunk_of(event).get("choices") == []
    ]
    assert len(finals) == 1
    return finals[0]


async def test_single_chunk_reports_its_audio_tokens():
    usage = final_usage(await serve([engine_input(88, 78)]))
    assert usage == {
        "prompt_tokens": 88,
        "total_tokens": 90,
        "completion_tokens": 2,
        "prompt_tokens_details": {"multimodal_tokens": {"audio": 78}},
    }


async def test_chunked_request_counts_every_chunk_in_prompt_and_audio_tokens():
    inputs = [engine_input(90, 40, 26), engine_input(85, 75), engine_input(40, 30)]
    usage = final_usage(await serve(inputs))
    assert final_usage(await upstream_stream(inputs))["prompt_tokens"] == 40
    assert usage == {
        "prompt_tokens": 215,
        "total_tokens": 221,
        "completion_tokens": 6,
        "prompt_tokens_details": {"multimodal_tokens": {"audio": 171}},
    }


async def test_chunked_request_counts_every_chunk_when_details_are_disabled():
    usage = final_usage(await serve([engine_input(90, 66), engine_input(40, 30)], enabled=False))
    assert usage == {"prompt_tokens": 130, "total_tokens": 134, "completion_tokens": 4}


async def test_single_chunk_stream_matches_upstream_when_details_are_disabled():
    inputs = [engine_input(88, 78)]
    assert await serve(inputs, enabled=False) == await upstream_stream(inputs)


async def test_stream_without_usage_matches_upstream():
    inputs = [engine_input(90, 66), engine_input(40, 30)]
    request = stt_request(include_usage=False)
    assert await serve(inputs, request) == await upstream_stream(inputs, request)


async def test_continuous_usage_deltas_match_upstream_and_only_the_final_chunk_is_corrected():
    inputs = [engine_input(90, 66), engine_input(40, 30)]
    request = stt_request(continuous=True)
    ours, theirs = await serve(inputs, request), await upstream_stream(inputs, request)
    assert ours[:-2] == theirs[:-2]
    assert [chunk_of(event)["usage"]["prompt_tokens"] for event in ours[:-2]] == [90, 90, 40, 40]
    assert final_usage(ours)["prompt_tokens"] == 130
    assert ours[-1] == theirs[-1] == "data: [DONE]\n\n"


async def test_model_counting_audio_tokens_from_duration_keeps_its_estimate_once():
    inputs = [engine_input(90), engine_input(40)]
    usage = final_usage(await serve(inputs, audio_tokens_per_second=13))
    estimate = math.ceil(DURATION_S * 13)
    assert final_usage(await upstream_stream(inputs, audio_tokens_per_second=13))["prompt_tokens"] == 40 + estimate
    assert usage["prompt_tokens"] == 90 + 40 + estimate
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_engine_failure_mid_stream_passes_upstreams_error_event_through():
    inputs = [engine_input(88, 78)]
    ours = await serve(inputs, fail=True)
    assert ours == await upstream_stream(inputs, fail=True)
    assert "engine died" in ours[-2]


async def test_final_chunk_keeps_non_ascii_text_unescaped():
    request = stt_request(model="rhythm-ü")
    final = (await serve([engine_input(88, 78)], request))[-2]
    assert '"model":"rhythm-ü"' in final


async def test_non_streamed_result_is_unchanged():
    assert await serve([engine_input(88, 78), engine_input(40, 30)], stt_request(stream=False)) == {"text": "Hi there"}


async def test_concurrent_requests_keep_their_own_counts():
    shapes = [[(88, 78)], [(90, 66), (40, 30)], [(90, 40, 26), (85, 75), (40, 30)], [(60, 50)]] * 5

    async def one(shape):
        usage = final_usage(await serve([engine_input(*chunk) for chunk in shape]))
        expected_audio = sum(sum(chunk[1:]) for chunk in shape)
        return usage["prompt_tokens"] == sum(chunk[0] for chunk in shape) and (
            usage["prompt_tokens_details"]["multimodal_tokens"]["audio"] == expected_audio
        )

    assert all(await asyncio.gather(*(one(shape) for shape in shapes)))


def test_transcription_serving_streams_through_the_prompt_usage():
    assert OmniOpenAIServingTranscription._create_speech_to_text is _StreamedPromptUsage._create_speech_to_text
