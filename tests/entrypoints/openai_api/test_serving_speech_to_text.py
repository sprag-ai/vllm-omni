# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm.config import SpeechToTextConfig

from vllm_omni.entrypoints.openai.serving_speech_to_text import _AudioDurationBound

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
