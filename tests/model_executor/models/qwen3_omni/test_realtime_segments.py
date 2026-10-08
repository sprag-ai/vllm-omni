# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from vllm_omni.model_executor.models.qwen3_omni.realtime_segments import (
    SilenceAlignedBuffer,
    continuation_prompt,
    segment_separator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RATE = 16_000
WINDOW = 1_600


def buffer(segment_s: float = 5.0, search_s: float = 1.0) -> SilenceAlignedBuffer:
    return SilenceAlignedBuffer(
        sampling_rate=RATE, segment_duration_s=segment_s, search_duration_s=search_s, energy_window_samples=WINDOW
    )


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, int(seconds * RATE)).astype(np.float32)


@pytest.mark.parametrize(
    ("previous", "following", "expected"),
    [
        ("when I started listening.", "To him.", " "),
        ("listening", "to him", " "),
        ("listening. ", "To him.", ""),
        ("listening.", " To him.", ""),
        ("listening", ", to him", ""),
        ("", "To him.", ""),
        ("listening.", "", ""),
        ("我开始听。", "他说", ""),
        ("聞き始めた", "とき", ""),
        ("hello", "世界", ""),
        ("듣기 시작했다", "그에게", " "),
        ("เริ่มฟัง", "เขา", ""),
    ],
)
def test_segment_separator(previous, following, expected):
    assert segment_separator(previous, following) == expected


def test_continuation_closes_the_previous_turn_without_a_system_message():
    prompt = continuation_prompt("<|audio_start|><|audio_pad|><|audio_end|>")
    assert prompt == (
        "<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
    )
    assert "system" not in prompt


def test_holds_audio_until_a_full_segment_is_buffered():
    buf = buffer()
    buf.write_audio(speech(4.9))
    assert buf.read_audio() is None


def test_cuts_at_the_quietest_window_in_the_search_span():
    audio = speech(6.0)
    pause_start = int(4.3 * RATE)
    audio[pause_start : pause_start + WINDOW] = 0.0
    buf = buffer()
    buf.write_audio(audio)

    segment = buf.read_audio()

    assert segment is not None
    assert len(segment) == pause_start + WINDOW // 2
    assert buf.read_audio() is None
    remaining = buf.flush()
    assert remaining is not None
    np.testing.assert_array_equal(np.concatenate([segment, remaining]), audio)


def test_never_cuts_before_the_search_span():
    audio = speech(6.0)
    audio[int(2.0 * RATE) : int(2.5 * RATE)] = 0.0
    buf = buffer()
    buf.write_audio(audio)

    segment = buf.read_audio()

    assert segment is not None
    assert int(4.0 * RATE) <= len(segment) <= int(5.0 * RATE)


def test_releases_consecutive_segments_from_one_long_write():
    audio = speech(12.0)
    buf = buffer()
    buf.write_audio(audio)

    segments = []
    while (segment := buf.read_audio()) is not None:
        segments.append(segment)
    remaining = buf.flush()

    assert len(segments) == 2
    assert all(int(4.0 * RATE) <= len(s) <= int(5.0 * RATE) for s in segments)
    np.testing.assert_array_equal(np.concatenate([*segments, remaining]), audio)


def test_accumulates_small_writes():
    audio = speech(5.2)
    buf = buffer()
    for start in range(0, len(audio), int(0.2 * RATE)):
        buf.write_audio(audio[start : start + int(0.2 * RATE)])

    assert buf.read_audio() is not None


def test_zero_search_span_cuts_at_the_segment_limit():
    buf = buffer(search_s=0.0)
    buf.write_audio(speech(6.0))

    segment = buf.read_audio()

    assert segment is not None
    assert len(segment) == 5 * RATE


def test_flush_on_an_empty_buffer_returns_none():
    assert buffer().flush() is None


@pytest.mark.parametrize("search_s", [5.0, 6.0, -1.0])
def test_rejects_a_search_span_that_does_not_fit_the_segment(search_s):
    with pytest.raises(ValueError):
        buffer(search_s=search_s)
