# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streamed Qwen3-ASR transcription usage reports prompt tokens by modality over every audio chunk."""

import io
import json
import math

import numpy as np
import pytest
import requests
import soundfile as sf
from vllm.assets.audio import AudioAsset

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.slow]

MODEL = "Qwen/Qwen3-ASR-1.7B"
CLIP_S = 600
"""Longest clip ``Qwen3ASRLongFormGeneration`` transcribes in one generation; longer audio is chunked."""

SERVER = OmniServerParams(model=MODEL, server_args=["--enable-prompt-tokens-details", "--max-model-len", "32768"])


def speech(copies: int) -> tuple[bytes, float]:
    """Return ``copies`` back-to-back repeats of a spoken clip as 16-bit WAV, and its duration in seconds."""
    audio, rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    tiled = np.tile(np.asarray(audio, dtype=np.float32), copies)
    buffer = io.BytesIO()
    sf.write(buffer, tiled, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), len(tiled) / rate


def transcribe(server, audio: bytes, *, stream: bool) -> dict:
    """Return the usage of one transcription: the stream's final usage chunk, or the response's usage."""
    url = f"http://{server.host}:{server.port}/v1/audio/transcriptions"
    fields = {"model": server.model}
    if stream:
        fields |= {"stream": "true", "stream_include_usage": "true"}
    response = requests.post(url, files={"file": ("speech.wav", audio, "audio/wav")}, data=fields, stream=stream)
    response.raise_for_status()
    if not stream:
        return response.json()["usage"]
    usages = [
        json.loads(line.removeprefix("data: "))["usage"]
        for line in response.iter_lines(decode_unicode=True)
        if line.startswith("data: {") and '"usage"' in line
    ]
    assert len(usages) == 1
    return usages[0]


def audio_tokens(usage: dict) -> int:
    """Return the prompt's audio tokens from ``prompt_tokens_details.multimodal_tokens``."""
    return usage["prompt_tokens_details"]["multimodal_tokens"]["audio"]


def text_tokens(usage: dict) -> int:
    """Return the prompt tokens that are not audio."""
    return usage["prompt_tokens"] - audio_tokens(usage)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(SERVER, id="qwen3-asr")], indirect=True)
def test_unchunked_audio_tokens_scale_with_duration(omni_server):
    one_audio, one_s = speech(1)
    two_audio, two_s = speech(2)
    assert two_s < CLIP_S
    one = transcribe(omni_server, one_audio, stream=True)
    two = transcribe(omni_server, two_audio, stream=True)

    assert 0 < audio_tokens(one) < one["prompt_tokens"]
    assert text_tokens(two) == text_tokens(one)
    assert audio_tokens(two) / two_s == pytest.approx(audio_tokens(one) / one_s, rel=0.02)
    assert one["total_tokens"] == one["prompt_tokens"] + one["completion_tokens"]
    assert transcribe(omni_server, one_audio, stream=False)["seconds"] == math.ceil(one_s)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(SERVER, id="qwen3-asr")], indirect=True)
def test_chunked_prompt_and_audio_tokens_cover_every_chunk(omni_server):
    one_audio, one_s = speech(1)
    long_audio, long_s = speech(math.ceil((CLIP_S + 60) / one_s))
    chunks = math.ceil(long_s / CLIP_S)
    assert chunks == 2
    one = transcribe(omni_server, one_audio, stream=True)
    long = transcribe(omni_server, long_audio, stream=True)

    assert text_tokens(long) == chunks * text_tokens(one)
    assert audio_tokens(long) / long_s == pytest.approx(audio_tokens(one) / one_s, rel=0.02)
    assert long["total_tokens"] == long["prompt_tokens"] + long["completion_tokens"]
    assert transcribe(omni_server, long_audio, stream=False)["seconds"] == math.ceil(long_s)
