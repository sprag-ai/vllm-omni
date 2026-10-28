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
import torch
from vllm.assets.audio import AudioAsset
from vllm.config import SpeechToTextConfig
from vllm.model_executor.models.qwen3_asr import _get_feat_extract_output_lengths
from vllm.multimodal.audio import split_audio
from vllm.multimodal.media.audio import load_audio

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.slow]

MODEL = "Qwen/Qwen3-ASR-1.7B"
ASR_CONFIG = SpeechToTextConfig(max_audio_clip_s=600, sample_rate=16_000)
"""The clip length and sample rate ``Qwen3ASRLongFormGeneration`` transcribes at; longer audio is chunked."""

HOP_LENGTH = 160
"""Audio samples per mel frame in Qwen3-ASR's feature extractor."""

AUDIO_TOKENS_PER_SECOND = 13
"""Qwen3-ASR audio tokens per second of audio: 13 per 100 mel frames."""

SERVER = OmniServerParams(
    model=MODEL,
    server_args=["--enable-prompt-tokens-details", "--max-model-len", "32768"],
    env_dict={"VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600"},
)


def speech(copies: int) -> tuple[bytes, float]:
    """Return ``copies`` back-to-back repeats of a spoken clip as 16-bit WAV, and its duration in seconds."""
    audio, rate = AudioAsset("mary_had_lamb").audio_and_sample_rate
    tiled = np.tile(np.asarray(audio, dtype=np.float32), copies)
    buffer = io.BytesIO()
    sf.write(buffer, tiled, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), len(tiled) / rate


def expected_audio_tokens(audio: bytes) -> list[int]:
    """Return the audio tokens of each chunk the server splits ``audio`` into, decoded and split as upstream does."""
    with io.BytesIO(audio) as buffer:
        samples, rate = load_audio(buffer, sr=ASR_CONFIG.sample_rate, mono=True)
    chunks = [samples]
    if len(samples) / rate > ASR_CONFIG.max_audio_clip_s:
        chunks = split_audio(
            audio_data=samples,
            sample_rate=int(rate),
            max_clip_duration_s=ASR_CONFIG.max_audio_clip_s,
            overlap_duration_s=ASR_CONFIG.overlap_chunk_second,
            min_energy_window_size=ASR_CONFIG.min_energy_split_window_size,
        )
    frames = torch.tensor([len(chunk) // HOP_LENGTH for chunk in chunks])
    return _get_feat_extract_output_lengths(frames).tolist()


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


def assert_usage(server, audio: bytes, chunk_text_tokens: int | None = None) -> dict:
    """Stream ``audio`` and check its usage against the expected per-chunk counts; return the usage."""
    expected = expected_audio_tokens(audio)
    usage = transcribe(server, audio, stream=True)
    assert audio_tokens(usage) == sum(expected)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    if chunk_text_tokens is not None:
        assert text_tokens(usage) == len(expected) * chunk_text_tokens
    seconds = transcribe(server, audio, stream=False)["seconds"]
    assert math.ceil(audio_tokens(usage) / AUDIO_TOKENS_PER_SECOND) == seconds
    return usage


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(SERVER, id="qwen3-asr")], indirect=True)
def test_unchunked_audio_tokens_match_each_clip(omni_server):
    one_audio, _ = speech(1)
    two_audio, two_s = speech(2)
    assert two_s < ASR_CONFIG.max_audio_clip_s
    one = assert_usage(omni_server, one_audio)
    assert 0 < text_tokens(one)
    assert_usage(omni_server, two_audio, chunk_text_tokens=text_tokens(one))


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", [pytest.param(SERVER, id="qwen3-asr")], indirect=True)
def test_chunked_prompt_and_audio_tokens_cover_every_chunk(omni_server):
    one_audio, one_s = speech(1)
    long_audio, _ = speech(math.floor(ASR_CONFIG.max_audio_clip_s / one_s) + 1)
    expected = expected_audio_tokens(long_audio)
    assert len(expected) == 2
    assert expected[-1] < 0.02 * sum(expected)
    one = assert_usage(omni_server, one_audio)
    assert_usage(omni_server, long_audio, chunk_text_tokens=text_tokens(one))
