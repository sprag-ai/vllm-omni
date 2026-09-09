import io
import math

import soundfile as sf
from fastapi import Request
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionRequest
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_BASE_TOKENS = 128
"""Allowance independent of length, covering short clips whose rate is dominated by the framing."""

_TOKENS_PER_SECOND = 8
"""Output tokens allowed per second of audio.

Measured over 136k leaderboard utterances: the median transcript runs 3.3 tokens/s and the 99.99th
percentile 8.7, with the excess above this rate confined to clips short enough for ``_BASE_TOKENS`` to
cover. No utterance in that corpus exceeds the resulting bound.
"""


def audio_token_bound(audio_data: bytes) -> int | None:
    """Return the generation ceiling ``audio_data`` justifies, or None when its duration is unreadable.

    Reads the container header rather than decoding, so the cost does not scale with clip length.
    """
    try:
        info = sf.info(io.BytesIO(audio_data))
        seconds = info.frames / info.samplerate if info.samplerate else None
    except Exception:  # noqa: BLE001 — an unreadable header must not fail the request
        seconds = None
    if not seconds or seconds <= 0:
        return None
    return _BASE_TOKENS + math.ceil(_TOKENS_PER_SECOND * seconds)


class OmniOpenAIServingTranscription(OpenAIServingTranscription):
    """Transcription serving that bounds generation by the duration of the audio it transcribes.

    A transcript cannot outrun the speech it came from, but the upstream ceiling is the remaining
    context window, so a repetition loop on quasi-periodic audio generates until the context is full --
    65k tokens from a 7 second clip, observed. Bounding per request keeps a degenerate decode from
    holding a replica for minutes, and because the bound is derived from the audio it cannot truncate a
    transcript that the audio could legitimately produce.
    """

    async def create_transcription(
        self,
        audio_data: bytes,
        request: TranscriptionRequest,
        raw_request: Request | None = None,
    ):
        """Transcribe ``audio_data``, narrowing ``max_completion_tokens`` to what its duration allows."""
        bound = audio_token_bound(audio_data)
        if bound is not None:
            asked = request.max_completion_tokens
            narrowed = bound if asked is None else min(asked, bound)
            if narrowed != asked:
                request = request.model_copy(update={"max_completion_tokens": narrowed})
        else:
            logger.warning("transcription audio duration unreadable; generation left unbounded")
        return await super().create_transcription(
            audio_data=audio_data, request=request, raw_request=raw_request
        )
