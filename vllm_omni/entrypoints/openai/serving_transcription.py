import math

from vllm.entrypoints.serve.engine.typing import SpeechToTextRequest
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.inputs import EngineInput

_BASE_TOKENS = 128
"""Allowance independent of length, covering short clips whose rate is dominated by the framing."""

_TOKENS_PER_SECOND = 8
"""Output tokens allowed per second of audio.

Measured over 136k leaderboard utterances: the median transcript runs 3.3 tokens/s and the 99.99th
percentile 8.7, with the excess above this rate confined to clips short enough for ``_BASE_TOKENS`` to
cover. No utterance in that corpus exceeds the resulting bound.
"""


class OmniOpenAIServingTranscription(OpenAIServingTranscription):
    """Transcription serving that bounds generation by the duration of the audio it transcribes.

    A transcript cannot outrun the speech it came from, but the upstream ceiling is the remaining
    context window, so a repetition loop on quasi-periodic audio generates until the context is full --
    65k tokens from a 7 second clip, observed. Bounding per request keeps a degenerate decode from
    holding a replica for minutes, and because the bound is derived from the audio it cannot truncate a
    transcript that the audio could legitimately produce.
    """

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[EngineInput], float, list[float]]:
        """Preprocess as upstream does, then narrow ``max_completion_tokens`` to what the audio allows.

        Overriding here rather than at the entrypoint reuses the duration the base class already measured
        off the decoded waveform, so every container it can ingest is covered and nothing is read twice.
        The caller reads ``max_completion_tokens`` back off this request when it builds the sampling
        params, which is why the narrowing is applied in place.
        """
        engine_inputs, duration, chunk_start_offsets = await super()._preprocess_speech_to_text(
            request=request, audio_data=audio_data, request_id=request_id
        )
        # ``allow_audio_chunking`` -- the upstream split of a long clip into one generation per window,
        # unrelated to vllm-omni's inter-stage ``async_chunk`` -- would charge this bound against each
        # window rather than the whole clip: looser than necessary, never truncating. It is off for this
        # model, whose config leaves ``min_energy_split_window_size`` None.
        bound = _BASE_TOKENS + math.ceil(_TOKENS_PER_SECOND * duration)
        asked = request.max_completion_tokens
        request.max_completion_tokens = bound if asked is None else min(asked, bound)
        return engine_inputs, duration, chunk_start_offsets
