import json
import math
from collections.abc import AsyncGenerator
from contextvars import ContextVar

from vllm.config import SpeechToTextConfig
from vllm.entrypoints.openai.chat_completion.serving import _get_mm_token_counts, _make_prompt_tokens_details
from vllm.entrypoints.serve.engine.protocol import PromptTokenUsageInfo
from vllm.entrypoints.serve.engine.typing import SpeechToTextRequest
from vllm.entrypoints.speech_to_text.transcription.serving import (
    OpenAIServingTranscription,
)
from vllm.entrypoints.speech_to_text.translation.serving import (
    OpenAIServingTranslation,
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


class _AudioDurationBound:
    """Bounds speech-to-text generation by the duration of the audio it transcribes.

    A transcript cannot outrun the speech it came from, but the upstream ceiling is the remaining
    context window, so a repetition loop on quasi-periodic audio generates until the context is full --
    65k tokens from a 7 second clip, observed. Bounding per request keeps a degenerate decode from
    holding a replica for minutes, and because the bound is derived from the audio it cannot truncate a
    transcript that the audio could legitimately produce.

    Mixed into both speech-to-text endpoints: they share ``_preprocess_speech_to_text`` through
    ``SpeechToTextBaseServing``, so a repetition loop reaches a replica through either one.
    """

    asr_config: SpeechToTextConfig

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
        # With ``allow_audio_chunking`` -- upstream's split of a long clip into one generation per window,
        # unrelated to vllm-omni's inter-stage ``async_chunk`` -- every window is its own generation sharing
        # these sampling params, so the bound covers one window. Upstream's chunks never exceed
        # ``max_audio_clip_s``; the last, shorter window gets the full budget, which is loose but never
        # truncates.
        bounded_s = duration
        if self.asr_config.allow_audio_chunking:
            bounded_s = min(duration, self.asr_config.max_audio_clip_s)
        bound = _BASE_TOKENS + math.ceil(_TOKENS_PER_SECOND * bounded_s)
        asked = request.max_completion_tokens
        request.max_completion_tokens = bound if asked is None else min(asked, bound)
        return engine_inputs, duration, chunk_start_offsets


_SSE_PREFIX = "data: "

_PROMPT_MM_TOKENS: ContextVar[dict[str, int] | None] = ContextVar("_prompt_mm_tokens", default=None)
"""The current request's prompt tokens by modality, summed over its audio chunks while it is preprocessed."""


def _with_prompt_tokens_details(event: str, details: PromptTokenUsageInfo) -> str:
    """Set ``prompt_tokens_details`` on the stream's final usage chunk; return any other event unchanged."""
    if not event.startswith(_SSE_PREFIX):
        return event
    try:
        chunk = json.loads(event[len(_SSE_PREFIX) :])
    except json.JSONDecodeError:
        return event
    if not isinstance(chunk, dict) or chunk.get("choices") != [] or not isinstance(chunk.get("usage"), dict):
        return event
    chunk["usage"]["prompt_tokens_details"] = details.model_dump(exclude_none=True)
    return f"{_SSE_PREFIX}{json.dumps(chunk, separators=(',', ':'))}\n\n"


class _StreamedPromptTokenDetails:
    """Reports a streamed request's prompt tokens by modality in ``prompt_tokens_details``, as chat does.

    ``multimodal_tokens`` sums every audio chunk the request was split into.
    """

    def __init__(self, *args, enable_prompt_tokens_details: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.enable_prompt_tokens_details = enable_prompt_tokens_details

    async def _create_speech_to_text(self, *args, **kwargs):
        """Serve the request as upstream does, adding ``prompt_tokens_details`` to a stream's final usage chunk."""
        counts: dict[str, int] = {}
        token = _PROMPT_MM_TOKENS.set(counts)
        try:
            result = await super()._create_speech_to_text(*args, **kwargs)
        finally:
            _PROMPT_MM_TOKENS.reset(token)
        details = _make_prompt_tokens_details(self.enable_prompt_tokens_details, None, None, counts)
        if details is None or not isinstance(result, AsyncGenerator):
            return result
        return self._stream_with_details(result, details)

    async def _preprocess_speech_to_text(
        self,
        request: SpeechToTextRequest,
        audio_data: bytes,
        request_id: str,
    ) -> tuple[list[EngineInput], float, list[float]]:
        """Preprocess as upstream does, counting the rendered prompts' tokens by modality."""
        engine_inputs, duration, chunk_start_offsets = await super()._preprocess_speech_to_text(
            request=request, audio_data=audio_data, request_id=request_id
        )
        counts = _PROMPT_MM_TOKENS.get()
        if counts is not None:
            for engine_input in engine_inputs:
                for modality, tokens in _get_mm_token_counts(engine_input).items():
                    counts[modality] = counts.get(modality, 0) + tokens
        return engine_inputs, duration, chunk_start_offsets

    @staticmethod
    async def _stream_with_details(
        stream: AsyncGenerator[str, None], details: PromptTokenUsageInfo
    ) -> AsyncGenerator[str, None]:
        """Re-yield the stream with ``details`` on its final usage chunk."""
        async for event in stream:
            yield _with_prompt_tokens_details(event, details)


class OmniOpenAIServingTranscription(_StreamedPromptTokenDetails, _AudioDurationBound, OpenAIServingTranscription):
    """``/v1/audio/transcriptions``, bounded by the duration of its audio, its stream's prompt tokens by modality."""


class OmniOpenAIServingTranslation(_AudioDurationBound, OpenAIServingTranslation):
    """``/v1/audio/translations``, bounded by the duration of its audio."""
