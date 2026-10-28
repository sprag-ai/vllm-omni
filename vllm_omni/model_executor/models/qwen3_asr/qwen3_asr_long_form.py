# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-ASR transcribing batch audio in ten-minute clips."""

from vllm.config import ModelConfig, SpeechToTextConfig
from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration
from vllm.transformers_utils.processor import cached_processor_from_config


class Qwen3ASRLongFormGeneration(Qwen3ASRForConditionalGeneration):
    """Qwen3-ASR that splits batch transcription into clips it can transcribe in one pass."""

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig, task_type: str) -> SpeechToTextConfig:
        """Split batch audio into ten-minute clips, each cut at the quietest point before its boundary.

        The model transcribes a single clip of up to twenty minutes; past that its output stops tracking
        the audio.
        """
        processor = cached_processor_from_config(model_config)
        return SpeechToTextConfig(max_audio_clip_s=600, sample_rate=processor.feature_extractor.sampling_rate)
