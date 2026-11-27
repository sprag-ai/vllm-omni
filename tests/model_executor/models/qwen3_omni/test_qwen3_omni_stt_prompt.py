# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the Qwen3-Omni speech-to-text generation prompt."""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.inputs.speech_to_text import SPEECH_TO_TEXT_INSTRUCTIONS
from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni_moe_thinker as thinker_module
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"


class _RecordingProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        return "rendered"


@pytest.fixture
def processor(monkeypatch: pytest.MonkeyPatch) -> _RecordingProcessor:
    recorder = _RecordingProcessor()
    monkeypatch.setattr(thinker_module, "cached_processor_from_config", lambda *args, **kwargs: recorder)
    return recorder


def _render(prompt: str | None, instructions: str | None = None, task_type: str = "transcribe") -> None:
    params = SimpleNamespace(
        audio=np.zeros(16000, dtype=np.float32),
        stt_config=SimpleNamespace(sample_rate=16000),
        model_config=None,
        language=None,
        task_type=task_type,
        to_language=None,
        request_prompt=prompt,
    )
    token = SPEECH_TO_TEXT_INSTRUCTIONS.set(instructions)
    try:
        Qwen3OmniMoeForConditionalGeneration.get_generation_prompt(params)
    finally:
        SPEECH_TO_TEXT_INSTRUCTIONS.reset(token)


def test_instructions_are_the_system_message_and_the_prompt_precedes_the_audio(processor) -> None:
    _render("Return only the transcript.", instructions="You are strict.")

    assert processor.messages == [
        {"role": "system", "content": "You are strict."},
        {"role": "user", "content": f"Return only the transcript.{AUDIO_PLACEHOLDER}"},
    ]


def test_instructions_without_a_prompt_keep_the_default_instruction_before_the_audio(processor) -> None:
    _render(None, instructions="You are strict.")

    assert processor.messages == [
        {"role": "system", "content": "You are strict."},
        {"role": "user", "content": f"Transcribe this audio.{AUDIO_PLACEHOLDER}"},
    ]


def test_a_prompt_without_instructions_follows_the_audio_in_the_user_turn(processor) -> None:
    _render("Names: Ian.")

    assert processor.messages == [{"role": "user", "content": f"{AUDIO_PLACEHOLDER}Transcribe this audio. Names: Ian."}]


def test_neither_field_leaves_the_default_instruction_after_the_audio(processor) -> None:
    _render(None)

    assert processor.messages == [{"role": "user", "content": f"{AUDIO_PLACEHOLDER}Transcribe this audio."}]
