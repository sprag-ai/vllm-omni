# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.config import SpeechToTextConfig

from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

UPSTREAM = SpeechToTextConfig(max_audio_clip_s=30, min_energy_split_window_size=None)


@pytest.fixture(autouse=True)
def upstream_thinker_config(monkeypatch):
    monkeypatch.setattr(
        qwen3_omni.VllmQwen3OmniMoeThinker,
        "get_speech_to_text_config",
        classmethod(lambda cls, model_config, task_type: UPSTREAM),
    )


def stt_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SPRAG_AUDIO_CHUNK_S", raising=False)
    else:
        monkeypatch.setenv("SPRAG_AUDIO_CHUNK_S", value)
    return Qwen3OmniMoeForConditionalGeneration.get_speech_to_text_config(None, "transcribe")


def test_default_window_is_90s(monkeypatch):
    config = stt_config(monkeypatch, None)
    assert config.allow_audio_chunking
    assert config.max_audio_clip_s == 90


def test_positive_override_sets_window(monkeypatch):
    assert stt_config(monkeypatch, "60").max_audio_clip_s == 60


@pytest.mark.parametrize("value", ["-1", "0"])
def test_non_positive_override_disables_chunking(monkeypatch, value):
    # A zero window never terminates in upstream's split_audio, so zero must
    # disable chunking like a negative value rather than reach the splitter.
    assert stt_config(monkeypatch, value) == UPSTREAM


def test_unparseable_override_keeps_default(monkeypatch):
    assert stt_config(monkeypatch, "ninety").max_audio_clip_s == 90
