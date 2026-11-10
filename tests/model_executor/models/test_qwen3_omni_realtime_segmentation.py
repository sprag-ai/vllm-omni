# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def setenv(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


def transcription(monkeypatch, segment, search):
    setenv(monkeypatch, "SPRAG_REALTIME_TRANSCRIPTION_SEGMENT_S", segment)
    setenv(monkeypatch, "SPRAG_REALTIME_TRANSCRIPTION_CUT_SEARCH_S", search)
    return qwen3_omni._transcription_segmentation()


def conversation(monkeypatch, segment):
    setenv(monkeypatch, "SPRAG_REALTIME_CONVERSATION_SEGMENT_S", segment)
    return qwen3_omni._conversation_segment_seconds()


@pytest.mark.parametrize("segment", [None, ""])
def test_transcription_decodes_each_turn_once_by_default(monkeypatch, segment):
    assert transcription(monkeypatch, segment, None) is None


def test_transcription_overrides_set_segment_and_search(monkeypatch):
    assert transcription(monkeypatch, "3.5", "0.5") == (3.5, 0.5)


def test_a_transcription_segment_keeps_the_default_search(monkeypatch):
    assert transcription(monkeypatch, "8", None) == (8.0, 1.0)


def test_a_zero_search_cuts_at_the_segment_limit(monkeypatch):
    assert transcription(monkeypatch, "4", "0") == (4.0, 0.0)


@pytest.mark.parametrize(
    ("segment", "search"),
    [
        ("abc", None),
        ("5", "x"),
        ("2", "2"),
        ("2", "3"),
        ("0", None),
        ("5", "-1"),
        ("inf", None),
        ("1e309", None),
        ("nan", None),
    ],
)
def test_an_invalid_transcription_pair_decodes_each_turn_once(monkeypatch, segment, search):
    assert transcription(monkeypatch, segment, search) is None


def test_the_old_shared_setting_no_longer_segments_transcription(monkeypatch):
    monkeypatch.setenv("SPRAG_REALTIME_SEGMENT_S", "5")
    assert transcription(monkeypatch, None, None) is None


@pytest.mark.parametrize("segment", [None, ""])
def test_conversation_decodes_each_turn_once_by_default(monkeypatch, segment):
    assert conversation(monkeypatch, segment) is None


def test_conversation_override_sets_the_segment(monkeypatch):
    assert conversation(monkeypatch, "8") == 8.0


def test_conversation_ignores_the_transcription_setting(monkeypatch):
    monkeypatch.setenv("SPRAG_REALTIME_TRANSCRIPTION_SEGMENT_S", "5")
    assert conversation(monkeypatch, None) is None


@pytest.mark.parametrize("segment", ["abc", "0", "-1", "inf", "nan"])
def test_an_invalid_conversation_segment_decodes_each_turn_once(monkeypatch, segment):
    assert conversation(monkeypatch, segment) is None
