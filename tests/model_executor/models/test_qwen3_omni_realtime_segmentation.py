# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def segmentation(monkeypatch, segment, search):
    for name, value in (("SPRAG_REALTIME_SEGMENT_S", segment), ("SPRAG_REALTIME_CUT_SEARCH_S", search)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return qwen3_omni._realtime_segmentation()


def test_defaults_are_5s_segments_searched_over_the_last_second(monkeypatch):
    assert segmentation(monkeypatch, None, None) == (5.0, 1.0)


def test_overrides_set_segment_and_search(monkeypatch):
    assert segmentation(monkeypatch, "3.5", "0.5") == (3.5, 0.5)


def test_a_segment_override_keeps_the_default_search(monkeypatch):
    assert segmentation(monkeypatch, "8", None) == (8.0, 1.0)


def test_a_zero_search_cuts_at_the_segment_limit(monkeypatch):
    assert segmentation(monkeypatch, "4", "0") == (4.0, 0.0)


@pytest.mark.parametrize(
    ("segment", "search"),
    [
        ("abc", None),
        (None, "x"),
        ("2", "2"),
        ("2", "3"),
        ("0", None),
        (None, "-1"),
        ("inf", None),
        ("1e309", None),
        ("nan", None),
    ],
)
def test_an_invalid_pair_falls_back_to_both_defaults(monkeypatch, segment, search):
    assert segmentation(monkeypatch, segment, search) == (5.0, 1.0)
