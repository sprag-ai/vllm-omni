# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A step drafts only while the running requests accept enough of the drafts."""

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.adaptive_speculation import AcceptanceDraftSchedule, AdaptiveSpeculationConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _schedule(running, **config) -> AcceptanceDraftSchedule:
    return AcceptanceDraftSchedule(AdaptiveSpeculationConfig(**config), 3, lambda: running)


def _verify(schedule, req_id, tokens, steps=8):
    for _ in range(steps):
        schedule.record(req_id, tokens)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ({}, None),
        ({"adaptive_speculation": False}, None),
        ({"adaptive_speculation": True}, AdaptiveSpeculationConfig()),
        ({"adaptive_speculation": {"min_tokens_per_step": 2.0}}, AdaptiveSpeculationConfig(min_tokens_per_step=2.0)),
    ],
)
def test_config_reads_the_stage_setting(raw, expected):
    assert AdaptiveSpeculationConfig.from_additional_config(raw) == expected


@pytest.mark.parametrize(
    "value",
    [{"min_tokens": 2}, {"min_tokens_per_step": 1.0}, {"smoothing": 0}, {"smoothing": 1.5}, "yes"],
)
def test_config_refuses_unknown_keys_and_out_of_range_values(value):
    with pytest.raises(ValueError, match="adaptive_speculation"):
        AdaptiveSpeculationConfig.from_additional_config({"adaptive_speculation": value})


def test_new_requests_draft_until_measured():
    schedule = _schedule(["a"])
    assert schedule[1] == 3
    assert not schedule.drafter_idle


def test_one_poor_step_does_not_retire_a_request():
    schedule = _schedule(["asr"])
    schedule.record("asr", 1)
    assert schedule[1] == 3


def test_a_request_the_drafter_misses_stops_drafting_and_idles_the_drafter():
    schedule = _schedule(["prose"])
    _verify(schedule, "prose", 1)
    assert schedule[1] == 0
    assert schedule.drafter_idle


def test_requests_the_drafter_idled_past_stop_being_measured():
    running = ["prose"]
    schedule = _schedule(running)
    _verify(schedule, "prose", 1)
    assert schedule[1] == 0 and schedule.drafter_idle
    _verify(schedule, "prose", 4)
    assert schedule[1] == 0 and schedule.drafter_idle


def test_a_newcomer_among_idled_requests_forfeits_drafting():
    running = [f"prose{i}" for i in range(8)]
    schedule = _schedule(running)
    for req_id in running:
        _verify(schedule, req_id, 1)
    assert schedule[8] == 0 and schedule.drafter_idle
    running.append("new")
    assert schedule[9] == 0 and schedule.drafter_idle
    running[:] = ["new"]
    assert schedule[1] == 0 and schedule.drafter_idle


def test_a_fresh_batch_drafts_beside_requests_that_accept():
    running = ["asr"]
    schedule = _schedule(running)
    _verify(schedule, "asr", 4)
    running.append("new")
    assert schedule[2] == 3


def test_the_batch_mean_decides_a_mixed_step():
    schedule = _schedule(["prose", "asr"], min_tokens_per_step=1.7)
    _verify(schedule, "prose", 1)
    _verify(schedule, "asr", 4)
    assert schedule[2] == 3

    schedule = _schedule(["prose", "prose2", "asr"], min_tokens_per_step=2.5)
    for req_id in ("prose", "prose2"):
        _verify(schedule, req_id, 1)
    _verify(schedule, "asr", 4)
    assert schedule[3] == 0
    # The drafter keeps the accepting request's cache current.
    assert not schedule.drafter_idle


def test_finished_requests_are_forgotten():
    schedule = _schedule(["r0", "r1"])
    for i in range(100):
        schedule.record(f"old{i}", 1)
    schedule[1]
    assert schedule._tokens_per_step == {}


def test_scheduler_records_tokens_committed_by_verified_steps():
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

    scheduler = object.__new__(OmniARScheduler)
    recorded = []
    scheduler._adaptive_drafts = SimpleNamespace(record=lambda req_id, tokens: recorded.append((req_id, tokens)))
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1, -1, -1], "b": [], "c": [-1]})
    runner_output = SimpleNamespace(sampled_token_ids=[[1, 2], [3], []], req_id_to_index={"a": 0, "b": 1, "c": 2})

    scheduler._record_verified_drafts(scheduler_output, runner_output)

    assert recorded == [("a", 2)]
