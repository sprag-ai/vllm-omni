# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import pytest

from vllm_omni.core.sched import omni_scheduler_mixin
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Base:
    def make_stats(self):
        return {"running": self.running}


class _Scheduler(OmniSchedulerMixin, _Base):
    def __init__(self) -> None:
        self.running = 0

    def has_unfinished_requests(self) -> bool:
        return self.running > 0


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(omni_scheduler_mixin.time, "monotonic", lambda: now[0])
    return now


def test_busy_steps_within_the_interval_are_throttled(clock) -> None:
    sched = _Scheduler()
    sched.running = 2
    assert sched.make_stats() == {"running": 2}
    clock[0] += 0.2
    assert sched.make_stats() is None
    clock[0] += 1.0
    assert sched.make_stats() == {"running": 2}


def test_the_step_that_drains_the_stage_always_reports(clock) -> None:
    sched = _Scheduler()
    sched.running = 1
    assert sched.make_stats() == {"running": 1}
    clock[0] += 0.2
    sched.running = 0
    assert sched.make_stats() == {"running": 0}


def test_idle_steps_after_the_drain_report_are_throttled(clock) -> None:
    sched = _Scheduler()
    sched.running = 1
    sched.make_stats()
    clock[0] += 0.2
    sched.running = 0
    sched.make_stats()
    clock[0] += 0.2
    assert sched.make_stats() is None
