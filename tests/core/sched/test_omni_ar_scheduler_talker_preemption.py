# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A preempted talker request is failed with an explicit error instead of resumed."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# isort: off
import vllm_omni  # noqa: F401 - import for side effects (patch vLLM)
from vllm.v1.engine import FinishReason
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_ar_scheduler as ar_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

# isort: on

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(model_stage: str) -> OmniARScheduler:
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model_stage=model_stage))
    return scheduler


def _request(request_id: str, client_index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(request_id=request_id, client_index=client_index)


@pytest.fixture
def upstream_preempt(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    preempt = MagicMock()
    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "_preempt_request", preempt)
    return preempt


def test_talker_preemption_is_recorded(upstream_preempt: MagicMock) -> None:
    scheduler = _make_scheduler("talker")
    request = _request("req-talker", client_index=2)

    OmniARScheduler._preempt_request(scheduler, request, 1.0)

    upstream_preempt.assert_called_once_with(request, 1.0, drop_stale_output=False)
    assert scheduler._unresumable_preemptions["req-talker"][0] == 2


def test_thinker_preemption_is_left_to_upstream(upstream_preempt: MagicMock) -> None:
    scheduler = _make_scheduler("thinker")

    OmniARScheduler._preempt_request(scheduler, _request("req-thinker"), 1.0)

    upstream_preempt.assert_called_once()
    assert not getattr(scheduler, "_unresumable_preemptions", None)


def test_recorded_preemption_is_finished_and_reported() -> None:
    scheduler = _make_scheduler("talker")
    scheduler._unresumable_preemptions = {"req-talker": (1, "preempted")}
    request = _request("req-talker", client_index=1)
    scheduler.finish_requests = MagicMock(return_value=[request])

    scheduler._finish_unresumable_preemptions()

    ids, status = scheduler.finish_requests.call_args.args
    assert set(ids) == {"req-talker"}
    assert status is RequestStatus.FINISHED_ERROR
    assert scheduler._unresumable_preemptions == {}

    outputs: dict[int, list] = {}
    scheduler._emit_failed_preemption_outputs(outputs)

    (output,) = outputs[1]
    assert output.request_id == "req-talker"
    assert output.finish_reason is FinishReason.ERROR
    assert output.stop_reason == "preempted"
    assert scheduler._failed_preemptions == {}


def test_preemption_already_finished_is_not_reported() -> None:
    scheduler = _make_scheduler("talker")
    scheduler._unresumable_preemptions = {"req-aborted": (0, "preempted")}
    scheduler.finish_requests = MagicMock(return_value=[])

    scheduler._finish_unresumable_preemptions()

    outputs: dict[int, list] = {}
    scheduler._emit_failed_preemption_outputs(outputs)
    assert outputs == {}


def test_schedule_finishes_preemptions_before_upstream_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler = _make_scheduler("talker")
    scheduler._unresumable_preemptions = {"req-talker": (0, "preempted")}
    scheduler.finish_requests = MagicMock(return_value=[_request("req-talker")])
    scheduler.waiting = []
    scheduler.max_num_running_reqs = 8
    scheduler._drop_aborted_queued_requests = lambda: None
    scheduler._process_pending_omni_inputs = lambda model_mode: None
    scheduler._resync_streaming_input_counter = lambda: None
    scheduler._should_defer_waiting_admission = lambda: False
    scheduler._async_chunk_transport_enabled = lambda: False
    scheduler._restore_omni_wait_queues = lambda: None
    scheduler._postprocess_omni_schedule_output = lambda *args, **kwargs: None
    scheduler.get_finished_requests_needing_kv_transfer = lambda: {}
    scheduler._wrap_omni_scheduler_output = lambda output, **kwargs: output

    def upstream_schedule(_self, _throttle_prefills: bool = False):
        scheduler.finish_requests.assert_called_once()
        return "scheduled"

    monkeypatch.setattr(ar_sched_mod.VLLMScheduler, "schedule", upstream_schedule)

    assert OmniARScheduler.schedule(scheduler) == "scheduled"
