# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.worker_v2.omni_ar_model_runner import OmniAsyncOutput
from vllm_omni.worker_v2.prefix_cache import SchedulerBlockView

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BLOCK_SIZE = 4
HIDDEN = 3


def _new_req(req_id, block_ids, num_computed=0):
    return SimpleNamespace(req_id=req_id, block_ids=(list(block_ids),), num_computed_tokens=num_computed)


def _sched(*, new=(), cached=(), finished=(), preempted=(), scheduled=None):
    cached_reqs = SimpleNamespace(
        req_ids=[req_id for req_id, _, _ in cached],
        num_computed_tokens=[computed for _, computed, _ in cached],
        new_block_ids=[None if blocks is None else (list(blocks),) for _, _, blocks in cached],
        resumed_req_ids=set(),
    )
    return SimpleNamespace(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached_reqs,
        finished_req_ids=set(finished),
        preempted_req_ids=set(preempted),
        num_scheduled_tokens=dict(scheduled or {}),
    )


def _rows(values):
    return torch.tensor(values, dtype=torch.float32).unsqueeze(1).expand(len(values), HIDDEN).clone()


def _step(manager, adapter, view, scheduler_output, rows):
    """Run one engine step through the view, adapter and manager; returns the step id."""
    step = adapter.translate_step(scheduler_output)
    manager.new_step_starts(step)
    view.observe(scheduler_output)
    view.bind_batch(list(scheduler_output.num_scheduled_tokens))
    layout = adapter.build_write_layout(view, num_scheduled_tokens=dict(step.scheduled_tokens))
    return manager.save_outputs(
        rows,
        {},
        num_tokens_unpadded=rows.shape[0],
        num_tokens_padded=rows.shape[0],
        write_layout=layout,
    )


def test_prefix_hit_merges_the_rows_an_earlier_request_computed():
    manager = OmniPrefixCacheManager(PrefixCacheConfig(num_blocks=16, block_size=BLOCK_SIZE), eager=True)
    adapter = PrefixCacheSchedulerAdapter()
    view = SchedulerBlockView(BLOCK_SIZE)

    prompt_rows = _rows(range(10, 18))
    first = _step(manager, adapter, view, _sched(new=[_new_req("a", [0, 1])], scheduled={"a": 8}), prompt_rows)
    manager.materialize(first, ["a"])
    decode = _step(manager, adapter, view, _sched(cached=[("a", 8, [2])], scheduled={"a": 1}), _rows([18]))
    manager.materialize(decode, ["a"])

    tail_rows = _rows([90, 91])
    hit = _step(
        manager,
        adapter,
        view,
        _sched(new=[_new_req("b", [0, 1, 5], num_computed=8)], finished=["a"], scheduled={"b": 2}),
        tail_rows,
    )
    merged = manager.materialize(hit, ["b"]).hidden_states["b"]

    assert torch.equal(merged, torch.cat([prompt_rows, tail_rows]))


def test_view_maps_positions_through_appended_blocks():
    view = SchedulerBlockView(BLOCK_SIZE)
    view.observe(_sched(new=[_new_req("a", [7])], scheduled={"a": 4}))
    view.observe(_sched(cached=[("a", 4, [2])], scheduled={"a": 2}))

    assert view.step_slots_cpu(["a"], {"a": 2}).tolist() == [8, 9]


def test_view_forgets_finished_and_preempted_requests():
    view = SchedulerBlockView(BLOCK_SIZE)
    view.observe(_sched(new=[_new_req("a", [0]), _new_req("b", [1])], scheduled={"a": 1, "b": 1}))
    view.observe(_sched(finished=["a"], preempted=["b"]))

    for req_id in ("a", "b"):
        with pytest.raises(OmniPrefixCacheUnmatchError):
            view.step_slots_cpu([req_id], {req_id: 1})


def test_view_refuses_a_position_past_the_mirrored_blocks():
    view = SchedulerBlockView(BLOCK_SIZE)
    view.observe(_sched(new=[_new_req("a", [0])], scheduled={"a": 5}))

    with pytest.raises(OmniPrefixCacheUnmatchError):
        view.step_slots_cpu(["a"], {"a": 5})


class _RecordingManager:
    def __init__(self, merged):
        self.merged = merged
        self.calls = []

    def materialize(self, step_id, req_ids):
        self.calls.append(("materialize", step_id, list(req_ids)))
        return self.merged

    def discard_step(self, step_id):
        self.calls.append(("discard", step_id))


def _async_output(manager, req_ids, hit_req_ids):
    output = object.__new__(OmniAsyncOutput)
    output.model_runner_output = SimpleNamespace(req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)})
    output._num_reqs = len(req_ids)
    output._prefix_cache_hits = (manager, 7, hit_req_ids)
    return output


def test_hit_payload_keeps_the_step_keys_the_cache_does_not_hold():
    prefix = torch.ones(5, HIDDEN)
    merged = SimpleNamespace(hidden_states={"b": prefix}, mm_outputs={"hidden.24": {"b": prefix}})
    manager = _RecordingManager(merged)
    output = _async_output(manager, ["a", "b"], ["b"])
    step_payloads = [{"hidden.24": torch.zeros(1, HIDDEN)}, {"hidden.24": torch.zeros(2, HIDDEN), "embed.sampled": 1}]

    payloads = output._overlay_prefix_hit_payloads(
        step_payloads, output._take_prefix_cache_payloads(merge=True, include_hidden=False)
    )

    assert payloads[0] is step_payloads[0]
    assert payloads[1]["embed.sampled"] == 1
    assert torch.equal(payloads[1]["hidden.24"], prefix)
    assert "hidden" not in payloads[1]
    assert manager.calls == [("materialize", 7, ["b"])]


def test_step_context_is_discarded_when_no_payload_reads_it():
    manager = _RecordingManager(None)
    output = _async_output(manager, ["a"], ["a"])

    output._take_prefix_cache_payloads(merge=False, include_hidden=False)
    output._take_prefix_cache_payloads(merge=True, include_hidden=False)

    assert manager.calls == [("discard", 7)]
