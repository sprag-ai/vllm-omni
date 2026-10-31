# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError, PrefixCacheConfig
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner, OmniAsyncOutput
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


def _runner(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    runner = object.__new__(OmniARModelRunner)
    runner.is_pooling_model = False
    runner.omni_prefix_cache = OmniPrefixCacheManager(
        PrefixCacheConfig(num_blocks=16, block_size=BLOCK_SIZE), eager=True
    )
    runner._prefix_cache_adapter = PrefixCacheSchedulerAdapter()
    runner._prefix_cache_group_view = SchedulerBlockView(BLOCK_SIZE)
    return runner


def _runner_step(runner, scheduler_output, rows):
    runner._begin_prefix_cache_step(scheduler_output)
    batch = SimpleNamespace(
        req_ids=list(scheduler_output.num_scheduled_tokens),
        num_tokens=rows.shape[0],
        num_tokens_after_padding=rows.shape[0],
    )
    return runner._save_prefix_cache_step(rows, {}, batch)


def test_runner_holds_only_steps_with_a_prefix_hit(monkeypatch):
    runner = _runner(monkeypatch)
    prompt_rows = _rows(range(10, 18))

    assert _runner_step(runner, _sched(new=[_new_req("a", [0, 1])], scheduled={"a": 8}), prompt_rows) is None
    hit = _runner_step(
        runner,
        _sched(new=[_new_req("b", [0, 1, 5], num_computed=8)], finished=["a"], scheduled={"b": 1}),
        _rows([90]),
    )

    assert hit is not None and hit[1] == ["b"]
    merged = runner.omni_prefix_cache.materialize(hit[0], ["b"]).hidden_states["b"]
    assert torch.equal(merged, torch.cat([prompt_rows, _rows([90])]))


def test_runner_skips_vllm_warmup_steps(monkeypatch):
    runner = _runner(monkeypatch)

    saved = _runner_step(runner, _sched(new=[_new_req("_warmup_0_", [0])], scheduled={"_warmup_0_": 2}), _rows([1, 2]))

    assert saved is None
    assert runner._prefix_cache_step is None


def _kv_cache_runner(monkeypatch, engine_output_type):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    monkeypatch.setattr(OmniGPUModelRunner, "initialize_kv_cache", lambda *args, **kwargs: None)
    runner = object.__new__(OmniARModelRunner)
    group = SimpleNamespace(kv_cache_spec=object.__new__(FullAttentionSpec))
    runner.kv_cache_config = SimpleNamespace(num_blocks=16, kv_cache_groups=[group])
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True, block_size=BLOCK_SIZE)
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.scheduler_config = SimpleNamespace(max_num_batched_tokens=64, max_model_len=128)
    runner.model_config = SimpleNamespace(engine_output_type=engine_output_type, max_model_len=128)
    runner.is_pooling_model = False
    runner.speculative_config = None
    return runner


def test_text_output_stage_runs_no_prefix_cache(monkeypatch):
    runner = _kv_cache_runner(monkeypatch, "text")
    runner.speculative_config = object()

    runner.initialize_kv_cache(runner.kv_cache_config)

    assert runner._omni_prefix_cache_cfg is None


def test_payload_stage_stages_a_prefix_cache_config(monkeypatch):
    runner = _kv_cache_runner(monkeypatch, "latent")

    runner.initialize_kv_cache(runner.kv_cache_config)

    assert runner._omni_prefix_cache_cfg.block_size == BLOCK_SIZE


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
