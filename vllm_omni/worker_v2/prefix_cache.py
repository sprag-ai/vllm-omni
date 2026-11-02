# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 binding of the omni prefix cache.

``MRv2PrefixCacheMixin`` owns the cache lifecycle through the runner's
``load_model`` / ``initialize_kv_cache`` / ``shutdown`` chain. A runner only
calls ``_prefix_cache_step_begin`` before it retires finished requests, and an
AR runner ``_prefix_cache_save_mrv2`` after postprocess plus
``_prefix_cache_output_finalizer`` for its async output. A speculative AR
runner also calls ``_prefix_cache_record_rejections`` after sampling.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm_omni.core.prefix_cache import stage_prefix_cache_config
from vllm_omni.core.prefix_cache.adapter import PrefixCacheSchedulerAdapter
from vllm_omni.core.prefix_cache.group_view import MRv2FullAttentionGroupView, check_mrv2_prefix_cache_layout
from vllm_omni.core.prefix_cache.interface import (
    ModelCachePolicy,
    OmniPrefixCacheUnmatchError,
    PrefixCacheConfig,
)
from vllm_omni.core.prefix_cache.manager import OmniPrefixCacheManager
from vllm_omni.core.prefix_cache.runner_mixin import PrefixCacheRunnerMixin

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as _RunnerBase
else:
    _RunnerBase = object

OutputFinalizer = Callable[[Any], Any]


class PrefixCacheRewrite:
    """A saved step whose request payloads need merged rows.

    Owns the step id until ``apply`` or ``discard`` consumes it, exactly once.
    """

    def __init__(self, cache: OmniPrefixCacheManager, step_id: int, rows: tuple[tuple[int, str], ...]):
        self._cache: OmniPrefixCacheManager | None = cache
        self.step_id = step_id
        # (batch row, req id) of each rewritten request.
        self.rows = rows

    def _take(self) -> OmniPrefixCacheManager:
        cache, self._cache = self._cache, None
        if cache is None:
            raise RuntimeError(f"prefix-cache step {self.step_id} was already consumed")
        return cache

    def discard(self) -> None:
        self._take().discard_step(self.step_id)

    def apply(self, output: Any) -> None:
        """Replace cached keys in the rewritten requests' payloads in place.

        Payload lists are indexed by batch row and hold the request's
        flattened step slice, so their keys are the cache keys.
        """
        outs = self._take().materialize(self.step_id, [req_id for _, req_id in self.rows], include_uncached=False)
        payload_lists = {
            id(payloads): payloads
            for payloads in (output.inter_stage_outputs, output.pooler_output, output.multimodal_outputs)
            if payloads
        }
        for row, req_id in self.rows:
            payloads = {id(p[row]): p[row] for p in payload_lists.values() if row < len(p) and p[row]}
            for payload in payloads.values():
                if outs.hidden_states is not None and "hidden" in payload:
                    payload["hidden"] = outs.hidden_states[req_id]
                for key, per_req in outs.mm_outputs.items():
                    if key in payload:
                        payload[key] = per_req[req_id]


@dataclass
class _StepRejections:
    """Rejected draft counts of one sampled step, copied to the host after ``event``."""

    step: int
    req_ids: tuple[str, ...]
    rejected: torch.Tensor
    event: torch.cuda.Event
    reported: bool = False

    def by_request(self) -> dict[str, int]:
        self.event.synchronize()
        return {req_id: n for req_id, n in zip(self.req_ids, self.rejected.tolist()) if n > 0}


class MRv2PrefixCacheMixin(PrefixCacheRunnerMixin, _RunnerBase):
    """Stage-output prefix cache for Model Runner V2.

    MRv2 keeps its block table on the device only, so the adapter records
    each request's block ids from the scheduler output and the view derives
    step slots from them and the bound ``InputBatch``. Models that do not set
    ``supports_mrv2_prefix_cache_outputs`` reuse native KV alone.
    """

    _prefix_cache_coalesce_decode = True
    # True for a runner that calls _prefix_cache_record_rejections after sampling.
    _prefix_cache_corrects_draft_positions = False
    _prefix_cache_step_count = 0
    _prefix_cache_rejections: deque[_StepRejections] | None = None

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        self._snapshot_prefix_cache_model_policy(self.model)

    def initialize_kv_cache(self, kv_cache_config: Any, *args: Any, **kwargs: Any) -> None:
        super().initialize_kv_cache(kv_cache_config, *args, **kwargs)
        is_profiling = bool(kwargs.get("is_profiling", args[0] if args else False))
        self._init_omni_prefix_cache(is_profiling=is_profiling)

    def shutdown(self) -> None:
        # Runs after the subclass drained its output worker, whose
        # materializations consume step ids.
        try:
            self._shutdown_omni_prefix_cache()
        finally:
            super().shutdown()

    def _snapshot_prefix_cache_model_policy(self, model: Any) -> None:
        """MTP stages complete frames after sampling and model-owned output
        snapshots are not mergeable, so only opted-in models cache outputs."""
        if getattr(model, "supports_mrv2_prefix_cache_outputs", False):
            super()._snapshot_prefix_cache_model_policy(model)
        else:
            self._omni_cache_policy = ModelCachePolicy(needs_full_hidden_states=False, native_kv_cache_only=True)

    def _new_prefix_cache_adapter(self) -> PrefixCacheSchedulerAdapter:
        return PrefixCacheSchedulerAdapter(track_requests=True)

    def _build_prefix_cache_group_view(
        self, cfg: PrefixCacheConfig, adapter: PrefixCacheSchedulerAdapter
    ) -> MRv2FullAttentionGroupView:
        return MRv2FullAttentionGroupView(adapter, cfg.block_size)

    def _init_omni_prefix_cache(self, *, is_profiling: bool) -> None:
        # A re-initialized KV cache invalidates every cached slot.
        self._shutdown_omni_prefix_cache()
        if is_profiling:
            return
        cfg = stage_prefix_cache_config(
            kv_cache_config=self.kv_cache_config,
            cache_config=self.cache_config,
            kv_transfer_config=getattr(self.vllm_config, "kv_transfer_config", None),
            scheduler_config=self.scheduler_config,
            model_config=self.model_config,
            is_pooling_model=self.is_pooling_model,
            speculative_config=getattr(self.vllm_config, "speculative_config", None),
            model_cache_policy=self._omni_cache_policy,
            corrects_draft_positions=self._prefix_cache_corrects_draft_positions,
        )
        if cfg is None:
            return
        check_mrv2_prefix_cache_layout(
            block_sizes=list(self.block_tables.block_sizes),
            block_size=cfg.block_size,
            dcp_size=int(getattr(self, "dcp_size", 1)),
        )
        # Rewritten steps hold their step id until the async output is
        # materialized: up to max_concurrent_batches queued for the engine plus
        # as many in the native output worker, plus the step being saved.
        depth = 2 * max(1, int(getattr(self.vllm_config, "max_concurrent_batches", 1))) + 1
        if cfg.staging_depth < depth:
            cfg = replace(cfg, staging_depth=depth)
        self._omni_prefix_cache_cfg = cfg
        self._prefix_cache_rejections = (
            deque() if getattr(self.vllm_config, "speculative_config", None) is not None else None
        )
        # Built now, not on the first step: the adapter's block ledger must
        # observe every request from its first scheduled step.
        self._ensure_omni_prefix_cache()

    def _prefix_cache_scheduler_lag(self) -> int:
        """Steps whose rejected drafts the scheduler has not yet subtracted when it schedules the next one."""
        return 1 if getattr(self.scheduler_config, "async_scheduling", False) else 0

    def _prefix_cache_step_begin(self, scheduler_output: Any) -> None:
        self._prefix_cache_step_count += 1
        self._prefix_cache_report_rejected_rows()
        super()._prefix_cache_step_begin(scheduler_output)

    def _prefix_cache_report_rejected_rows(self) -> None:
        """Hand the adapter the rejected drafts of every step the scheduler has seen the output of."""
        records, adapter = self._prefix_cache_rejections, self._prefix_cache_adapter
        if not records or adapter is None:
            return
        settled = self._prefix_cache_step_count - 1 - self._prefix_cache_scheduler_lag()
        for record in records:
            if record.step <= settled and not record.reported:
                for req_id, count in record.by_request().items():
                    adapter.add_rejected_rows(req_id, count)
                record.reported = True
        while records and records[0].reported and records[0].step < self._prefix_cache_step_count - 1:
            records.popleft()

    def _prefix_cache_record_rejections(self, input_batch: Any, num_rejected: torch.Tensor) -> None:
        """Copy this step's per-request rejected draft counts to the host without waiting on them."""
        if self._prefix_cache_rejections is None or self.omni_prefix_cache is None:
            return
        n = input_batch.num_reqs
        rejected = torch.empty(n, dtype=num_rejected.dtype, pin_memory=True)
        rejected.copy_(num_rejected[:n], non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._prefix_cache_rejections.append(
            _StepRejections(self._prefix_cache_step_count, tuple(input_batch.req_ids[:n]), rejected, event)
        )

    def _prefix_cache_position_corrections(self) -> dict[str, int]:
        """Drafts of the previous step that this step's scheduled positions still count as accepted."""
        records = self._prefix_cache_rejections
        if not records or self._prefix_cache_scheduler_lag() == 0:
            return {}
        previous = records[-1]
        if previous.step != self._prefix_cache_step_count - 1:
            return {}
        return previous.by_request()

    def _prefix_cache_save_mrv2(
        self, input_batch: Any, hidden_states: torch.Tensor, multimodal_outputs: dict[str, Any] | None
    ) -> PrefixCacheRewrite | None:
        """Save this step; keep its id only when payloads must be rewritten.

        A step whose merged rows equal its slice for every request (no hit,
        replay or trim) is discarded at once, so misses keep the plain
        output path.
        """
        cache, view = self.omni_prefix_cache, self._prefix_cache_group_view
        if cache is None or view is None:
            return None
        view.bind(input_batch, self._prefix_cache_position_corrections())
        try:
            step_id = self._prefix_cache_save_step(
                hidden_states,
                multimodal_outputs,
                num_tokens_unpadded=input_batch.num_tokens,
                num_tokens_padded=input_batch.num_tokens_after_padding,
            )
        finally:
            view.bind(None)
        if step_id is None:
            return None
        rewrite_ids = cache.step_rewrite_req_ids(step_id)
        emits_payloads = getattr(self.vllm_config.model_config, "engine_output_type", "text") != "text"
        if not rewrite_ids or not emits_payloads or self._model_mm_outputs_written_in_sample():
            cache.discard_step(step_id)
            return None
        batch_req_ids = input_batch.req_ids[: input_batch.num_reqs]
        rows = tuple((row, req_id) for row, req_id in enumerate(batch_req_ids) if req_id in rewrite_ids)
        return PrefixCacheRewrite(cache, step_id, rows)

    @staticmethod
    def _prefix_cache_output_finalizer(
        rewrite: PrefixCacheRewrite | None, finalize: OutputFinalizer | None, *, mergeable: bool
    ) -> OutputFinalizer | None:
        """``finalize`` preceded by the step's payload rewrite.

        The rewrite runs once the async output has its host payloads, before
        they are published. A model that finalizes its own outputs or streams
        audio partitions them itself, so a rewrite there is refused.
        """
        if rewrite is None:
            return finalize
        if not mergeable:
            rewrite.discard()
            raise OmniPrefixCacheUnmatchError(
                "omni prefix cache hit/replay cannot be merged into model-owned outputs; "
                "disable enable_prefix_caching for this stage"
            )

        def _finalize(output: Any) -> Any:
            rewrite.apply(output)
            return output if finalize is None else finalize(output)

        return _finalize
