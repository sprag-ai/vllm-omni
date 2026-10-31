# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV-cache group view of the omni prefix cache for the MRv2 runner.

MRv2 keeps its block tables on the device, so the slots of a step are
derived from the group-0 block ids and computed-token counts that the
scheduler sends, mirrored here on the host.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError


class SchedulerBlockView:
    """Group-0 block ids and computed-token counts mirrored from scheduler output."""

    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self._block_ids: dict[str, list[int]] = {}
        self._num_computed: dict[str, int] = {}
        self._batch_req_ids: list[str] = []

    def observe(self, scheduler_output: Any) -> None:
        """Apply one step's request changes; call once per real step, before the forward."""
        removed = set(scheduler_output.finished_req_ids)
        removed.update(getattr(scheduler_output, "preempted_req_ids", None) or ())
        for req_id in removed:
            self._block_ids.pop(req_id, None)
            self._num_computed.pop(req_id, None)
        for new_req in scheduler_output.scheduled_new_reqs:
            groups = new_req.block_ids
            self._block_ids[new_req.req_id] = list(groups[0]) if groups else []
            self._num_computed[new_req.req_id] = int(new_req.num_computed_tokens)
        cached = scheduler_output.scheduled_cached_reqs
        for req_id, num_computed, new_block_ids in zip(
            cached.req_ids, cached.num_computed_tokens, cached.new_block_ids
        ):
            if new_block_ids is not None:
                self._block_ids.setdefault(req_id, []).extend(new_block_ids[0])
            self._num_computed[req_id] = int(num_computed)

    def bind_batch(self, req_ids: list[str]) -> None:
        """Set the row order of the step being saved."""
        self._batch_req_ids = list(req_ids)

    def batch_req_ids(self) -> list[str]:
        return list(self._batch_req_ids)

    def step_slots_cpu(self, req_ids: list[str], num_scheduled: dict[str, int]) -> torch.Tensor:
        """Return this step's KV slots in row order.

        Raises:
            OmniPrefixCacheUnmatchError: A scheduled position has no mirrored block.
        """
        bs = self.block_size
        parts: list[torch.Tensor] = []
        for req_id in req_ids:
            n = int(num_scheduled.get(req_id, 0))
            if n <= 0:
                continue
            blocks = self._block_ids.get(req_id)
            start = self._num_computed.get(req_id)
            if blocks is None or start is None:
                raise OmniPrefixCacheUnmatchError(f"no mirrored block table for req {req_id}")
            first, last = start // bs, (start + n - 1) // bs
            if last >= len(blocks):
                raise OmniPrefixCacheUnmatchError(
                    f"req {req_id} schedules position {start + n - 1} past its {len(blocks)} mirrored blocks"
                )
            pos = torch.arange(start, start + n, dtype=torch.long)
            table = torch.tensor(blocks[first : last + 1], dtype=torch.long)
            parts.append(table[pos // bs - first] * bs + pos % bs)
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)
