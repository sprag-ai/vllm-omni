# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-step speculative draft count chosen from the running requests' acceptance.

Enabled per stage with ``additional_config.adaptive_speculation``, ``true`` or a
mapping whose optional keys are ``min_tokens_per_step`` and ``smoothing``. A
step drafts ``num_speculative_tokens`` when the running requests' recent tokens
per verification step reach ``min_tokens_per_step`` and drafts nothing
otherwise, so traffic the drafter predicts poorly decodes without paying for
verification it would lose.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

ADAPTIVE_SPECULATION_KEY = "adaptive_speculation"


@dataclass(frozen=True)
class AdaptiveSpeculationConfig:
    """Thresholds for choosing a step's draft count.

    Attributes:
        min_tokens_per_step: Mean committed tokens per verification step at which
            drafting pays; above 1.
        smoothing: Weight of the newest verification step in a request's moving
            average, in (0, 1].
    """

    min_tokens_per_step: float = 1.7
    smoothing: float = 0.3

    @classmethod
    def from_additional_config(cls, additional_config: Mapping[str, Any] | None) -> "AdaptiveSpeculationConfig | None":
        """Read the stage's setting; None when adaptive speculation is off.

        Raises:
            ValueError: The setting is not a mapping or ``true``, names an unknown key,
                or holds a value outside its range.
        """
        raw = (additional_config or {}).get(ADAPTIVE_SPECULATION_KEY)
        if raw is None or raw is False:
            return None
        if raw is True:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{ADAPTIVE_SPECULATION_KEY} must be true or a mapping, got {raw!r}")
        unknown = set(raw) - {"min_tokens_per_step", "smoothing"}
        if unknown:
            raise ValueError(f"unknown {ADAPTIVE_SPECULATION_KEY} keys: {sorted(unknown)}")
        config = cls(**raw)
        if config.min_tokens_per_step <= 1 or not 0 < config.smoothing <= 1:
            raise ValueError(f"{ADAPTIVE_SPECULATION_KEY} out of range: {config}")
        return config


class AcceptanceDraftSchedule:
    """Draft count for the next step, indexed like vLLM's dynamic speculative-decoding lookup.

    ``schedule[batch_size]`` ignores the batch size and decides from the requests
    ``running_ids`` returns. A request counts as accepting every draft until it has
    ``CONFIRMING_STEPS`` verifications, so new traffic drafts until measured; then
    it counts at the mean of those, followed by a moving average.

    When a step drafts nothing and no running request is confirmed at or above the
    threshold over at least ``CONFIRMING_STEPS`` verifications, ``drafter_idle`` is
    set and the drafter may skip the step. Its own cache then misses those
    requests' positions, so their averages freeze, an unmeasured one at a single
    token per step, and they never again keep the drafter from idling.
    """

    CONFIRMING_STEPS = 2

    def __init__(
        self,
        config: AdaptiveSpeculationConfig,
        num_speculative_tokens: int,
        running_ids: Callable[[], Iterable[str]],
    ) -> None:
        """Bind the schedule to a stage.

        Args:
            config: The stage's thresholds.
            num_speculative_tokens: Draft count of a drafting step.
            running_ids: Ids of the requests the next step decodes.
        """
        self._config = config
        self._num_speculative_tokens = num_speculative_tokens
        self._running_ids = running_ids
        self._tokens_per_step: dict[str, float] = {}
        self._verifications: dict[str, int] = {}
        self._stale: set[str] = set()
        self.drafter_idle = False

    def record(self, req_id: str, tokens: int) -> None:
        """Fold one verification step into a request's moving average.

        Args:
            req_id: The verified request.
            tokens: Tokens the step committed, the accepted drafts plus one.
        """
        if req_id in self._stale:
            return
        count = self._verifications.get(req_id, 0)
        previous = self._tokens_per_step.get(req_id, 0.0)
        weight = 1 / (count + 1) if count < self.CONFIRMING_STEPS else self._config.smoothing
        self._tokens_per_step[req_id] = (1 - weight) * previous + weight * tokens
        self._verifications[req_id] = count + 1

    def _accepts(self, req_id: str) -> bool:
        return (
            self._verifications.get(req_id, 0) >= self.CONFIRMING_STEPS
            and self._tokens_per_step[req_id] >= self._config.min_tokens_per_step
        )

    def __getitem__(self, batch_size: int) -> int:
        running = list(self._running_ids())
        self.drafter_idle = False
        if len(self._tokens_per_step) + len(self._stale) > 4 * max(len(running), 16):
            live = set(running)
            self._tokens_per_step = {r: v for r, v in self._tokens_per_step.items() if r in live}
            self._verifications = {r: v for r, v in self._verifications.items() if r in live}
            self._stale &= live
        if not running:
            return self._num_speculative_tokens
        optimistic = float(self._num_speculative_tokens + 1)
        averages = [
            self._tokens_per_step[r]
            if r in self._stale or self._verifications.get(r, 0) >= self.CONFIRMING_STEPS
            else optimistic
            for r in running
        ]
        if sum(averages) / len(averages) >= self._config.min_tokens_per_step:
            return self._num_speculative_tokens
        if not any(self._accepts(r) for r in running if r not in self._stale):
            self.drafter_idle = True
            for r in running:
                self._tokens_per_step.setdefault(r, 1.0)
            self._stale.update(running)
        return 0
