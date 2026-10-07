# SPDX-License-Identifier: Apache-2.0
"""Compatibility imports and RPC helpers for the legacy serial engine."""

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_decision_base import (
    Qwen3OmniDecisionThinker,
    apply_qv_adapter,
)

__all__ = ["Qwen3OmniDecisionThinker", "apply_qv_adapter", "arm_worker", "take_worker"]


def arm_worker(worker, threshold, request_id, mode):
    worker.model_runner.model.arm_decision(threshold, request_id, mode)


def take_worker(worker):
    return worker.model_runner.model.take_decision()
