# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind scoring anchors after the scheduler has reordered its batch."""

from vllm.v1.worker.gpu_worker import Worker


def spev_batch(runner, scheduler_output):
    result = []
    for request_id in runner.input_batch.req_ids:
        state = runner.requests[request_id]
        params = state.sampling_params
        data = params.extra_args["spev"]
        tokens = scheduler_output.num_scheduled_tokens[request_id]
        if state.num_computed_tokens != 0 or tokens != state.num_prompt_tokens or params.max_tokens != 1:
            raise RuntimeError("SPEV requires one complete prefill per request")
        positions = data["positions"]
        if not positions or positions != sorted(set(positions)) or any(p < 0 or p >= tokens for p in positions):
            raise RuntimeError("Invalid SPEV scoring anchors")
        result.append({**data, "input_tokens": tokens})
    if len({r["request_id"] for r in result}) != len(result):
        raise RuntimeError("Duplicate SPEV request IDs")
    return result


class SPEVWorker(Worker):
    def execute_model(self, scheduler_output):
        if self.use_v2_model_runner:
            raise RuntimeError("SPEV requires VLLM_USE_V2_MODEL_RUNNER=0")
        model = self.get_model()
        model.batch_provider = lambda: spev_batch(self.model_runner, scheduler_output)
        try:
            return super().execute_model(scheduler_output)
        finally:
            model.batch_provider = None

    def take_spev_result(self, request_id):
        return self.get_model().completed_scores.pop(request_id, None)
