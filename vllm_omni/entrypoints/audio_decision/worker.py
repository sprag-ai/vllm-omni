# SPDX-License-Identifier: Apache-2.0
"""Bind decisions after vLLM has compacted and reordered its input batch."""

from vllm.v1.worker.gpu_worker import Worker


def decision_batch(runner, scheduler_output):
    result = []
    for request_id in runner.input_batch.req_ids:
        state = runner.requests[request_id]
        params = state.sampling_params
        data = params.extra_args["audio_decision"]
        tokens = scheduler_output.num_scheduled_tokens[request_id]
        if state.num_computed_tokens != 0 or tokens != state.num_prompt_tokens or params.max_tokens != 1:
            raise RuntimeError("Decision serving requires one complete prefill per request")
        if len(scheduler_output.scheduled_encoder_inputs.get(request_id, ())) != 1:
            raise RuntimeError("Decision prefill must schedule its own audio encoder input")
        result.append({**data, "input_tokens": tokens})
    if len({r["request_id"] for r in result}) != len(result):
        raise RuntimeError("Duplicate decision request IDs in scheduler batch")
    return result


class DecisionWorker(Worker):
    def execute_model(self, scheduler_output):
        if self.use_v2_model_runner:
            raise RuntimeError("Decision batches currently require VLLM_USE_V2_MODEL_RUNNER=0")
        model = self.get_model()
        model.decision_encoder_calls = 0
        model.batch_audio_items = 0
        model.batch_provider = lambda: decision_batch(self.model_runner, scheduler_output)
        try:
            return super().execute_model(scheduler_output)
        finally:
            model.clear_batch()

    def take_decision_result(self, request_id):
        return self.get_model().take_result(request_id)
