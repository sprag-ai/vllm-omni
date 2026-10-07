# SPDX-License-Identifier: Apache-2.0
"""Per-request readouts over a scheduler-owned batch of complete prefills."""

import torch
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.entrypoints.audio_decision.model import Qwen3OmniDecisionThinker
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    Qwen3OmniMoeThinkerProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniBatchedDecisionThinker(Qwen3OmniDecisionThinker):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.batch_provider = None
        self.batch_requests = None
        self.batch_scores = None
        self.batch_results = None
        self.completed_decisions = {}
        self.batch_audio_items = 0

    def _process_audio_input(self, audio_input):
        self.batch_audio_items += int(audio_input["audio_feature_lengths"].numel())
        return super()._process_audio_input(audio_input)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if self.batch_provider is None:
            return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs)
        if intermediate_tensors is not None:
            raise RuntimeError("Decision batches require PP=1")
        requests = self.batch_provider()
        if not requests:
            raise RuntimeError("Scheduled decision batch is empty")
        self.batch_requests = requests
        decoder = self.language_model.model
        hidden = inputs_embeds if inputs_embeds is not None else decoder.embed_input_ids(input_ids)
        lengths = [r["input_tokens"] for r in requests]
        if hidden.ndim != 2 or sum(lengths) != hidden.shape[0]:
            raise RuntimeError("Decision rows do not match complete scheduler prefills")
        if self.batch_audio_items != len(requests):
            raise RuntimeError("Expected one freshly encoded audio item per decision")
        ends = torch.tensor(lengths, device=hidden.device).cumsum(0) - 1
        residual = None
        scores = None
        embeddings = None
        fallback = None
        head_confidence = None
        executed = 0
        for index, layer in enumerate(decoder.layers):
            hidden, residual = layer(positions, hidden, residual)
            executed += 1
            if index + 1 == self.decision_spec["depth"]:
                logical = hidden[ends] if residual is None else hidden[ends] + residual[ends]
                h = self.decision_head
                scores = ((logical.double() - h["mean"]) / h["scale"]) @ h["weight"] + h["bias"]
                scores = scores / self.decision_spec["head_temperature"]
                head_confidence = torch.softmax(scores, dim=-1).amax(dim=-1).cpu().tolist()
                fallback = [
                    r["mode"] in ("full", "raw") or (r["mode"] == "auto" and confidence < r["threshold"])
                    for r, confidence in zip(requests, head_confidence)
                ]
                if any(r["mode"] == "embedding" for r in requests):
                    embeddings = logical.float().cpu().tolist()
                if not any(fallback):
                    break
        if scores is None or fallback is None:
            raise RuntimeError("Decision head depth was not executed")
        # Keep accepted rows' head scores. Mixed batches continue together; no
        # incorrect KV compaction or per-row early-exit claim is made.
        if any(fallback):
            hidden, _ = decoder.norm(hidden, residual)
        self.batch_scores = scores
        self.batch_results = []
        for index, (request, use_full) in enumerate(zip(requests, fallback)):
            result = {
                "request_id": request["request_id"],
                "used_full_decoder": use_full,
                "decoder_depth": executed if use_full else self.decision_spec["depth"],
                "batch_decoder_depth": executed,
                "head_confidence": head_confidence[index],
                "threshold": request["threshold"],
                "input_tokens": lengths[index],
                "batch_size": len(requests),
                "audio_encoder_items": 1,
                "batch_audio_encoder_calls": self.decision_encoder_calls,
                "batch_audio_items": self.batch_audio_items,
            }
            if request["mode"] == "embedding":
                result["embedding"] = embeddings[index]
            self.batch_results.append(result)
        if inputs_embeds is not None:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))
        return hidden

    def compute_logits(self, hidden_states):
        if self.batch_provider is None:
            return super().compute_logits(hidden_states)
        if self.batch_requests is None or hidden_states.shape[0] != len(self.batch_requests):
            raise RuntimeError("Sampled rows do not match the scheduler decision batch")
        scores = self.batch_scores.clone()
        if any(r["used_full_decoder"] for r in self.batch_results):
            # Call the ordinary parent logits path; legacy serial state is unset.
            stock = super().compute_logits(hidden_states)
            for index, (request, result) in enumerate(zip(self.batch_requests, self.batch_results)):
                if result["used_full_decoder"]:
                    scores[index] = stock[index, self.decision_ids].double()
                    if request["mode"] != "raw":
                        scores[index] /= self.decision_spec["full_temperature"]
        probabilities = torch.softmax(scores, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Nonfinite decision probabilities")
        # These three FP64 readouts already cross to the host for the response.
        # vLLM's optional invariant CUDA log-softmax only supports lower dtypes.
        logprobs = torch.log_softmax(scores.cpu(), dim=-1).tolist()
        actions = self.decision_spec["actions"]
        for index, (result, p) in enumerate(zip(self.batch_results, probabilities.cpu().tolist())):
            key = result["request_id"]
            if key in self.completed_decisions:
                raise RuntimeError("Duplicate decision result")
            result.update(
                action=actions[max(range(3), key=p.__getitem__)],
                probabilities=dict(zip(actions, p)),
                confidence=max(p),
                label_logprobs=logprobs[index],
            )
            self.completed_decisions[key] = result
        logits = torch.full(
            (len(self.batch_requests), self.config.text_config.vocab_size),
            -float("inf"),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        logits[:, self.decision_ids] = scores.float()
        return logits

    def clear_batch(self):
        self.batch_provider = None
        self.batch_requests = None
        self.batch_scores = None
        self.batch_results = None

    def take_result(self, request_id):
        return self.completed_decisions.pop(request_id, None)
