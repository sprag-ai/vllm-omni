# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for the Qwen3-Omni thinker forward return contract.

Background
----------
Since omni architectures unconditionally override upstream's registry entries,
``Qwen3OmniMoeForConditionalGeneration`` (and the standalone thinker) are also
constructed by plain, non-staged vLLM runs — e.g. ``vllm serve
Qwen/Qwen3-Omni-30B-A3B-Instruct`` in the nightly vLLM-text perf job. Stock
vLLM's ``GPUModelRunner`` indexes the forward output directly
(``hidden_states[logit_indices]``), so a thinker forward that always returns a
``(hidden_states, captured_hidden_states)`` tuple kills the engine during
``profile_run`` with ``TypeError: only integer tensors of a single element can
be converted to an index`` (Buildkite build 2952).

Contract under test:

- No capture requested → the forward returns a **bare tensor** (stock-runner
  compatible).
- Capture requested (``capture_layer_indices`` + ``return_hidden_states``,
  staged runs with a talker downstream) → the forward returns the
  ``(hidden_states, captured_layer_dict)`` tuple omni's runner consumes via
  ``make_omni_output``.
- The combined model requests capture only in staged runs
  (``model_config.model_stage`` injected), never in plain vLLM runs.
- EAGLE3 aux layers set → the decoder appends a third element and the thinker
  re-nests it as ``((hidden_states, captured_layer_dict), aux_hidden_states)``,
  so the drafter's pair is the outer shape and omni's pair stays inner.
- ``make_omni_output`` tolerates every return shape.

Pure shape/contract tests: CPU-only, no weights, no GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3MoeLLMModel,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm_omni.worker.gpu_model_runner import _split_aux_hidden_states

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HIDDEN = 8
_TOKENS = 4


def _hidden_states() -> torch.Tensor:
    return torch.zeros(_TOKENS, _HIDDEN)


def test_single_rank_intermediate_capture_includes_deferred_residual(mocker):
    """Capture the Talker state on the default single-PP-rank path."""
    import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as module

    pp_group = mocker.Mock(is_first_rank=True, is_last_rank=True)
    mocker.patch.object(module, "get_pp_group", return_value=pp_group)

    class SplitStateLayer(nn.Module):
        def forward(self, positions, hidden_states, residual):
            next_residual = hidden_states if residual is None else hidden_states + residual
            return hidden_states * 2, next_residual

    class Combine(nn.Module):
        def forward(self, hidden_states, residual):
            return hidden_states + residual, None

    model = object.__new__(Qwen3MoeLLMModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([SplitStateLayer(), SplitStateLayer()])
    model.start_layer = 0
    model.end_layer = 2
    model.norm = Combine()
    inputs = torch.ones(_TOKENS, _HIDDEN)

    output_without_capture, _ = Qwen3MoeLLMModel.forward(
        model,
        input_ids=None,
        positions=torch.arange(_TOKENS),
        inputs_embeds=inputs,
    )
    output_with_capture, captured = Qwen3MoeLLMModel.forward(
        model,
        input_ids=None,
        positions=torch.arange(_TOKENS),
        inputs_embeds=inputs,
        capture_layer_indices=[0, 1],
        return_hidden_states=True,
    )

    layers = captured["hidden_states"]["layers"]
    torch.testing.assert_close(layers[0], inputs)
    torch.testing.assert_close(layers[1], torch.full_like(inputs, 3))
    torch.testing.assert_close(output_with_capture, output_without_capture)
    torch.testing.assert_close(output_with_capture, torch.full_like(inputs, 7))


class _InnerModelStub:
    """Stands in for ``Qwen3MoeLLMModel``: returns the internal 2- or 3-tuple."""

    def __init__(self, output=None, aux_rows=0):
        self.calls: list[dict] = []
        self._output = output
        self._aux_rows = aux_rows

    def __call__(self, input_ids, positions, intermediate_tensors, **kwargs):
        self.calls.append(kwargs)
        if self._output is not None:
            return self._output
        captured = {"hidden_states": {"layers": {0: _hidden_states()}}} if kwargs.get("return_hidden_states") else None
        if self._aux_rows:
            return _hidden_states(), captured, [_hidden_states() for _ in range(self._aux_rows)]
        return _hidden_states(), captured


def _make_thinker(
    inner_output=None, aux_rows: int = 0
) -> tuple[Qwen3OmniMoeThinkerForConditionalGeneration, _InnerModelStub]:
    thinker = object.__new__(Qwen3OmniMoeThinkerForConditionalGeneration)
    nn.Module.__init__(thinker)
    inner = _InnerModelStub(output=inner_output, aux_rows=aux_rows)
    thinker.use_deepstack = False
    thinker.language_model = SimpleNamespace(model=inner)
    return thinker, inner


def _forward_args():
    return {
        "input_ids": torch.zeros(_TOKENS, dtype=torch.long),
        "positions": torch.arange(_TOKENS),
    }


def test_thinker_forward_returns_bare_tensor_without_capture():
    thinker, _ = _make_thinker()
    out = thinker.forward(**_forward_args())
    assert isinstance(out, torch.Tensor)
    assert out.shape == (_TOKENS, _HIDDEN)


def test_thinker_forward_returns_tuple_when_capturing():
    thinker, _ = _make_thinker()
    out = thinker.forward(
        **_forward_args(),
        capture_layer_indices=[0, 2],
        return_hidden_states=True,
    )
    assert isinstance(out, tuple)
    hidden, captured = out
    assert isinstance(hidden, torch.Tensor)
    assert "hidden_states" in captured


def test_thinker_forward_nests_capture_inside_aux_pair():
    thinker, _ = _make_thinker(aux_rows=3)
    out = thinker.forward(
        **_forward_args(),
        capture_layer_indices=[0, 2],
        return_hidden_states=True,
    )
    primary, aux = out
    assert isinstance(aux, list) and len(aux) == 3
    assert all(isinstance(t, torch.Tensor) for t in aux)
    hidden, captured = primary
    assert isinstance(hidden, torch.Tensor)
    assert "hidden_states" in captured


def test_thinker_forward_returns_bare_primary_with_aux_and_no_capture():
    thinker, _ = _make_thinker(aux_rows=3)
    primary, aux = thinker.forward(**_forward_args())
    assert isinstance(primary, torch.Tensor)
    assert len(aux) == 3


def test_split_aux_rejects_the_capture_mapping():
    """A capture dict in slot 1 is not auxiliary hidden states.

    Treating it as aux both starves the drafter and empties the talker payload,
    and neither failure surfaces until a tensor op rejects a str.
    """
    captured = {"hidden_states": {"layers": {0: _hidden_states()}}}
    pair = (_hidden_states(), captured)
    out, aux = _split_aux_hidden_states(pair)
    assert aux is None
    assert out is pair

    aux_pair = (_hidden_states(), [_hidden_states(), _hidden_states()])
    out, aux = _split_aux_hidden_states(aux_pair)
    assert isinstance(aux, list) and len(aux) == 2
    assert isinstance(out, torch.Tensor)

    bare = _hidden_states()
    out, aux = _split_aux_hidden_states(bare)
    assert aux is None and out is bare


def test_thinker_forward_passes_through_intermediate_tensors():
    it = IntermediateTensors({"hidden_states": _hidden_states(), "residual": _hidden_states()})
    thinker, _ = _make_thinker(inner_output=it)
    out = thinker.forward(**_forward_args())
    assert out is it


class _ThinkerStub(nn.Module):
    """Records capture kwargs; mirrors the fixed thinker return contract."""

    def __init__(self):
        super().__init__()
        self.received: dict = {}

    def forward(self, **kwargs):
        self.received = kwargs
        if kwargs.get("return_hidden_states"):
            return _hidden_states(), {"hidden_states": {"layers": {}}}
        return _hidden_states()


def _make_combined(*, is_staged_run: bool, accept_hidden_layer=6) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "thinker"
    model.is_staged_run = is_staged_run
    model.talker_config = SimpleNamespace(accept_hidden_layer=accept_hidden_layer)
    model.thinker = _ThinkerStub()
    return model


def test_combined_forward_plain_run_skips_capture_and_returns_tensor():
    model = _make_combined(is_staged_run=False)
    out = model.forward(**_forward_args())
    assert "capture_layer_indices" not in model.thinker.received
    assert isinstance(out, torch.Tensor)


def test_combined_forward_staged_run_captures_and_returns_tuple():
    model = _make_combined(is_staged_run=True)
    out = model.forward(**_forward_args())
    assert model.thinker.received["capture_layer_indices"] == [0, 6]
    assert model.thinker.received["return_hidden_states"] is True
    assert isinstance(out, tuple)


def test_combined_forward_staged_run_without_accept_layer_returns_tensor():
    model = _make_combined(is_staged_run=True, accept_hidden_layer=None)
    out = model.forward(**_forward_args())
    assert "capture_layer_indices" not in model.thinker.received
    assert isinstance(out, torch.Tensor)


def test_make_omni_output_accepts_bare_tensor():
    model = _make_combined(is_staged_run=False)
    out = model.make_omni_output(_hidden_states())
    assert isinstance(out, OmniOutput)
    assert out.text_hidden_states.shape == (_TOKENS, _HIDDEN)
    assert out.multimodal_outputs == {}


def test_make_omni_output_accepts_capture_tuple():
    model = _make_combined(is_staged_run=True)
    captured: dict[str, dict[str, dict[int, torch.Tensor]]] = {"hidden_states": {"layers": {}}}
    out = model.make_omni_output((_hidden_states(), captured))
    assert isinstance(out, OmniOutput)
    assert out.multimodal_outputs == captured
