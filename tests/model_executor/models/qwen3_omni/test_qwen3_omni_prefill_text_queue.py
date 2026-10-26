# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Output text that reaches the talker inside its prefill is spoken, not dropped, under async_chunk."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IM_START, SYSTEM, USER, ASSISTANT, NEWLINE, AUDIO = 1, 2, 3, 4, 5, 6
PROMPT = [IM_START, SYSTEM, NEWLINE, IM_START, USER, AUDIO, NEWLINE, IM_START, ASSISTANT, NEWLINE]


def _model() -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.config = SimpleNamespace(im_start_token_id=IM_START, assistant_token_id=ASSISTANT)
    return model


def _rows(count: int) -> torch.Tensor:
    return torch.arange(count, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)


def _queue(model, outputs: int, embed: dict | None = None, update_dict: dict | None = None) -> torch.Tensor | None:
    rows = _rows(len(PROMPT) + outputs)
    prompt_ids = torch.tensor([PROMPT])
    all_ids = torch.tensor(PROMPT + [100 + i for i in range(outputs)])
    update_dict = {} if update_dict is None else update_dict
    model._queue_unspoken_prefill_text(rows, prompt_ids, all_ids, embed or {}, update_dict)
    return update_dict.get("embed", {}).get("cached_decode")


def test_a_speculative_first_step_queues_every_output_row_after_the_first() -> None:
    queued = _queue(_model(), outputs=4)

    assert queued[:, 0].tolist() == [len(PROMPT) + 1, len(PROMPT) + 2, len(PROMPT) + 3]


def test_a_single_output_row_queues_nothing() -> None:
    assert _queue(_model(), outputs=1) is None


def test_unspoken_rows_precede_decode_rows_already_queued() -> None:
    update_dict = {"embed": {"cached_decode": torch.full((2, 3), 99.0)}}

    queued = _queue(_model(), outputs=3, update_dict=update_dict)

    assert queued[:, 0].tolist() == [len(PROMPT) + 1, len(PROMPT) + 2, 99.0, 99.0]


def test_unspoken_rows_precede_a_queue_carried_on_the_payload() -> None:
    queued = _queue(_model(), outputs=2, embed={"cached_decode": torch.full((1, 3), 7.0)})

    assert queued[:, 0].tolist() == [len(PROMPT) + 1, 7.0]


def test_rows_past_the_known_ids_are_not_queued() -> None:
    model = _model()
    rows = _rows(len(PROMPT) + 5)
    update_dict: dict = {}

    model._queue_unspoken_prefill_text(
        rows, torch.tensor([PROMPT]), torch.tensor(PROMPT + [100, 101, 102]), {}, update_dict
    )

    assert update_dict["embed"]["cached_decode"][:, 0].tolist() == [len(PROMPT) + 1, len(PROMPT) + 2]
