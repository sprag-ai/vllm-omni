# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A speculative Thinker step hands the Talker every committed row, and no others."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.model_executor.stage_input_processors.qwen3_omni as q3
from vllm_omni.model_executor.models.qwen3_omni.tool_speech import TOOL_CALL_START_TOKEN_ID

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ACCEPT = 24
HIDDEN = 8
PROMPT = [151644, 872, 5, 6, 151645, 151644, 77091, 198]
EOS = 151645


class _Manager:
    def __init__(self) -> None:
        self.put_req_chunk: defaultdict[str, int] = defaultdict(int)
        self.request_payload: dict[str, dict] = {}
        self._pending_streaming_prefills: dict[str, dict] = {}
        self.config = SimpleNamespace(
            hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=ACCEPT))
        )

    def process(self, rows, request, new_token_ids, is_finished=False):
        payload = q3.thinker2talker_async_chunk(
            self, _output(rows), request, is_finished=is_finished, new_token_ids=new_token_ids
        )
        if payload is not None:
            self.put_req_chunk[request.external_req_id] += 1
        return payload


def _request(output_token_ids, *, history=True, resumable=False):
    return SimpleNamespace(
        external_req_id="r",
        prompt_token_ids=list(PROMPT) if history else [],
        output_token_ids=list(output_token_ids) if history else [],
        all_token_ids=list(PROMPT) + list(output_token_ids) if history else [],
        output_token_count=len(output_token_ids),
        last_output_token_id=output_token_ids[-1] if output_token_ids else None,
        resumable=resumable,
        additional_information=None,
        sampling_params=SimpleNamespace(all_stop_token_ids={EOS}, max_tokens=64),
    )


def _output(token_ids):
    layers = {
        0: torch.tensor([[float(t)] * HIDDEN for t in token_ids], dtype=torch.bfloat16),
        ACCEPT: torch.randn(len(token_ids), HIDDEN).to(torch.bfloat16),
    }
    embed = {"tts_bos": torch.ones(1, HIDDEN), "tts_eos": torch.ones(1, HIDDEN), "tts_pad": torch.ones(1, HIDDEN)}
    return {"hidden_states": {"layers": layers}, "embed": embed}


def _spoken(commits, *, is_finished_last=False):
    """Run a captured turn; ``commits`` is each decode step's committed tokens.

    A captured step's rows are the previous step's last token followed by the
    accepted drafts, so its row count is the committed count before any stop trim.
    """
    manager = _Manager()
    first = commits[0]
    assert manager.process(PROMPT, _request(first), first) is None
    output, rows = list(first), []
    for i, (accepted, committed) in enumerate(commits[1:]):
        step_rows = [output[-1], *accepted[:-1]]
        output.extend(committed)
        history = manager.put_req_chunk["r"] == 0
        finished = is_finished_last and i == len(commits) - 2
        payload = manager.process(step_rows, _request(output, history=history), committed, is_finished=finished)
        if payload is None:
            continue
        sent = payload.embed.prefill[len(PROMPT) :] if payload.embed.prefill is not None else payload.embed.decode
        rows.extend(int(row[0]) for row in sent)
    return rows, manager


def test_every_committed_row_is_spoken():
    # (accepted drafts + bonus as sampled, committed after the stop trim)
    rows, _ = _spoken([[11], ([12, 13, 14], [12, 13, 14]), ([15], [15]), ([16, EOS], [16, EOS])])

    assert rows == [11, 12, 13, 14, 15, 16]


def test_rows_past_a_stop_inside_an_accepted_span_are_dropped():
    rows, _ = _spoken([[11], ([12], [12]), ([13, EOS, 99], [13, EOS])])

    assert rows == [11, 12, 13]


def test_a_stop_in_the_first_decode_step_is_dropped_from_the_held_prefill():
    rows, _ = _spoken([[11], ([12, EOS, 99], [12, EOS])])

    assert rows == [11, 12]


def test_speech_ends_at_a_tool_call_inside_an_accepted_span():
    rows, manager = _spoken(
        [[11], ([12], [12]), ([13, TOOL_CALL_START_TOKEN_ID, 99, 100], [13, TOOL_CALL_START_TOKEN_ID, 99, 100])]
    )

    assert rows == [11, 12, 13]
    assert manager.request_payload["r"].get(q3._TOOL_SPEECH_CUT) is True


def test_a_streaming_multi_row_decode_step_is_not_a_new_segment():
    manager = _Manager()
    request = _request([11, 12, 13, 14], resumable=True)
    rows = torch.tensor([[float(t)] * HIDDEN for t in (11, 12, 13)])

    payload = q3._construct_thinker2talker_streaming_input_async_chunk(
        False, request, rows, torch.randn(3, HIDDEN), manager, (12, 13, 14)
    )

    assert payload is not None and payload.embed.prefill is None
    assert [int(row[0]) for row in payload.embed.decode] == [11, 12, 13]
    assert "r" not in manager._pending_streaming_prefills


def test_a_streaming_final_step_trims_rows_past_the_stop():
    manager = _Manager()
    request = _request([11, 12, 13, EOS], resumable=True)
    rows = torch.tensor([[float(t)] * HIDDEN for t in (12, 13, EOS)])

    payload = q3._construct_thinker2talker_streaming_input_async_chunk(
        True, request, rows, torch.randn(3, HIDDEN), manager, (13, EOS)
    )

    assert [int(row[0]) for row in payload.embed.decode] == [12, 13]


def test_a_streaming_new_segment_is_still_cached():
    manager = _Manager()
    request = _request([11], resumable=True)
    rows = torch.tensor([[float(t)] * HIDDEN for t in PROMPT])

    payload = q3._construct_thinker2talker_streaming_input_async_chunk(
        False, request, rows, torch.randn(len(PROMPT), HIDDEN), manager, (11,)
    )

    assert payload is None and "r" in manager._pending_streaming_prefills
