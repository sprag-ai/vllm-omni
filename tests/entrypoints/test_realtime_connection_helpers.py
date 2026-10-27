# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field

import numpy as np
import pytest
import torch
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection, _PendingToolCall, _Round
from vllm_omni.entrypoints.openai.realtime_conversation import Conversation

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


@dataclass
class _FakeModelConfig:
    """Only the field tests vary."""

    async_chunk: bool = False


@dataclass
class _FakeServing:
    model_config: _FakeModelConfig = field(default_factory=_FakeModelConfig)


@pytest.fixture
def tool_call_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._tools = None
    conn._tool_result_queue = asyncio.Queue()
    conn._pending_tool_calls = {}
    conn._awaiting_tool_results = False
    conn._cancel_requested = False
    conn._live_item_id = None
    conn._live_truncate_ms = None
    conn._conversation = Conversation()
    conn.generation_task = None
    conn._unreported_dropped = []
    conn._unreported_added = []
    conn.serving = _FakeServing()
    # handle_event's session.update branch now also assigns these.
    conn._speaker = None
    conn._instructions = None
    return conn


@pytest.fixture
def instructions_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._instructions = None
    # ...and the siblings session.update also assigns.
    conn._tools = None
    conn._speaker = None
    # session.update now consults serving.model_config for the async_chunk gate.
    conn.serving = _FakeServing()
    return conn


@pytest.fixture
def speaker_conn() -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn._speaker = None
    # ...and the siblings, so a session.update event exercises the whole branch.
    conn._tools = None
    conn._instructions = None
    # session.update now consults serving.model_config for the async_chunk gate.
    conn.serving = _FakeServing()
    return conn


class TestRealtimeConnectionTensorAndPcm:
    def test_tensor_to_numpy_none(self) -> None:
        assert RealtimeConnection._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_1d_numpy(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (2,)

    def test_tensor_to_numpy_2d_numpy_flattened(self) -> None:
        arr = np.array([[0.5], [-0.5]], dtype=np.float32)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.shape == (2,)

    def test_tensor_to_numpy_torch(self) -> None:
        t = torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        out = RealtimeConnection._tensor_to_numpy(t)
        assert out is not None
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.25, -0.25], rtol=1e-5)

    def test_pcm16_b64_roundtrip(self) -> None:
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        b64 = RealtimeConnection._pcm16_b64(audio)
        raw = base64.b64decode(b64)
        assert len(raw) == 6
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert pcm[0] == 0
        assert pcm[1] == 32767
        assert pcm[2] == -32767


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)


class TestRealtimeConnectionToolCallEventRouting:
    """handle_event's tool-calling additions (session.update.tools capture,
    conversation.item.create routing) - see realtime_tool_calls.py for the
    <tool_call> text parser these events feed."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_tools_and_delegates_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        event = {"type": "session.update", "model": "qwen3-omni", "tools": tools}

        asyncio.run(tool_call_conn.handle_event(event))

        assert tool_call_conn._tools == tools
        base_handle_event.assert_awaited_once_with(event)

    def test_session_update_without_tools_leaves_existing_tools_untouched(self, tool_call_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
        tool_call_conn._tools = [{"type": "function", "function": {"name": "existing"}}]

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert tool_call_conn._tools == [{"type": "function", "function": {"name": "existing"}}]

    def test_function_call_output_goes_to_the_turn_waiting_for_it(self, tool_call_conn) -> None:
        tool_call_conn._awaiting_tool_results = True
        item = {"type": "function_call_output", "call_id": "call_1", "output": "sunny and 72"}

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": item}))

        assert tool_call_conn._tool_result_queue.get_nowait() == item
        assert tool_call_conn._conversation.items == []

    def test_a_result_for_the_running_turns_call_is_queued_before_its_wait(self, tool_call_conn, mocker) -> None:
        tool_call_conn.generation_task = mocker.Mock(done=lambda: False)
        tool_call_conn._pending_tool_calls = {0: _PendingToolCall(call_id="call_1", name="get_weather")}
        item = {"type": "function_call_output", "call_id": "call_1", "output": "sunny and 72"}

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": item}))

        assert tool_call_conn._tool_result_queue.get_nowait() == item
        assert tool_call_conn._conversation.items == []

    def test_function_call_output_without_a_waiting_turn_joins_the_conversation(self, tool_call_conn) -> None:
        item = {"type": "function_call_output", "id": "item_out", "call_id": "call_1", "output": "sunny and 72"}

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": item}))

        assert tool_call_conn._tool_result_queue.empty()
        (added,) = tool_call_conn._conversation.items
        assert (added.item_id, added.kind, added.call_id, added.text) == (
            "item_out",
            "function_call_output",
            "call_1",
            "sunny and 72",
        )

    def test_conversation_item_create_unsupported_item_type_sends_error(self, tool_call_conn, mocker) -> None:
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)

        asyncio.run(tool_call_conn.handle_event({"type": "conversation.item.create", "item": {"type": "not_a_thing"}}))

        send_error.assert_awaited_once()
        assert tool_call_conn._tool_result_queue.empty()

    def test_unrelated_event_types_still_delegate_to_base(self, tool_call_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(tool_call_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


class TestRealtimeConnectionSpeakerRouting:
    """handle_event's voice/speaker-selection addition to session.update -
    threaded into buffer_realtime_audio's own `speaker` param (see
    Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio), the same
    `additional_information={"speaker": [...]}` shape /v1/chat/completions
    already uses (serving_chat.py)."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_voice_field(self, speaker_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)

        asyncio.run(speaker_conn.handle_event({"type": "session.update", "model": "qwen3-omni", "voice": "aiden"}))

        assert speaker_conn._speaker == "aiden"

    def test_session_update_captures_speaker_field(self, speaker_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)

        asyncio.run(speaker_conn.handle_event({"type": "session.update", "model": "qwen3-omni", "speaker": "ethan"}))

        assert speaker_conn._speaker == "ethan"

    def test_session_update_without_voice_or_speaker_leaves_existing_value_untouched(
        self, speaker_conn, mocker
    ) -> None:
        self._patch_base_handle_event(mocker)
        speaker_conn._speaker = "aiden"

        asyncio.run(speaker_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert speaker_conn._speaker == "aiden"

    def test_unrelated_event_types_still_delegate_to_base(self, speaker_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(speaker_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


class TestRealtimeConnectionInstructionsRouting:
    """handle_event's instructions (system prompt) addition to session.update."""

    def _patch_base_handle_event(self, mocker):
        return mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

    def test_session_update_captures_instructions(self, instructions_conn, mocker) -> None:
        self._patch_base_handle_event(mocker)
        event = {
            "type": "session.update",
            "model": "qwen3-omni",
            "instructions": "Only use tools directly relevant to what the user asked.",
        }

        asyncio.run(instructions_conn.handle_event(event))

        assert instructions_conn._instructions == "Only use tools directly relevant to what the user asked."

    def test_session_update_without_instructions_leaves_existing_value_untouched(
        self, instructions_conn, mocker
    ) -> None:
        self._patch_base_handle_event(mocker)
        instructions_conn._instructions = "existing"

        asyncio.run(instructions_conn.handle_event({"type": "session.update", "model": "qwen3-omni"}))

        assert instructions_conn._instructions == "existing"

    def test_unrelated_event_types_still_delegate_to_base(self, instructions_conn, mocker) -> None:
        base_handle_event = self._patch_base_handle_event(mocker)
        event = {"type": "input_audio_buffer.commit", "final": True}

        asyncio.run(instructions_conn.handle_event(event))

        base_handle_event.assert_awaited_once_with(event)


def _tool_round(*call_ids: str) -> _Round:
    return _Round(tool_calls=[_PendingToolCall(call_id=c, name="get_weather") for c in call_ids])


class TestToolChainIsBounded:
    """A model that keeps re-emitting tool calls instead of answering must not
    loop without bound."""

    def _turn(self, conn, mocker, rounds: list[_Round]):
        conn._is_connected = True
        conn._next_user_item_id = conn._next_response_item_id = None
        conn.audio_queue = asyncio.Queue()
        mocker.patch.object(conn, "_prepare_prompt", new_callable=mocker.AsyncMock, return_value=(None, [], []))
        generate = mocker.patch.object(conn, "_generate_round", new_callable=mocker.AsyncMock, side_effect=rounds)
        mocker.patch.object(
            conn,
            "_await_tool_results",
            new_callable=mocker.AsyncMock,
            side_effect=lambda calls: [(call, {"output": "ok"}) for call in calls],
        )
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(conn, "send", new_callable=mocker.AsyncMock)
        send_json = mocker.patch.object(conn, "send_json", new_callable=mocker.AsyncMock)
        asyncio.run(conn._run_conversation_turn(None))
        return generate, send_error, send_json

    def test_exceeding_max_rounds_reports_error_and_stops(self, tool_call_conn, mocker) -> None:
        rounds = [_tool_round(f"call_{i}") for i in range(RealtimeConnection.MAX_TOOL_ROUNDS + 1)]

        generate, send_error, send_json = self._turn(tool_call_conn, mocker, rounds)

        assert generate.await_count == RealtimeConnection.MAX_TOOL_ROUNDS + 1
        assert send_error.await_args.args[1] == "tool_call_loop"
        (done,) = [c.args[0] for c in send_json.await_args_list if c.args[0]["type"] == "response.done"]
        assert done["status"] == "failed"

    def test_rounds_below_the_cap_still_continue(self, tool_call_conn, mocker) -> None:
        generate, send_error, _ = self._turn(tool_call_conn, mocker, [_tool_round("call_a"), _Round(spoken="Sunny.")])

        assert generate.await_count == 2
        send_error.assert_not_awaited()
        kinds = [(item.kind, item.call_id) for item in tool_call_conn._conversation.items]
        assert kinds == [("function_call", "call_a"), ("function_call_output", "call_a"), ("message", None)]


class TestToolResultWaitReleasesOnDisconnect:
    """A client that vanishes mid-tool-call must not park the generation task
    forever. The wait is bounded so `_is_connected` is re-checked instead of
    blocking indefinitely on an empty queue."""

    def test_wait_returns_when_client_disconnects(self, tool_call_conn) -> None:
        tool_call_conn._is_connected = False
        calls = [_PendingToolCall(call_id="call_x", name="get_weather")]

        result = asyncio.run(asyncio.wait_for(tool_call_conn._await_tool_results(calls), timeout=5))

        assert result is None
        assert tool_call_conn._awaiting_tool_results is False


class TestParallelToolResultsKeepCallOrder:
    """Each result is its own tool message, in the order the model made the calls."""

    @staticmethod
    def _wait(conn, *call_ids: str) -> list[str]:
        conn._is_connected = True
        calls = [_PendingToolCall(call_id=c, name="get_weather") for c in call_ids]
        results = asyncio.run(asyncio.wait_for(conn._await_tool_results(calls), timeout=5))
        return [item["output"] for _, item in results]

    def test_two_results_come_back_in_call_order(self, tool_call_conn) -> None:
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny 72"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_b", "output": "rainy 55"})

        assert self._wait(tool_call_conn, "call_a", "call_b") == ["sunny 72", "rainy 55"]

    def test_call_order_is_kept_when_results_arrive_reversed(self, tool_call_conn) -> None:
        """Arrival order is the client's choice; call order is the model's."""
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_b", "output": "rainy 55"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny 72"})

        assert self._wait(tool_call_conn, "call_a", "call_b") == ["sunny 72", "rainy 55"]

    def test_a_duplicate_result_for_one_call_does_not_desync_the_wait(self, tool_call_conn) -> None:
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "first"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "second"})

        assert self._wait(tool_call_conn, "call_a") == ["first"]


class TestToolResultValidation:
    """Review #5555: any dict with `type="function_call_output"` was enqueued, so a
    missing/non-string `call_id` or a non-string `output` was accepted and the turn
    then waited for a result that could never match - with nothing reported to the
    client. Shape problems are protocol errors."""

    @staticmethod
    def _create(item: dict) -> dict:
        return {"type": "conversation.item.create", "item": item}

    def _run(self, conn, mocker, item: dict):
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        asyncio.run(conn.handle_event(self._create(item)))
        return send_error

    def test_missing_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_non_string_call_id_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(tool_call_conn, mocker, {"type": "function_call_output", "call_id": 7, "output": "x"})
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_non_string_output_is_rejected(self, tool_call_conn, mocker) -> None:
        send_error = self._run(
            tool_call_conn, mocker, {"type": "function_call_output", "call_id": "call_a", "output": {"a": 1}}
        )
        assert send_error.await_args.args[1] == "invalid_function_call_output"
        assert tool_call_conn._tool_result_queue.empty()

    def test_well_formed_result_is_enqueued(self, tool_call_conn, mocker) -> None:
        tool_call_conn._awaiting_tool_results = True
        send_error = self._run(
            tool_call_conn, mocker, {"type": "function_call_output", "call_id": "call_a", "output": "sunny"}
        )
        send_error.assert_not_awaited()
        assert tool_call_conn._tool_result_queue.qsize() == 1

    def test_unknown_call_id_is_reported_to_the_client(self, tool_call_conn, mocker) -> None:
        """Previously only logged, so a client typo produced silence."""
        tool_call_conn._is_connected = True
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_TYPO", "output": "sunny"})
        tool_call_conn._tool_result_queue.put_nowait({"call_id": "call_a", "output": "sunny"})
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        calls = [_PendingToolCall(call_id="call_a", name="get_weather")]

        asyncio.run(asyncio.wait_for(tool_call_conn._await_tool_results(calls), timeout=5))

        assert send_error.await_args.args[1] == "unknown_tool_call_id"


class TestToolsAcceptedUnderAsyncChunk:
    """A conversation turn is one decode, so tools work whether or not async_chunk is on."""

    @pytest.mark.parametrize("async_chunk", [True, False])
    def test_tools_are_accepted(self, tool_call_conn, mocker, async_chunk: bool) -> None:
        tool_call_conn.serving = _FakeServing(_FakeModelConfig(async_chunk=async_chunk))
        send_error = mocker.patch.object(tool_call_conn, "send_error", new_callable=mocker.AsyncMock)
        mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        asyncio.run(tool_call_conn.handle_event({"type": "session.update", "model": "m", "tools": tools}))

        send_error.assert_not_awaited()
        assert tool_call_conn._tools == tools


class TestRenderPromptPropagatesAdditionalInformation:
    """Regression test for a real bug: BaseRenderer.render_cmpl_async's
    internal pipeline (process_for_engine_async) only carries over fields
    it explicitly knows about, so `additional_information` set on the
    pre-render prompt (buffer_realtime_audio's `speaker`) was silently
    dropped and never reached the engine - the voice selection feature
    looked correct in isolation but produced no audible effect. Fixed by
    reapplying it to the rendered engine_input, mirroring how
    serving_chat.py._preprocess_chat does it for /v1/chat/completions."""

    def _conn_with_fake_renderer(self, mocker, rendered_engine_input: dict):
        conn = RealtimeConnection.__new__(RealtimeConnection)
        conn.serving = mocker.Mock()
        conn.serving.model_config.is_encoder_decoder = False
        conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(return_value=[rendered_engine_input])
        return conn

    def test_additional_information_survives_render(self, mocker) -> None:
        # render_cmpl_async's fake return simulates the real pipeline: it never
        # echoes back fields it doesn't recognize, so additional_information
        # must NOT be in here even though the input prompt has it.
        rendered = {"prompt_token_ids": [1, 2, 3]}
        conn = self._conn_with_fake_renderer(mocker, rendered)
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3], additional_information={"speaker": ["aiden"]})

        result = asyncio.run(conn._render_prompt(prompt))

        assert result.prompt["additional_information"] == {"speaker": ["aiden"]}

    def test_no_additional_information_is_a_noop(self, mocker) -> None:
        rendered = {"prompt_token_ids": [1, 2, 3]}
        conn = self._conn_with_fake_renderer(mocker, rendered)
        prompt = TokensPrompt(prompt_token_ids=[1, 2, 3])

        result = asyncio.run(conn._render_prompt(prompt))

        assert "additional_information" not in result.prompt
