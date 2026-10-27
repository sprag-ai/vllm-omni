# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conversation turns on /v1/realtime: history across turns, item events, cancel, tool rounds, and usage."""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.engine.protocol import StreamingInput
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDone
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai import realtime_connection
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.entrypoints.openai.realtime_conversation import Conversation, ConversationItem
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_the_omni_model_names_its_realtime_audio_placeholder() -> None:
    assert (
        Qwen3OmniMoeForConditionalGeneration.realtime_audio_placeholder() == "<|audio_start|><|audio_pad|><|audio_end|>"
    )


AUDIO_TOKENS = 50


def _text(text: str, cached: int = 0, prompt: int = 100, tokens: int = 1) -> SimpleNamespace:
    completion = SimpleNamespace(text=text, token_ids=[1] * tokens, finish_reason=None)
    return SimpleNamespace(
        stage_id=0,
        outputs=[completion],
        prompt_token_ids=[0] * prompt,
        num_cached_tokens=cached,
        multimodal_output=None,
    )


def _audio(seconds: float) -> SimpleNamespace:
    return SimpleNamespace(
        stage_id=2,
        outputs=[],
        multimodal_output={"audio": np.full(int(seconds * 24_000), 0.1, dtype=np.float32), "sr": 24_000},
    )


class _FakeEngine:
    """Plays one scripted list of outputs per generate call and records the prompts it was given."""

    def __init__(self, rounds: list[list[SimpleNamespace]], on_output=None) -> None:
        self.default_sampling_params_list = [SamplingParams()]
        self.rounds = list(rounds)
        self.prompts: list[dict] = []
        self.on_output = on_output

    def generate(self, prompt, **kwargs):
        outputs = self.rounds.pop(0)

        async def stream():
            async for streaming_input in prompt:
                self.prompts.append(streaming_input.prompt)
            for output in outputs:
                yield output
                if self.on_output is not None:
                    outcome = self.on_output(output)
                    if asyncio.iscoroutine(outcome):
                        await outcome

        return stream()


@pytest.fixture
def conn(mocker) -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn.connection_id = "ws1"
    conn.serving = mocker.Mock()
    conn.serving.model_cls.realtime_audio_placeholder = lambda: "<audio>"
    conn.audio_queue = asyncio.Queue()
    conn.generation_task = None
    conn._is_connected = True
    conn._tools = None
    conn._speaker = None
    conn._instructions = None
    conn._transcription = False
    conn._pending_tool_calls = {}
    conn._tool_result_queue = asyncio.Queue()
    conn._awaiting_tool_results = False
    conn._conversation = Conversation()
    conn._next_user_item_id = None
    conn._next_response_item_id = None
    conn._cancel_requested = False
    conn._live_item_id = None
    conn._live_truncate_ms = None
    conn._realtime_audio_ref = None
    mocker.patch.object(conn, "_chat_text", side_effect=lambda messages: json.dumps(messages))
    mocker.patch.object(conn, "_sampling_rate", return_value=16_000)
    mocker.patch.object(
        realtime_connection,
        "cached_tokenizer_from_config",
        return_value=SimpleNamespace(encode=lambda text: [0] * 10),
    )

    async def render(prompt):
        audio = (prompt.get("multi_modal_data") or {}).get("audio") or []
        placeholders = {"audio": [SimpleNamespace(length=AUDIO_TOKENS) for _ in audio]} if audio else {}
        return StreamingInput(prompt={**prompt, "mm_placeholders": placeholders})

    mocker.patch.object(conn, "_render_prompt", side_effect=render)
    conn.sent = []
    mocker.patch.object(conn, "send", new_callable=mocker.AsyncMock, side_effect=lambda event: conn.sent.append(event))
    mocker.patch.object(
        conn, "send_json", new_callable=mocker.AsyncMock, side_effect=lambda payload: conn.sent.append(payload)
    )
    mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
    return conn


def _turn(conn: RealtimeConnection, seconds: float | None = 1.0, user: str | None = None, response: str | None = None):
    conn._next_user_item_id, conn._next_response_item_id = user, response
    conn.sent.clear()

    async def audio():
        yield np.zeros(int(seconds * 16_000), dtype=np.float32)

    asyncio.run(conn._run_conversation_turn(None if seconds is None else audio()))
    return conn.sent


def _event(sent: list, event_type: str) -> dict:
    (event,) = [e for e in sent if isinstance(e, dict) and e["type"] == event_type]
    return event


class TestHistory:
    def test_a_turn_records_the_user_audio_and_the_reply_under_the_gateway_ids(self, conn) -> None:
        conn.engine = _FakeEngine([[_text("Hello", cached=40), _audio(0.5), _text(" there.")]])

        sent = _turn(conn, user="u1", response="r1")

        assert [(i.item_id, i.role) for i in conn._conversation.items] == [("u1", "user"), ("r1", "assistant")]
        reply = conn._conversation.get("r1")
        assert reply.text == "Hello there."
        assert reply.audio_out_seconds == pytest.approx(0.5)
        done = _event(sent, "response.done")
        assert done["status"] == "completed"
        assert done["item_ids"] == ["u1", "r1"]
        assert done["context_item_ids"] == ["u1", "r1"]
        assert done["usage"]["prompt_tokens"] == 100
        assert done["usage"]["completion_tokens"] == 2
        assert done["usage"]["prompt_tokens_details"] == {
            "cached_tokens": 40,
            "multimodal_tokens": {"audio": AUDIO_TOKENS},
        }
        assert isinstance(sent[-3], TranscriptionDone)
        assert sent[-1] == {"type": "response.output_audio.done", "has_audio": True}

    def test_the_next_turn_generates_over_the_whole_conversation(self, conn) -> None:
        conn.engine = _FakeEngine([[_text("Hi.")], [_text("Again.")]])

        _turn(conn, user="u1", response="r1")
        _turn(conn, user="u2", response="r2")

        prompt = conn.engine.prompts[-1]
        assert len(prompt["multi_modal_data"]["audio"]) == 2
        keys = [conn._conversation.get(i).audio_uuid for i in ("u1", "u2")]
        assert prompt["multi_modal_uuids"] == {"audio": [f"ws1:{key}" for key in keys]}

    def test_a_reused_item_id_never_reuses_the_old_audio_cache_key(self, conn) -> None:
        conn.engine = _FakeEngine([[_text("Hi.")], [_text("Again.")]])
        _turn(conn, user="u1", response="r1")
        first = conn.engine.prompts[0]["multi_modal_uuids"]["audio"]

        conn._conversation.delete("u1")
        conn._conversation.delete("r1")
        _turn(conn, user="u1", response="r1")

        assert conn.engine.prompts[1]["multi_modal_uuids"]["audio"] != first
        assert conn._conversation.get("u1").audio_tokens == AUDIO_TOKENS

    def test_a_turn_without_audio_generates_over_the_conversation_as_it_stands(self, conn) -> None:
        conn._conversation.add(ConversationItem("t1", "message", "user", text="What time is it?"))
        conn.engine = _FakeEngine([[_text("Noon.")]])

        sent = _turn(conn, seconds=None, response="r1")

        assert "multi_modal_data" not in conn.engine.prompts[0]
        assert _event(sent, "response.done")["item_ids"] == ["r1"]

    def test_dropped_items_are_reported(self, conn, monkeypatch) -> None:
        monkeypatch.setenv("SPRAG_REALTIME_CONTEXT_TOKENS", "100")
        conn.engine = _FakeEngine([[_text("Hi.")], [_text("Again.")]])

        _turn(conn, seconds=5, user="u1", response="r1")
        sent = _turn(conn, seconds=5, user="u2", response="r2")

        done = _event(sent, "response.done")
        assert done["dropped_item_ids"] == ["u1", "r1"]
        assert done["context_item_ids"] == ["u2", "r2"]


class TestCancel:
    def test_cancel_stops_the_turn_and_keeps_the_text_generated_so_far(self, conn) -> None:
        def cancel_after_first_word(output):
            conn._cancel_requested = True

        conn.engine = _FakeEngine([[_text("The train"), _text(" to Chicago.")]], on_output=cancel_after_first_word)

        sent = _turn(conn, user="u1", response="r1")

        assert conn._conversation.get("r1").text == "The train"
        assert _event(sent, "response.done")["status"] == "cancelled"
        assert sent[-1]["type"] == "response.output_audio.done"

    def test_a_truncate_during_the_reply_applies_once_the_reply_ends(self, conn) -> None:
        async def barge_in_after_audio(output):
            if output.stage_id == 2:
                await conn.handle_event({"type": "response.cancel"})
                await conn.handle_event({"type": "conversation.item.truncate", "item_id": "r1", "audio_end_ms": 1000})

        conn.engine = _FakeEngine(
            [[_text("The train to Chicago leaves at six."), _audio(2.0), _text(" More.")]],
            on_output=barge_in_after_audio,
        )

        async def run():
            async def stream():
                yield np.zeros(16_000, dtype=np.float32)

            conn._next_user_item_id, conn._next_response_item_id = "u1", "r1"
            conn.generation_task = asyncio.current_task()
            await conn._run_conversation_turn(stream())

        conn.sent.clear()
        asyncio.run(run())

        conn.send_error.assert_not_awaited()
        reply = conn._conversation.get("r1")
        assert reply.text == "The train"
        assert reply.audio_out_seconds == 1.0
        assert _event(conn.sent, "response.done")["status"] == "cancelled"

    def test_the_reply_item_is_announced_before_its_text_and_audio(self, conn) -> None:
        conn.engine = _FakeEngine([[_text("Hi."), _audio(0.5)]])

        sent = _turn(conn, user="u1", response="r1")

        announced = _event(sent, "response.output_item.added")
        assert announced["item"] == {"type": "message", "role": "assistant", "id": "r1"}
        assert sent.index(announced) == 0


class TestToolRounds:
    def test_cancel_releases_a_turn_waiting_for_tool_results(self, conn) -> None:
        call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Austin"}}\n</tool_call>'
        conn.engine = _FakeEngine([[_text(call)]])

        async def cancel_when_waiting():
            while not conn._awaiting_tool_results:
                await asyncio.sleep(0.01)
            conn._cancel_requested = True

        async def run():
            async def stream():
                yield np.zeros(16_000, dtype=np.float32)

            conn._next_user_item_id, conn._next_response_item_id = "u1", "r1"
            await asyncio.wait_for(
                asyncio.gather(conn._run_conversation_turn(stream()), cancel_when_waiting()), timeout=5
            )

        asyncio.run(run())

        assert _event(conn.sent, "response.done")["status"] == "cancelled"
        assert conn._awaiting_tool_results is False

    def test_a_tool_round_continues_with_the_result_and_usage_sums_over_rounds(self, conn) -> None:
        call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Austin"}}\n</tool_call>'
        conn.engine = _FakeEngine([[_text(call, prompt=100, cached=0)], [_text("Sunny.", prompt=130, cached=100)]])

        async def answer_when_asked():
            while not conn._awaiting_tool_results:
                await asyncio.sleep(0.01)
            call_id = next(iter(conn._pending_tool_calls.values())).call_id
            await conn.handle_event(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id, "output": "sunny"},
                }
            )

        async def run():
            audio = np.zeros(16_000, dtype=np.float32)

            async def stream():
                yield audio

            conn._next_user_item_id, conn._next_response_item_id = "u1", "r1"
            await asyncio.gather(conn._run_conversation_turn(stream()), answer_when_asked())

        asyncio.run(run())

        kinds = [(i.kind, i.role) for i in conn._conversation.items]
        assert kinds == [
            ("message", "user"),
            ("function_call", "assistant"),
            ("function_call_output", "tool"),
            ("message", "assistant"),
        ]
        second_round_messages = conn._chat_text.call_args_list[-1].args[0]
        assert {"role": "tool", "content": "sunny"} in second_round_messages
        done = _event(conn.sent, "response.done")
        assert done["usage"]["prompt_tokens"] == 230
        assert done["usage"]["prompt_tokens_details"]["cached_tokens"] == 100
        assert done["usage"]["prompt_tokens_details"]["multimodal_tokens"] == {"audio": 2 * AUDIO_TOKENS}


class TestItemEvents:
    @staticmethod
    def _handle(conn, event: dict) -> None:
        asyncio.run(conn.handle_event(event))

    def test_a_commit_hands_the_base_only_its_own_fields(self, conn, mocker) -> None:
        base = mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

        self._handle(
            conn, {"type": "input_audio_buffer.commit", "final": False, "item_id": "u1", "response_item_id": "r1"}
        )

        base.assert_awaited_once_with({"type": "input_audio_buffer.commit", "final": False})
        assert (conn._next_user_item_id, conn._next_response_item_id) == ("u1", "r1")

    def test_a_commit_naming_a_taken_id_reports_it_and_uses_a_new_one(self, conn, mocker) -> None:
        mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        conn._conversation.add(ConversationItem("u1", "message", "user", text="hi"))

        self._handle(
            conn, {"type": "input_audio_buffer.commit", "final": False, "item_id": "u1", "response_item_id": "r1"}
        )

        assert conn.send_error.await_args.args[1] == "item_conflict"
        assert (conn._next_user_item_id, conn._next_response_item_id) == (None, "r1")

    def test_a_commit_naming_one_id_twice_keeps_it_for_the_user_item(self, conn, mocker) -> None:
        mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)

        self._handle(
            conn, {"type": "input_audio_buffer.commit", "final": False, "item_id": "x1", "response_item_id": "x1"}
        )

        assert (conn._next_user_item_id, conn._next_response_item_id) == ("x1", None)

    def test_message_items_are_added_after_the_named_item(self, conn) -> None:
        conn._conversation.add(ConversationItem("u1", "message", "user", text="hi"))
        pcm = base64.b64encode(np.zeros(160, dtype=np.int16).tobytes()).decode()

        self._handle(
            conn,
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "id": "a0",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello."}],
                },
            },
        )
        self._handle(
            conn,
            {
                "type": "conversation.item.create",
                "previous_item_id": "u1",
                "item": {
                    "type": "message",
                    "id": "u0",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": pcm}],
                },
            },
        )

        assert [i.item_id for i in conn._conversation.items] == ["u1", "u0", "a0"]
        assert conn._conversation.get("u0").audio.shape == (160,)
        assert conn._conversation.get("a0").text == "Hello."

    @pytest.mark.parametrize(
        "item, code",
        [
            ({"type": "message", "role": "system", "content": []}, "unsupported_item"),
            ({"type": "message", "role": "user", "content": [{"type": "input_image"}]}, "unsupported_item"),
            ({"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "!!"}]}, "invalid_audio"),
        ],
    )
    def test_an_invalid_message_is_reported(self, conn, item: dict, code: str) -> None:
        self._handle(conn, {"type": "conversation.item.create", "item": item})

        assert conn.send_error.await_args.args[1] == code
        assert conn._conversation.items == []

    def test_a_duplicate_id_is_reported(self, conn) -> None:
        conn._conversation.add(ConversationItem("u1", "message", "user", text="hi"))

        self._handle(
            conn,
            {
                "type": "conversation.item.create",
                "item": {"type": "message", "id": "u1", "role": "user", "content": []},
            },
        )

        assert conn.send_error.await_args.args[1] == "item_conflict"

    def test_delete_removes_an_item_and_reports_a_missing_one(self, conn) -> None:
        conn._conversation.add(ConversationItem("u1", "message", "user", text="hi"))

        self._handle(conn, {"type": "conversation.item.delete", "item_id": "u1"})
        assert conn._conversation.items == []
        conn.send_error.assert_not_awaited()

        self._handle(conn, {"type": "conversation.item.delete", "item_id": "u1"})
        assert conn.send_error.await_args.args[1] == "item_not_found"

    def test_truncate_cuts_the_reply(self, conn) -> None:
        conn._conversation.add(
            ConversationItem(
                "a1", "message", "assistant", text="The train to Chicago leaves at six.", audio_out_seconds=4.0
            )
        )

        self._handle(conn, {"type": "conversation.item.truncate", "item_id": "a1", "audio_end_ms": 2000})

        assert conn._conversation.get("a1").text == "The train to"

    @pytest.mark.parametrize("audio_end_ms", [None, -1, True, 1.5])
    def test_truncate_requires_a_non_negative_integer(self, conn, audio_end_ms) -> None:
        self._handle(conn, {"type": "conversation.item.truncate", "item_id": "a1", "audio_end_ms": audio_end_ms})

        assert conn.send_error.await_args.args[1] == "invalid_value"

    def test_response_create_starts_a_turn_without_audio(self, conn, mocker) -> None:
        run = mocker.patch.object(conn, "_run_conversation_turn", new_callable=mocker.MagicMock)

        async def start():
            run.return_value = asyncio.sleep(0)
            await conn.handle_event({"type": "response.create"})
            await conn.generation_task

        asyncio.run(start())

        run.assert_called_once_with(None)

    def test_response_create_names_the_reply_item(self, conn, mocker) -> None:
        mocker.patch.object(conn, "_run_conversation_turn", return_value=asyncio.sleep(0))

        async def start():
            await conn.handle_event({"type": "response.create", "response_item_id": "r9"})
            await conn.generation_task

        asyncio.run(start())

        assert conn._next_response_item_id == "r9"

    def test_response_cancel_flags_a_running_turn(self, conn) -> None:
        async def run():
            conn.generation_task = asyncio.create_task(asyncio.sleep(1))
            await conn.handle_event({"type": "response.cancel"})
            conn.generation_task.cancel()

        asyncio.run(run())

        assert conn._cancel_requested is True
