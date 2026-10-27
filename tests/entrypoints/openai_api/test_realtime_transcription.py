# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from openai.types import realtime as types
from vllm.sampling_params import SamplingParams

from tests.entrypoints.openai_api.test_realtime_prompt_preflight import (
    _FakeWebSocket,
    _make_connection,
    _message_item,
    _websocket_events,
)
from vllm_omni.entrypoints.openai.realtime.connection import SAMPLE_RATE_HZ
from vllm_omni.entrypoints.openai.realtime.transcription import (
    DEFAULT_TRANSCRIPTION_PROMPT,
    transcript_token_limit,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _transcribing_engine(pieces: list[str], calls: list[dict[str, Any]]) -> SimpleNamespace:
    async def generate(**kwargs: Any):
        calls.append(kwargs)
        for piece in pieces:
            yield SimpleNamespace(
                final_output_type="text",
                outputs=[SimpleNamespace(text=piece, token_ids=[0] * len(piece.split()))],
            )

    async def abort(_request_id: str) -> None:
        return None

    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1000, multimodal_config=None),
        default_sampling_params_list=[SamplingParams(), SamplingParams()],
        generate=generate,
        abort=abort,
    )


async def _start_transcription(connection: Any, prompt: str | None = None) -> None:
    transcription: dict[str, Any] = {"model": "symphony"}
    if prompt is not None:
        transcription["prompt"] = prompt
    await connection._handle_session_update(
        types.SessionUpdateEvent(
            type="session.update",
            session={"type": "transcription", "audio": {"input": {"transcription": transcription}}},
        )
    )


async def _commit(connection: Any, seconds: float) -> None:
    connection.session.input_audio_buffer.extend(b"\x01\x00" * int(SAMPLE_RATE_HZ * seconds))
    await connection._handle_audio_commit(
        types.InputAudioBufferCommitEvent(type="input_audio_buffer.commit", event_id="evt_commit")
    )


async def _settle(connection: Any) -> None:
    await asyncio.gather(*list(connection._transcription._tasks))


@pytest.mark.asyncio
async def test_a_committed_turn_streams_its_transcript_and_usage() -> None:
    calls: list[dict[str, Any]] = []
    websocket = _FakeWebSocket()
    connection, renderer = _make_connection(
        websocket=websocket, engine=_transcribing_engine(["Hello", " world."], calls)
    )
    await _start_transcription(connection, prompt="Transcribe this audio verbatim.")

    await _commit(connection, 2.0)
    await _settle(connection)

    events = _websocket_events(websocket)
    kinds = [event["type"] for event in events]
    assert kinds == [
        "session.updated",
        "input_audio_buffer.committed",
        "conversation.item.added",
        "conversation.item.done",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.delta",
        "conversation.item.input_audio_transcription.completed",
    ]
    assert events[0]["session"]["type"] == "transcription"
    item_id = events[1]["item_id"]
    completed = events[-1]
    assert completed["item_id"] == item_id
    assert completed["transcript"] == "Hello world."
    assert completed["usage"]["type"] == "tokens"
    assert completed["usage"]["output_tokens"] == 2

    assert connection.session.items == []
    assert renderer.conversations[-1][0] == {"role": "system", "content": "Transcribe this audio verbatim."}
    assert calls[0]["output_modalities"] == ["text"]
    assert calls[0]["sampling_params_list"][0].max_tokens == transcript_token_limit(2.0)


@pytest.mark.asyncio
async def test_turns_are_transcribed_in_commit_order_with_the_default_prompt() -> None:
    calls: list[dict[str, Any]] = []
    websocket = _FakeWebSocket()
    connection, renderer = _make_connection(websocket=websocket, engine=_transcribing_engine(["text"], calls))
    await _start_transcription(connection)

    await _commit(connection, 1.0)
    await _commit(connection, 1.0)
    await _settle(connection)

    events = _websocket_events(websocket)
    committed = [event["item_id"] for event in events if event["type"] == "input_audio_buffer.committed"]
    completed = [
        event["item_id"] for event in events if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert completed == committed
    second_added = [event for event in events if event["type"] == "conversation.item.added"][1]
    assert second_added["previous_item_id"] == committed[0]
    assert renderer.conversations[-1][0]["content"] == DEFAULT_TRANSCRIPTION_PROMPT


@pytest.mark.asyncio
async def test_a_failed_transcription_reports_the_failure() -> None:
    async def generate(**_kwargs: Any):
        raise RuntimeError("engine down")
        yield  # pragma: no cover

    engine = _transcribing_engine([], [])
    engine.generate = generate
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(websocket=websocket, engine=engine)
    await _start_transcription(connection)

    await _commit(connection, 1.0)
    await _settle(connection)

    failed = _websocket_events(websocket)[-1]
    assert failed["type"] == "conversation.item.input_audio_transcription.failed"
    assert failed["error"]["message"] == "engine down"


@pytest.mark.asyncio
async def test_a_transcription_session_refuses_response_create() -> None:
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(websocket=websocket, engine=_transcribing_engine([], []))
    await _start_transcription(connection)

    await connection._handle_response_create(types.ResponseCreateEvent(type="response.create", event_id="evt_create"))

    event = _websocket_events(websocket)[-1]
    assert event["type"] == "error"
    assert "transcription session" in event["error"]["message"]


@pytest.mark.asyncio
async def test_a_conversation_cannot_become_a_transcription_session() -> None:
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(websocket=websocket, engine=_transcribing_engine([], []))
    connection.session.insert_item(_message_item("hello"))

    await _start_transcription(connection)

    event = _websocket_events(websocket)[-1]
    assert event["type"] == "error"
    assert connection._transcription is None


@pytest.mark.asyncio
async def test_a_transcription_session_cannot_switch_back_to_realtime() -> None:
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(websocket=websocket, engine=_transcribing_engine([], []))
    await _start_transcription(connection)

    await connection._handle_session_update(
        types.SessionUpdateEvent(type="session.update", session={"type": "realtime", "instructions": "hi"})
    )

    event = _websocket_events(websocket)[-1]
    assert event["type"] == "error"
    assert connection._transcription is not None


def test_the_transcript_token_limit_grows_with_audio_duration() -> None:
    assert transcript_token_limit(0.0) == 128
    assert transcript_token_limit(6.97) == 128 + 8 * 7


@pytest.mark.asyncio
async def test_a_commit_past_the_pending_turn_limit_is_refused_and_keeps_its_audio() -> None:
    release = asyncio.Event()

    async def generate(**_kwargs: Any):
        await release.wait()
        yield SimpleNamespace(final_output_type="text", outputs=[SimpleNamespace(text="text", token_ids=[0])])

    engine = _transcribing_engine([], [])
    engine.generate = generate
    websocket = _FakeWebSocket()
    connection, _ = _make_connection(websocket=websocket, engine=engine)
    connection._max_pending_turns = 2
    await _start_transcription(connection)

    await _commit(connection, 1.0)
    await _commit(connection, 1.0)
    await _commit(connection, 1.0)

    refusal = _websocket_events(websocket)[-1]
    assert refusal["type"] == "error"
    assert refusal["error"]["code"] == "input_backpressure"
    assert refusal["error"]["event_id"] == "evt_commit"
    assert len(connection.session.input_audio_buffer) == 2 * SAMPLE_RATE_HZ

    release.set()
    await _settle(connection)
    completed = [
        event
        for event in _websocket_events(websocket)
        if event["type"] == "conversation.item.input_audio_transcription.completed"
    ]
    assert len(completed) == 2


def test_the_realtime_handler_takes_the_pending_turn_limit_from_the_deploy_config() -> None:
    from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
    from vllm_omni.entrypoints.duplex.openai import OpenAIRealtimeHandler

    engine = SimpleNamespace(duplex_session_config=DuplexSessionRuntimeConfig(max_pending_turns_per_session=7))

    assert OpenAIRealtimeHandler.from_engine_client(engine)._max_pending_turns == 7
