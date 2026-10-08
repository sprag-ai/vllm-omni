# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transcription sessions on /v1/realtime: session routing, text-only output, segment joins, prompts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.entrypoints.speech_to_text.realtime.connection import RealtimeConnection as VllmRealtimeConnection
from vllm.entrypoints.speech_to_text.realtime.protocol import TranscriptionDelta, TranscriptionDone
from vllm.sampling_params import SamplingParams

from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection
from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RATE = 16_000


@dataclass
class _FakeModelConfig:
    async_chunk: bool = True


@dataclass
class _FakeServing:
    model_config: _FakeModelConfig = field(default_factory=_FakeModelConfig)


def _segment_output(text: str, finished: bool = False) -> SimpleNamespace:
    completion = SimpleNamespace(text=text, token_ids=[1] if text else [], finish_reason="stop" if finished else None)
    return SimpleNamespace(stage_id=0, outputs=[completion], prompt_token_ids=[0])


class _FakeEngine:
    def __init__(self, outputs: list[SimpleNamespace]) -> None:
        self.default_sampling_params_list = [SamplingParams()]
        self.outputs = outputs
        self.generate_kwargs: dict = {}

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs

        async def stream():
            for output in self.outputs:
                yield output

        return stream()


def _connection(transcription: bool, engine: _FakeEngine | None = None) -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn.connection_id = "ws-test"
    conn.serving = _FakeServing()
    conn.engine = engine
    conn.audio_queue = asyncio.Queue()
    conn._is_connected = True
    conn._tools = None
    conn._speaker = None
    conn._instructions = None
    conn._transcription = transcription
    conn._pending_tool_calls = {}
    return conn


def _run(conn: RealtimeConnection, mocker) -> tuple[list[str], str]:
    send = mocker.patch.object(conn, "send", new_callable=mocker.AsyncMock)
    mocker.patch.object(conn, "send_json", new_callable=mocker.AsyncMock)

    async def no_input():
        return
        yield

    asyncio.run(conn._run_generation(no_input(), asyncio.Queue()))
    events = [call.args[0] for call in send.await_args_list]
    deltas = [event.delta for event in events if isinstance(event, TranscriptionDelta)]
    done = [event.text for event in events if isinstance(event, TranscriptionDone)]
    assert len(done) == 1
    return deltas, done[0]


class TestSessionTypeRouting:
    def _handle(self, conn, mocker, event):
        base = mocker.patch.object(VllmRealtimeConnection, "handle_event", new_callable=mocker.AsyncMock)
        send_error = mocker.patch.object(conn, "send_error", new_callable=mocker.AsyncMock)
        asyncio.run(conn.handle_event({"type": "session.update", "model": "symphony", **event}))
        return base, send_error

    def test_transcription_session_type_enables_transcription(self, mocker) -> None:
        conn = _connection(transcription=False)
        base, _ = self._handle(conn, mocker, {"session_type": "transcription"})
        assert conn._transcription is True
        base.assert_awaited_once()

    def test_realtime_session_type_disables_transcription(self, mocker) -> None:
        conn = _connection(transcription=True)
        self._handle(conn, mocker, {"session_type": "realtime"})
        assert conn._transcription is False

    def test_absent_session_type_keeps_the_current_mode(self, mocker) -> None:
        conn = _connection(transcription=True)
        self._handle(conn, mocker, {})
        assert conn._transcription is True

    def test_unknown_session_type_is_refused(self, mocker) -> None:
        conn = _connection(transcription=False)
        base, send_error = self._handle(conn, mocker, {"session_type": "translation"})
        assert send_error.await_args.args[1] == "invalid_session_type"
        assert conn._transcription is False
        base.assert_not_awaited()


class TestTranscriptionGeneration:
    def test_requests_text_output_only(self, mocker) -> None:
        engine = _FakeEngine([_segment_output("Hello.", finished=True)])
        _run(_connection(transcription=True, engine=engine), mocker)
        assert engine.generate_kwargs["output_modalities"] == ["text"]

    def test_conversational_session_keeps_every_output_modality(self, mocker) -> None:
        engine = _FakeEngine([_segment_output("Hello.", finished=True)])
        _run(_connection(transcription=False, engine=engine), mocker)
        assert engine.generate_kwargs["output_modalities"] is None

    def test_joins_segments_with_a_space(self, mocker) -> None:
        engine = _FakeEngine(
            [
                _segment_output("when I started"),
                _segment_output(" listening.", finished=True),
                _segment_output("To"),
                _segment_output(" him.", finished=True),
            ]
        )
        deltas, text = _run(_connection(transcription=True, engine=engine), mocker)
        assert text == "when I started listening. To him."
        assert "".join(deltas) == text

    def test_separator_waits_for_the_next_non_empty_delta(self, mocker) -> None:
        engine = _FakeEngine(
            [
                _segment_output("listening.", finished=True),
                _segment_output(""),
                _segment_output("To him.", finished=True),
            ]
        )
        _, text = _run(_connection(transcription=True, engine=engine), mocker)
        assert text == "listening. To him."

    def test_unspaced_scripts_join_without_a_space(self, mocker) -> None:
        engine = _FakeEngine([_segment_output("我开始听。", finished=True), _segment_output("他说", finished=True)])
        _, text = _run(_connection(transcription=True, engine=engine), mocker)
        assert text == "我开始听。他说"

    def test_conversational_session_text_is_unchanged(self, mocker) -> None:
        engine = _FakeEngine([_segment_output("listening.", finished=True), _segment_output("To him.", finished=True)])
        _, text = _run(_connection(transcription=False, engine=engine), mocker)
        assert text == "listening.To him."


class _FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return [text]


@pytest.fixture
def fake_model_io(monkeypatch):
    processor = SimpleNamespace(feature_extractor=SimpleNamespace(sampling_rate=RATE), chat_template=None)
    monkeypatch.setattr(qwen3_omni, "cached_processor_from_config", lambda model_config: processor)
    monkeypatch.setattr(qwen3_omni, "cached_tokenizer_from_config", lambda model_config: _FakeTokenizer())


def _prompts(transcription: bool, seconds: float) -> list[dict]:
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, int(seconds * RATE)).astype(np.float32)
    step = int(0.2 * RATE)

    async def audio_stream():
        for start in range(0, len(audio), step):
            yield audio[start : start + step]

    async def collect():
        return [
            prompt
            async for prompt in Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio(
                audio_stream(), asyncio.Queue(), _FakeModelConfig(), transcription=transcription
            )
        ]

    prompts = asyncio.run(collect())
    assert sum(len(prompt["multi_modal_data"]["audio"]) for prompt in prompts) == len(audio)
    return prompts


@pytest.mark.usefixtures("fake_model_io")
class TestTranscriptionPrompts:
    def test_first_segment_opens_the_conversation_and_later_segments_continue_it(self) -> None:
        prompts = _prompts(transcription=True, seconds=12.0)

        token_ids = [prompt["prompt_token_ids"][0] for prompt in prompts]
        assert len(prompts) == 3
        assert token_ids[0].startswith("<|im_start|>user\n")
        assert all(ids.startswith("<|im_end|>\n<|im_start|>user\n") for ids in token_ids[1:])

    def test_segments_are_cut_inside_the_search_span(self) -> None:
        prompts = _prompts(transcription=True, seconds=12.0)

        lengths = [len(prompt["multi_modal_data"]["audio"]) for prompt in prompts[:-1]]
        assert all(4 * RATE <= length <= 5 * RATE for length in lengths)

    def test_conversational_segments_keep_the_fixed_cut_and_repeated_prompt(self) -> None:
        prompts = _prompts(transcription=False, seconds=12.0)

        lengths = [len(prompt["multi_modal_data"]["audio"]) for prompt in prompts]
        assert lengths == [5 * RATE, 5 * RATE, 2 * RATE]
        assert len({prompt["prompt_token_ids"][0] for prompt in prompts}) == 1
