# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from vllm_omni.engine.duplex.vad import SileroStreamingVAD
from vllm_omni.entrypoints.openai.turns.connection import VAD_CONFIG, TurnsConnection
from vllm_omni.entrypoints.openai.turns.scorer import TurnPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PROMPT = TurnPrompt(
    question="Done?\nA. hold\nB. complete", context_prefix="State: ", window_seconds=8.0, max_context_chars=300
)


class FakeWebSocket:
    def __init__(self, frames: list[dict]) -> None:
        self.inbox = list(frames)
        self.sent: list[dict] = []
        self.closed = False

    async def receive(self) -> dict:
        await asyncio.sleep(0)
        if not self.inbox:
            return {"type": "websocket.disconnect"}
        return self.inbox.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def close(self) -> None:
        self.closed = True


class FakeScorer:
    def __init__(self, p_eot: float) -> None:
        self.model = "turns"
        self.prompt = PROMPT
        self.p_eot = p_eot
        self.calls: list[tuple[int, str]] = []

    async def score(self, samples: np.ndarray, context: str, request_id: str) -> float:
        self.calls.append((samples.size, context))
        return self.p_eot


def pcm(seconds: float, speech: bool) -> bytes:
    t = np.arange(int(seconds * 16_000)) / 16_000
    wave = 0.3 * np.sin(2 * np.pi * 220 * t) if speech else np.zeros_like(t)
    return (wave * 32767).astype("<i2").tobytes()


def audio_frames(*segments: tuple[float, bool]) -> list[dict]:
    frames = []
    for seconds, speech in segments:
        data = pcm(seconds, speech)
        frames += [{"type": "websocket.receive", "bytes": data[i : i + 640]} for i in range(0, len(data), 640)]
    return frames


def energy_vad() -> SileroStreamingVAD:
    return SileroStreamingVAD(VAD_CONFIG, frame_scorer=lambda frame: float(np.abs(frame).mean() > 0.01))


async def run(frames: list[dict], p_eot: float) -> tuple[FakeWebSocket, FakeScorer]:
    websocket, scorer = FakeWebSocket(frames), FakeScorer(p_eot)
    connection = TurnsConnection(websocket, scorer, energy_vad())
    await connection.run()
    if connection.inflight is not None:
        await asyncio.gather(connection.inflight, return_exceptions=True)
    return websocket, scorer


def types(websocket: FakeWebSocket) -> list[str]:
    return [event["type"] for event in websocket.sent]


def test_confident_pause_ends_the_turn_with_context():
    context = {"type": "websocket.receive", "text": json.dumps({"type": "context.update", "agent_text": "Your name?"})}
    close = {"type": "websocket.receive", "text": json.dumps({"type": "session.close"})}
    frames = [context] + audio_frames((1.0, True), (1.0, False)) + [close]
    websocket, scorer = asyncio.run(run(frames, 0.9))
    kinds = types(websocket)
    assert kinds[0] == "session.created" and kinds[1] == "turn.started"
    assert "turn.ended" in kinds
    ended = next(e for e in websocket.sent if e["type"] == "turn.ended")
    assert ended["reason"] == "model" and ended["silence_ms"] >= 200
    assert scorer.calls and all(context == "Your name?" for _, context in scorer.calls)
    assert websocket.closed


def test_unconfident_pause_times_out():
    update = {
        "type": "websocket.receive",
        "text": json.dumps({"type": "session.update", "session": {"eot_timeout_ms": 500}}),
    }
    close = {"type": "websocket.receive", "text": json.dumps({"type": "session.close"})}
    frames = [update] + audio_frames((1.0, True), (1.0, False)) + [close]
    websocket, _ = asyncio.run(run(frames, 0.1))
    assert "session.updated" in types(websocket)
    ended = next(e for e in websocket.sent if e["type"] == "turn.ended")
    assert ended["reason"] == "timeout" and ended["silence_ms"] >= 500


def test_odd_byte_audio_is_rejected_without_closing():
    frames = [{"type": "websocket.receive", "bytes": b"\x00\x01\x02"}]
    websocket, _ = asyncio.run(run(frames, 0.5))
    assert types(websocket) == ["session.created", "error"]
    assert websocket.sent[1]["error"]["code"] == "invalid_audio"


def test_unknown_event_is_rejected():
    frames = [{"type": "websocket.receive", "text": json.dumps({"type": "input_audio_buffer.append"})}]
    websocket, _ = asyncio.run(run(frames, 0.5))
    assert websocket.sent[1]["error"]["code"] == "invalid_event"


def test_context_prefix_renders_like_decisions():
    assert PROMPT.text("") == PROMPT.question
    assert PROMPT.text('say "<hi>"') == 'State: "say \\"\\u003chi\\u003e\\""\n' + PROMPT.question
