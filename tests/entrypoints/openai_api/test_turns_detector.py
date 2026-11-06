# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.turns.detector import TurnPhase, TurnTracker
from vllm_omni.entrypoints.openai.turns.protocol import (
    CLIENT_EVENT,
    TurnContextUpdate,
    TurnEagerEnd,
    TurnEnded,
    TurnResumed,
    TurnSessionConfig,
    TurnSessionPatch,
    TurnStarted,
    TurnUpdate,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def paused_tracker(**config) -> TurnTracker:
    tracker = TurnTracker(config=TurnSessionConfig(**config))
    tracker.speech_started(1_000)
    tracker.speech_stopped(2_000)
    return tracker


def test_speech_onset_opens_one_turn():
    tracker = TurnTracker(config=TurnSessionConfig())
    events = tracker.speech_started(1_000)
    assert [type(e) for e in events] == [TurnStarted]
    assert events[0].audio_ms == 1_000
    assert tracker.speech_started(1_100) == []
    assert tracker.phase is TurnPhase.SPEAKING


def test_readings_fall_due_every_100_ms_of_silence():
    tracker = paused_tracker()
    assert tracker.advance(2_050) == (None, [])
    reading, _ = tracker.advance(2_128)
    assert reading is not None and reading.silence_ms == 128
    assert tracker.advance(2_160) == (None, [])
    reading, _ = tracker.advance(2_224)
    assert reading is not None and reading.silence_ms == 224


def test_reading_above_threshold_ends_turn_only_after_min_silence():
    tracker = paused_tracker(eot_threshold=0.6, min_silence_ms=200)
    early, _ = tracker.advance(2_100)
    assert [type(e) for e in tracker.reading(early, 0.9)] == [TurnUpdate]
    late, _ = tracker.advance(2_200)
    events = tracker.reading(late, 0.9)
    assert [type(e) for e in events] == [TurnUpdate, TurnEnded]
    assert events[1].reason == "model" and events[1].silence_ms == 200
    assert tracker.phase is TurnPhase.IDLE and tracker.turn_id is None


def test_low_reading_keeps_turn_open():
    tracker = paused_tracker()
    reading, _ = tracker.advance(2_300)
    assert [type(e) for e in tracker.reading(reading, 0.2)] == [TurnUpdate]
    assert tracker.phase is TurnPhase.PAUSED


def test_timeout_ends_turn_with_last_reading():
    tracker = paused_tracker(eot_timeout_ms=1_000)
    reading, _ = tracker.advance(2_100)
    tracker.reading(reading, 0.3)
    reading, events = tracker.advance(3_000)
    assert reading is None
    assert [type(e) for e in events] == [TurnEnded]
    assert events[0].reason == "timeout" and events[0].p_eot == 0.3


def test_eager_end_then_resumed_speech():
    tracker = paused_tracker(eager_eot_threshold=0.4, eot_threshold=0.9)
    reading, _ = tracker.advance(2_100)
    assert [type(e) for e in tracker.reading(reading, 0.5)] == [TurnUpdate, TurnEagerEnd]
    reading, _ = tracker.advance(2_200)
    assert [type(e) for e in tracker.reading(reading, 0.6)] == [TurnUpdate]
    assert [type(e) for e in tracker.speech_started(2_300)] == [TurnResumed]
    assert tracker.phase is TurnPhase.SPEAKING


def test_resumed_speech_without_eager_end_is_silent():
    tracker = paused_tracker()
    assert tracker.speech_started(2_300) == []


def test_reading_from_an_earlier_pause_is_discarded():
    tracker = paused_tracker()
    stale, _ = tracker.advance(2_100)
    tracker.speech_started(2_150)
    tracker.speech_stopped(2_500)
    assert tracker.reading(stale, 0.99) == []
    assert tracker.phase is TurnPhase.PAUSED


def test_patch_keeps_unset_fields_and_can_disable_eager():
    config = TurnSessionConfig(eager_eot_threshold=0.4, eot_threshold=0.7)
    patched = TurnSessionPatch.model_validate({"eager_eot_threshold": None}).apply(config)
    assert patched.eager_eot_threshold is None and patched.eot_threshold == 0.7


def test_patch_rejects_null_threshold():
    with pytest.raises(ValidationError):
        TurnSessionPatch.model_validate({"eot_threshold": None}).apply(TurnSessionConfig())


def test_client_events_parse_by_type():
    event = CLIENT_EVENT.validate_json('{"type": "context.update", "agent_text": "What is your phone number?"}')
    assert isinstance(event, TurnContextUpdate)
    with pytest.raises(ValidationError):
        CLIENT_EVENT.validate_json('{"type": "context.update", "text": "x"}')


def test_timeout_waits_while_an_onset_is_pending():
    tracker = paused_tracker(eot_timeout_ms=500)
    assert tracker.advance(2_600, speech_pending=True) == (None, [])
    assert tracker.phase is TurnPhase.PAUSED
    _, ended = tracker.advance(2_632)
    assert [e.reason for e in ended] == ["timeout"]
