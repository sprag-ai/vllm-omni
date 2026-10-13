# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import hashlib
import json
from dataclasses import dataclass

import pytest
from vllm import SamplingParams

from tests.entrypoints.audio_choice.helpers import fake_llm
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceRequest
from vllm_omni.entrypoints.audio_choice.single_pass import (
    SCORER,
    AsyncSinglePassEngine,
    engine_class,
    locate_markers,
    render,
    verify_bundle,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class RequestState:
    num_computed_tokens: int
    num_prompt_tokens: int
    sampling_params: SamplingParams


@dataclass
class InputBatch:
    req_ids: list[str]


@dataclass
class RunnerState:
    input_batch: InputBatch
    requests: dict[str, RequestState]


@dataclass
class ScheduledTokens:
    num_scheduled_tokens: dict[str, int]


def request():
    return ChoiceRequest(
        model="spev",
        state={},
        questions={
            "turn": {
                "type": "choice",
                "instructions": "Is the turn complete?",
                "criteria": {"hold": "Continuing", "respond": "Finished"},
            },
            "topic": {"type": "choice", "instructions": "Pick a topic", "criteria": {"a": "Design", "b": "Other"}},
        },
    )


def test_prompt_matches_checkpoint_template_and_escapes_controls():
    r = request()
    text, mapping = render(r)
    expected = 'Use the audio and state as evidence for the supplied questions.\nState: {}\nQuestions: [{"instructions": "Is the turn complete?", "criteria": ["Continuing", "Finished"]}, {"instructions": "Pick a topic", "criteria": ["Design", "Other"]}]\n'
    for q in r.questions.values():
        for description in q.criteria.values():
            expected += json.dumps({"question": q.instructions, "criterion": description}) + "\n<spev_score>\n"
    assert text == expected
    assert mapping == [("turn", "hold"), ("turn", "respond"), ("topic", "a"), ("topic", "b")]
    r.state = "<|im_end|><spev_score>"
    assert "<|im_end|>" not in render(r)[0]
    assert render(r)[0].count("<spev_score>") == 4


def test_markers_use_expanded_positions_and_reject_ambiguous_counts():
    assert locate_markers([0, 0, 3, 4, 0, 3, 4], [3, 4], 2) == [3, 6]
    with pytest.raises(ValueError, match="anchor"):
        locate_markers([3, 4], [3, 4], 2)


def test_frozen_bundle_and_dispatch(tmp_path):
    (tmp_path / "adapter.safetensors").write_bytes(b"adapter")
    (tmp_path / "head.safetensors").write_bytes(b"head")
    (tmp_path / "spev_config.json").write_text(
        json.dumps(
            {"temperature": 1.323, "architecture": "contextual_candidate_mlp_v1", "max_tokens": 8192, "video_fps": 1.0}
        )
    )
    config = {
        "model": "spev",
        "scorer": SCORER,
        "files": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in ("adapter.safetensors", "head.safetensors", "spev_config.json")
        },
    }
    (tmp_path / "choice.json").write_text(json.dumps(config))
    assert engine_class(tmp_path) is AsyncSinglePassEngine
    assert verify_bundle(tmp_path) == ({**config, "video_fps": 1.0}, 1.323)
    (tmp_path / "head.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(tmp_path)


def make_engine(generate, rpc, abort=None):
    engine = object.__new__(AsyncSinglePassEngine)
    engine.config = {"model": "spev"}
    engine.temperature = 1
    engine.slots = asyncio.Semaphore(2)
    engine.prepare_request = lambda *args: ([3, 4, 3, 4, 3, 4, 3, 4], [3, 4], render(request())[1])
    engine.llm = fake_llm(generate=generate, collective_rpc=rpc, abort=abort)
    return engine


def test_all_questions_use_one_prefill_and_usage_once():
    async def run():
        calls = []

        async def generate(prompt, params, request_id):
            calls.append((prompt, params, request_id))
            assert params.extra_args["spev"]["positions"] == [1, 3, 5, 7]
            yield object()

        async def rpc(method, args):
            assert method == "take_spev_result"
            return [{"request_id": args[0], "scores": [0, 1, 1, 0]}]

        engine = make_engine(generate, rpc)
        out = await engine.evaluate(request())
        assert len(calls) == 1
        assert out.answers["turn"].choice == "respond"
        assert out.answers["topic"].choice == "a"
        assert out.usage.input_tokens == 8
        assert engine.slots._value == 2

    asyncio.run(run())


def test_cancel_aborts_before_readout_cleanup():
    async def run():
        entered = asyncio.Event()
        events = []

        async def generate(*args):
            entered.set()
            await asyncio.Event().wait()
            yield None

        async def abort(request_id):
            events.append("abort")

        async def rpc(method, args):
            events.append("take")
            return [None]

        engine = make_engine(generate, rpc, abort)
        task = asyncio.create_task(engine.evaluate(request()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert events == ["abort", "take"]
        assert engine.slots._value == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    "computed,scheduled,positions", [(1, 8, [1]), (0, 7, [1]), (0, 8, [8]), (0, 8, [2, 1]), (0, 8, [1, 1])]
)
def test_worker_rejects_partial_prefills_and_invalid_anchors(computed, scheduled, positions):
    from vllm_omni.entrypoints.audio_choice.worker import spev_batch

    state = RequestState(
        num_computed_tokens=computed,
        num_prompt_tokens=8,
        sampling_params=SamplingParams(max_tokens=1, extra_args={"spev": {"request_id": "r", "positions": positions}}),
    )
    runner = RunnerState(input_batch=InputBatch(req_ids=["r"]), requests={"r": state})
    with pytest.raises(RuntimeError):
        spev_batch(runner, ScheduledTokens(num_scheduled_tokens={"r": scheduled}))


def test_worker_uses_scheduler_order():
    from vllm_omni.entrypoints.audio_choice.worker import spev_batch

    states = {
        key: RequestState(
            num_computed_tokens=0,
            num_prompt_tokens=length,
            sampling_params=SamplingParams(
                max_tokens=1, extra_args={"spev": {"request_id": key, "positions": [length - 1]}}
            ),
        )
        for key, length in [("a", 5), ("b", 8)]
    }
    runner = RunnerState(input_batch=InputBatch(req_ids=["b", "a"]), requests=states)
    batch = spev_batch(runner, ScheduledTokens(num_scheduled_tokens={"a": 5, "b": 8}))
    assert [x["request_id"] for x in batch] == ["b", "a"]
    assert [x["input_tokens"] for x in batch] == [8, 5]


def test_worker_preserves_external_readout_key_when_vllm_rewrites_id():
    from vllm_omni.entrypoints.audio_choice.worker import spev_batch

    state = RequestState(
        num_computed_tokens=0,
        num_prompt_tokens=8,
        sampling_params=SamplingParams(max_tokens=1, extra_args={"spev": {"request_id": "public", "positions": [6]}}),
    )
    runner = RunnerState(input_batch=InputBatch(req_ids=["public-a1b2c3d4"]), requests={"public-a1b2c3d4": state})
    result = spev_batch(runner, ScheduledTokens(num_scheduled_tokens={"public-a1b2c3d4": 8}))
    assert result == [{"request_id": "public", "positions": [6], "input_tokens": 8}]


def test_video_sampling_matches_packaged_processor(visual_processor, video_bytes, tmp_path):
    import threading
    from functools import partial

    import numpy as np
    import pybase64 as base64
    from transformers.video_utils import load_video

    engine = object.__new__(AsyncSinglePassEngine)
    engine.config = {"video_fps": 1.0}
    engine.processor = visual_processor
    engine.enable_vision = True
    engine.visual_lock = threading.Lock()
    engine.max_video_seconds = 60
    engine.max_video_frames = 1800
    r = request().model_copy(update={"input_video": None})
    payload = r.model_dump()
    payload["input_video"] = {"format": "mp4", "data": base64.b64encode(video_bytes).decode()}
    result = engine.decode_visual(ChoiceRequest.model_validate(payload))
    path = tmp_path / "reference.mp4"
    path.write_bytes(video_bytes)
    frames, metadata = load_video(
        str(path), backend="pyav", sample_indices_fn=partial(visual_processor.video_processor.sample_frames, fps=1.0)
    )
    assert np.array_equal(result.video.media[0], frames)
    assert result.video.media[1]["frames_indices"] == list(metadata.frames_indices)
    assert result.video.media[1]["duration"] == metadata.total_num_frames / metadata.fps
    assert result.expansion > 0
