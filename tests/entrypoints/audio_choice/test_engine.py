# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.audio_choice.contract import render_question
from vllm_omni.entrypoints.audio_choice.engine import AsyncChoiceEngine, verify_bundle
from vllm_omni.entrypoints.audio_choice.protocol import ChoiceRequest


def request(question_id="routing"):
    return ChoiceRequest.model_validate(
        {
            "model": "spev",
            "state": {"text": "invoice"},
            "questions": {
                question_id: {
                    "type": "choice",
                    "instructions": "Select a department",
                    "criteria": {"billing": None, "technical": None},
                }
            },
        }
    )


def test_question_ids_never_reach_model_and_named_scores_supply_probabilities():
    async def run():
        engine = object.__new__(AsyncChoiceEngine)
        engine.config = {"model": "resolved-revision"}
        engine.temperature = 2
        prompts = []

        def prepare(q, state, wave):
            return render_question(q, state), {key: key for key in q.criteria}

        engine.prepare = prepare

        async def score(rendered, key, wave):
            prompts.append(rendered)
            return ({"billing": 0.0, "technical": -2.0}[key], 123)

        engine.score = score
        one = await engine.evaluate(request("PRIVATE-CALLER-ID"))
        two = await engine.evaluate(request("changed-id"))
        assert one.answers["PRIVATE-CALLER-ID"] == two.answers["changed-id"]
        assert one.answers["PRIVATE-CALLER-ID"].choice == "billing"
        assert one.answers["PRIVATE-CALLER-ID"].probabilities["billing"] == pytest.approx(0.7310585786300049)
        assert not any("PRIVATE-CALLER-ID" in p or "changed-id" in p for p in prompts)
        assert one.model == "resolved-revision" and one.usage.input_tokens == 123

    asyncio.run(run())


def test_failed_candidate_cancels_and_drains_other_candidates():
    async def run():
        engine = object.__new__(AsyncChoiceEngine)
        engine.config = {"model": "spev"}
        engine.temperature = 1
        engine.prepare = lambda q, *a: ("rendered", {key: key for key in q.criteria})
        entered = asyncio.Event()
        cleaned = asyncio.Event()

        async def score(rendered, key, wave):
            if key == "billing":
                await entered.wait()
                raise RuntimeError("failed scoring")
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        engine.score = score
        with pytest.raises(RuntimeError, match="failed scoring"):
            await engine.evaluate(request())
        assert cleaned.is_set()

    asyncio.run(run())


def test_interrupted_scoring_aborts_engine_request(monkeypatch):
    async def run():
        engine = object.__new__(AsyncChoiceEngine)
        engine.slots = asyncio.Semaphore(1)
        engine.tokenizer = SimpleNamespace(
            encode=lambda text, **kw: list(text.encode()), convert_tokens_to_ids=lambda _: 3
        )
        entered = asyncio.Event()
        aborted = []

        async def generate(prompt, params, request_id):
            entered.set()
            await asyncio.Event().wait()
            yield None

        async def abort(request_id):
            aborted.append(request_id)

        engine.llm = SimpleNamespace(generate=generate, abort=abort)
        task = asyncio.create_task(engine.score([10], [11, 3], None))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(aborted) == 1 and aborted[0].startswith("choice-")
        assert engine.slots._value == 1

    asyncio.run(run())


def bundle(tmp_path):
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter/adapter_model.safetensors").write_bytes(b"test weights")
    (tmp_path / "adapter/adapter_config.json").write_text("{}")
    calibration = {
        "adapter_sha256": hashlib.sha256(b"test weights").hexdigest(),
        "temperature": 1.0,
        "scorer": "vllm-full-sequence-choice-v1",
    }
    (tmp_path / "calibration.json").write_text(json.dumps(calibration))
    names = ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "calibration.json"]
    (tmp_path / "choice.json").write_text(
        json.dumps(
            {
                "model": "spev-revision",
                "files": {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in names},
            }
        )
    )
    return calibration


def test_bundle_requires_matching_frozen_adapter_calibration_and_scorer(tmp_path):
    bundle(tmp_path)
    assert verify_bundle(tmp_path)[1] == 1
    (tmp_path / "adapter/adapter_model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(tmp_path)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, **kwargs):
        self.calls.append(text)
        return list(text.encode())

    def convert_tokens_to_ids(self, text):
        return {"<|audio_pad|>": 90001, "<|im_end|>": 90002}[text]


def preparing_engine():
    engine = object.__new__(AsyncChoiceEngine)
    engine.prepare_lock = threading.Lock()
    engine.tokenizer = Tokenizer()
    engine.prompt = lambda *a: "shared prompt"
    return engine


def test_prefix_is_tokenized_once_and_candidates_are_ids():
    engine = preparing_engine()
    q = next(iter(request().questions.values()))
    prefix, candidates = engine.prepare(q, "state", None)
    assert engine.tokenizer.calls.count("shared prompt") == 1
    assert len(engine.tokenizer.calls) == 1 + len(q.criteria)
    assert prefix == list(b"shared prompt")
    assert all(ids[-1] == 90002 for ids in candidates.values())


def test_overlong_prefix_fails_before_candidate_tokenization():
    engine = preparing_engine()
    engine.prompt = lambda *a: "x" * 8192
    with pytest.raises(ValueError, match="8192-token"):
        engine.prepare(next(iter(request().questions.values())), "state", None)
    assert len(engine.tokenizer.calls) == 1


def test_candidate_length_and_discarded_token_are_in_context_budget():
    engine = preparing_engine()
    engine.prompt = lambda *a: "x" * 8180
    with pytest.raises(ValueError, match="prompt and target"):
        engine.prepare(next(iter(request().questions.values())), "state", None)


def test_audio_expansion_is_in_context_budget():
    engine = preparing_engine()
    engine.tokenizer.encode = lambda text, **kw: [90001] + [10] * 8100 if text == "shared prompt" else [11]
    engine.audio_tokens = lambda wave: 100
    with pytest.raises(ValueError, match="prompt and target"):
        engine.prepare(next(iter(request().questions.values())), "state", [0.0])


def test_prepare_runs_off_event_loop():
    async def run():
        engine = preparing_engine()
        engine.config = {"model": "test"}
        engine.temperature = 1
        entered = threading.Event()
        release = threading.Event()
        loop_thread = threading.get_ident()
        original = engine.prepare

        def slow(*a):
            assert threading.get_ident() != loop_thread
            entered.set()
            assert release.wait(5)
            return original(*a)

        engine.prepare = slow

        async def score(*a):
            return 0.0, 12

        engine.score = score
        task = asyncio.create_task(engine.evaluate(request()))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=1)
        finally:
            release.set()
        await task

    asyncio.run(run())


def test_score_submits_token_ids_without_retokenization():
    async def run():
        engine = preparing_engine()
        engine.slots = asyncio.Semaphore(1)
        seen = []

        async def generate(prompt, params, request_id):
            seen.append(prompt)
            yield SimpleNamespace(
                prompt_token_ids=[10, 11, 90002],
                prompt_logprobs=[None, {11: SimpleNamespace(logprob=-0.2)}, {90002: SimpleNamespace(logprob=-0.3)}],
            )

        engine.llm = SimpleNamespace(generate=generate)
        value = await engine.score([10], [11, 90002], None)
        assert value == (-0.5, 1)
        assert seen == [{"prompt_token_ids": [10, 11, 90002]}]
        assert engine.tokenizer.calls == []

    asyncio.run(run())
