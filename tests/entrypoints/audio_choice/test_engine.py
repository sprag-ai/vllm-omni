# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
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

        def prompt(q, state, has_audio):
            return render_question(q, state)

        engine.prompt = prompt

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
        engine.prompt = lambda *a: "rendered"
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
        task = asyncio.create_task(engine.score("prompt", "billing", None))
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
