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

        def prepare(q, state, wave, visual=None):
            return render_question(q, state), {key: key for key in q.criteria}

        engine.prepare = prepare

        async def score(rendered, key, wave, visual=None):
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

        async def score(rendered, key, wave, visual=None):
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


@pytest.mark.parametrize("modality", ["image", "video"])
def test_visual_expansion_is_in_context_budget(modality):
    engine = preparing_engine()
    tokens = {"<|audio_pad|>": 90001, "<|im_end|>": 90002, "<|image_pad|>": 90003, "<|video_pad|>": 90004}
    engine.tokenizer.convert_tokens_to_ids = tokens.__getitem__
    engine.tokenizer.encode = lambda text, **kw: (
        [tokens[f"<|{modality}_pad|>"]] + [10] * 8100 if text == "shared prompt" else [11]
    )
    visual = SimpleNamespace(
        image=object() if modality == "image" else None,
        video=object() if modality == "video" else None,
        expansion=100,
    )
    engine.processor = object()
    with pytest.raises(ValueError, match="prompt and target"):
        engine.prepare(next(iter(request().questions.values())), "state", None, visual)


def test_visual_payload_reaches_async_llm():
    from vllm_omni.entrypoints.audio_choice.media import VisualEvidence

    async def run():
        engine = preparing_engine()
        engine.slots = asyncio.Semaphore(1)
        seen = []
        image, video, wave = object(), object(), [0.0]

        async def generate(prompt, params, request_id):
            seen.append(prompt)
            yield SimpleNamespace(
                prompt_token_ids=[10, 11], prompt_logprobs=[None, {11: SimpleNamespace(logprob=-0.5)}]
            )

        engine.llm = SimpleNamespace(generate=generate)
        assert await engine.score([10], [11], wave, VisualEvidence(image, video)) == (-0.5, 1)
        assert seen[0]["multi_modal_data"] == {"image": [image], "video": [video], "audio": (wave, 16000)}
        assert set(seen[0]["multi_modal_uuids"]) == {"image", "video", "audio"}

    asyncio.run(run())


def test_new_primitives_and_visual_evidence_do_not_inherit_choice_temperature():
    async def run():
        engine = object.__new__(AsyncChoiceEngine)
        engine.config = {"model": "test"}
        engine.temperature = 2
        engine.prepare = lambda question, *args: ([], {key: [i] for i, key in enumerate(question.criteria)})

        async def score(prefix, ids, *args):
            return -2.0 * ids[0], 12

        engine.score = score
        ordinary = await engine.evaluate(request())
        visual = await engine.evaluate(request(), visual=object())
        assert ordinary.answers["routing"].probabilities["billing"] == pytest.approx(0.7310585786)
        assert visual.answers["routing"].probabilities["billing"] == pytest.approx(0.8807970780)
        mixed = ChoiceRequest(
            model="spev",
            state={},
            questions={
                "n": {"type": "noul", "instructions": "True?"},
                "s": {"type": "score", "instructions": "Level?", "criteria": ["low", "high"]},
            },
        )
        out = await engine.evaluate(mixed)
        assert out.answers["n"].noul == pytest.approx(0.8807970780)
        assert out.answers["s"].score == pytest.approx(0.1192029220)

    asyncio.run(run())


@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("count", [0, 2])
def test_visual_placeholder_count_mismatch(modality, count):
    engine = preparing_engine()
    tokens = {"<|audio_pad|>": 90001, "<|im_end|>": 90002, "<|image_pad|>": 90003, "<|video_pad|>": 90004}
    engine.tokenizer.convert_tokens_to_ids = tokens.__getitem__
    engine.tokenizer.encode = lambda *a, **kw: [tokens[f"<|{modality}_pad|>"]] * count
    visual = SimpleNamespace(
        image=object() if modality == "image" else None, video=object() if modality == "video" else None, expansion=1
    )
    with pytest.raises(ValueError, match=f"{modality} placeholder count mismatch"):
        engine.prepare(next(iter(request().questions.values())), "state", None, visual)


@pytest.mark.parametrize("error_type", ["client", "generate"])
def test_native_input_errors_are_redacted_and_aborted(error_type):
    from vllm.exceptions import VLLMClientError
    from vllm.v1.engine.exceptions import EngineGenerateError

    async def run():
        engine = preparing_engine()
        engine.slots = asyncio.Semaphore(1)
        aborted = []

        async def generate(*a):
            raise (VLLMClientError if error_type == "client" else EngineGenerateError)("PRIVATE IMAGE DATA")
            yield

        async def abort(request_id):
            aborted.append(request_id)

        engine.llm = SimpleNamespace(generate=generate, abort=abort, errored=False)
        with pytest.raises(ValueError, match="^Invalid or unsupported model input$"):
            await engine.score([1], [2], None)
        assert len(aborted) == 1
        assert engine.slots._value == 1

    asyncio.run(run())


def test_vision_is_disabled_before_decoding():
    engine = object.__new__(AsyncChoiceEngine)
    engine.enable_vision = False
    with pytest.raises(ValueError, match="choice-enable-vision"):
        engine.decode_visual(object())


@pytest.mark.parametrize("enabled", [False, True])
def test_engine_vision_limits_are_opt_in(enabled, tmp_path, monkeypatch):
    import transformers
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_omni.entrypoints.audio_choice import engine as module

    captured = []
    monkeypatch.setattr(module, "verify_bundle", lambda root: ({"model": "test"}, 1))
    monkeypatch.setattr(
        transformers.Qwen3OmniMoeProcessor, "from_pretrained", lambda *a, **kw: SimpleNamespace(tokenizer=object())
    )
    monkeypatch.setattr(AsyncLLM, "from_engine_args", lambda args: captured.append(args))
    kwargs = {"enable_vision": True} if enabled else {}
    AsyncChoiceEngine(str(tmp_path), tmp_path, **kwargs)
    assert captured[0].limit_mm_per_prompt == {"audio": 1, "image": int(enabled), "video": int(enabled)}


def test_visual_preflight_does_not_hold_text_preparation_lock(visual_processor, monkeypatch):
    import base64
    import io

    from PIL import Image

    from vllm_omni.entrypoints.audio_choice.media import VisualEvidence

    async def run():
        engine = preparing_engine()
        engine.processor = visual_processor
        engine.visual_lock = threading.Lock()
        engine.enable_vision = True
        engine.max_video_seconds, engine.max_video_frames = 60, 1800
        data = io.BytesIO()
        Image.new("RGB", (28, 28)).save(data, format="PNG")
        req = request().model_copy(
            update={"input_image": SimpleNamespace(data=base64.b64encode(data.getvalue()).decode())}
        )
        entered, release = threading.Event(), threading.Event()
        original = VisualEvidence.token_expansion

        def delayed(self, processor):
            entered.set()
            assert release.wait(5)
            return original(self, processor)

        monkeypatch.setattr(VisualEvidence, "token_expansion", delayed)
        task = asyncio.create_task(asyncio.to_thread(engine.decode_visual, req))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            prefix, _ = await asyncio.wait_for(
                asyncio.to_thread(engine.prepare, next(iter(request().questions.values())), "plain text", None),
                timeout=1,
            )
            assert prefix
        finally:
            release.set()
        await task

    asyncio.run(run())
