# SPDX-License-Identifier: Apache-2.0
import asyncio
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.entrypoints.audio_decision.worker import decision_batch
from vllm_omni.entrypoints.openai.serving_decision import DecisionServing
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_decision import Qwen3OmniBatchedDecisionThinker
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_decision_base import Qwen3OmniDecisionThinker


def fake_model(requests):
    model = Qwen3OmniBatchedDecisionThinker.__new__(Qwen3OmniBatchedDecisionThinker)
    nn.Module.__init__(model)
    calls = []

    def layer(p, h, r):
        calls.append(1)
        return h, None

    model.language_model = SimpleNamespace(model=SimpleNamespace(layers=[layer] * 3, norm=lambda h, r: (h, None)))
    model.decision_spec = {
        "depth": 1,
        "head_temperature": 1,
        "full_temperature": 2,
        "actions": ["keep_listening", "respond", "insufficient_evidence"],
    }
    model.decision_head = {
        "mean": torch.zeros(2, dtype=torch.float64),
        "scale": torch.ones(2, dtype=torch.float64),
        "weight": torch.tensor([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]], dtype=torch.float64),
        "bias": torch.zeros(3, dtype=torch.float64),
    }
    model.decision_ids = [1, 2, 3]
    model.config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=4))
    model.batch_provider = lambda: requests
    model.batch_audio_items = len(requests)
    model.decision_encoder_calls = 1
    model.completed_decisions = {}
    model._clear_deepstack_input_embeds = lambda n: None
    return model, calls


def test_mixed_modes_take_correct_last_row_and_keep_head_scores(monkeypatch):
    requests = [
        {"request_id": "head", "input_tokens": 2, "mode": "auto", "threshold": 0.95},
        {"request_id": "raw", "input_tokens": 3, "mode": "raw", "threshold": 0.8},
        {"request_id": "embed", "input_tokens": 1, "mode": "embedding", "threshold": 0.95},
    ]
    model, calls = fake_model(requests)
    hidden = torch.tensor([[99.0, 99.0], [10.0, 2.0], [99.0, 99.0], [99.0, 99.0], [0.0, 3.0], [-2.0, 4.0]])
    monkeypatch.setattr(
        Qwen3OmniDecisionThinker,
        "compute_logits",
        lambda self, h: torch.tensor([[0.0, 1.0, 5.0, 2.0]]).repeat(len(h), 1),
    )
    output = model.forward(None, torch.arange(6), inputs_embeds=hidden)
    logits = model.compute_logits(output[[1, 4, 5]])
    assert len(calls) == 3
    assert logits.argmax(1).tolist() == [1, 2, 3]
    first = model.take_result("head")
    assert first["decoder_depth"] == 1 and first["batch_decoder_depth"] == 3
    assert not first["used_full_decoder"] and first["input_tokens"] == 2
    assert torch.allclose(
        torch.tensor(list(first["probabilities"].values())), torch.softmax(torch.tensor([10.0, 0.0, -10.0]), 0)
    )
    raw = model.take_result("raw")
    assert raw["used_full_decoder"] and raw["input_tokens"] == 3
    assert torch.allclose(
        torch.tensor(list(raw["probabilities"].values())), torch.softmax(torch.tensor([1.0, 5.0, 2.0]), 0)
    )
    assert model.take_result("embed")["embedding"] == [-2.0, 4.0]
    assert model.take_result("head") is None


def test_all_accepted_batch_stops_at_head():
    requests = [{"request_id": str(i), "input_tokens": 1, "mode": "auto", "threshold": 0.95} for i in range(3)]
    model, calls = fake_model(requests)
    h = model.forward(None, torch.arange(3), inputs_embeds=torch.tensor([[10.0, 0.0], [-10.0, 1.0], [10.0, 2.0]]))
    model.compute_logits(h)
    assert len(calls) == 1
    assert [model.take_result(str(i))["action"] for i in range(3)] == [
        "keep_listening",
        "insufficient_evidence",
        "keep_listening",
    ]


def test_scheduler_order_and_complete_prefill_invariants():
    def state(key, tokens):
        return SimpleNamespace(
            num_computed_tokens=0,
            num_prompt_tokens=tokens,
            sampling_params=SimpleNamespace(
                max_tokens=1, extra_args={"audio_decision": {"request_id": key, "mode": "auto", "threshold": 0.95}}
            ),
        )

    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["b", "a"]), requests={"a": state("client-a", 2), "b": state("client-b", 4)}
    )
    schedule = SimpleNamespace(num_scheduled_tokens={"a": 2, "b": 4}, scheduled_encoder_inputs={"a": [0], "b": [0]})
    assert [(r["request_id"], r["input_tokens"]) for r in decision_batch(runner, schedule)] == [
        ("client-b", 4),
        ("client-a", 2),
    ]
    runner.requests["b"].num_computed_tokens = 1
    with pytest.raises(RuntimeError, match="complete prefill"):
        decision_batch(runner, schedule)


def test_async_serving_admits_concurrent_jobs_and_drains(monkeypatch):
    from vllm_omni.entrypoints.openai import serving_decision

    monkeypatch.setattr(serving_decision, "parse_audio", lambda body: body["wave"])

    async def run():
        entered = []
        release = asyncio.Event()

        class Engine:
            is_async = True

            async def decide(self, wave, threshold, mode):
                entered.append(wave)
                await release.wait()
                return {"wave": wave}

            def shutdown(self):
                pass

        handler = DecisionServing(Engine(), "test", 2)
        a = asyncio.create_task(handler.infer({"wave": 1}, "auto", 0.95))
        b = asyncio.create_task(handler.infer({"wave": 2}, "raw", 0.8))
        for _ in range(100):
            if len(entered) == 2:
                break
            await asyncio.sleep(0.01)
        assert sorted(entered) == [1, 2]
        assert (await handler.infer({"wave": 3}, "auto", 0.95)).error.code == 429
        a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a
        assert handler.pending == 2
        release.set()
        assert await b == {"wave": 2}
        await handler.close()
        assert handler.pending == 0 and handler.closed

    asyncio.run(run())


def test_async_engine_constructs_real_versioned_engine_args(monkeypatch, tmp_path):
    from transformers import Qwen3OmniMoeProcessor
    from vllm.v1.engine.async_llm import AsyncLLM

    from vllm_omni.entrypoints.audio_decision import async_engine

    captured = []
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(async_engine, "verify_bundle", lambda root: {"prompt": "task"})
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text, **kw: [ord(c) for c in text]),
        apply_chat_template=lambda *a, **kw: "",
    )
    monkeypatch.setattr(Qwen3OmniMoeProcessor, "from_pretrained", lambda *a, **kw: processor)
    monkeypatch.setattr(AsyncLLM, "from_engine_args", lambda args: captured.append(args))
    async_engine.AsyncDecisionEngine(str(tmp_path), tmp_path, max_num_seqs=4, max_num_batched_tokens=4096)
    assert captured[0].max_num_seqs == 4
    assert captured[0].max_num_batched_tokens == 4096
    assert captured[0].worker_cls.endswith("DecisionWorker")


def test_async_engine_rejects_batch_variant_kernels(monkeypatch):
    from vllm_omni.entrypoints.audio_decision.async_engine import AsyncDecisionEngine

    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
    with pytest.raises(ValueError, match="VLLM_BATCH_INVARIANT=1"):
        AsyncDecisionEngine("unused", "unused")


def test_embedding_accepts_batch_encoder_accounting():
    import base64
    import io

    import numpy as np
    import soundfile as sf
    from fastapi.testclient import TestClient

    from tests.entrypoints.audio_decision.test_openai import Engine, args
    from vllm_omni.entrypoints.openai.serving_decision import build_decision_app

    class BatchEngine(Engine):
        def decide(self, *a, **kw):
            result = super().decide(*a, **kw)
            result.pop("audio_encoder_calls")
            result.update(batch_size=3, batch_audio_encoder_calls=1, batch_audio_items=3)
            return result

    f = io.BytesIO()
    sf.write(f, np.zeros(1600), 16000, format="WAV")
    with TestClient(build_decision_app(args(), BatchEngine())) as client:
        r = client.post(
            "/v1/embeddings",
            json={
                "model": "test-model",
                "input": "audio_turn_decision",
                "input_audio": {"format": "wav", "data": base64.b64encode(f.getvalue()).decode()},
            },
            headers={"Authorization": "Bearer test-secret"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["decision"]["batch_size"] == 3
        assert "probabilities" not in r.json()["decision"]


def test_fresh_engine_process_registers_decision_model():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from vllm.plugins import load_general_plugins; load_general_plugins(); "
            "from vllm.model_executor.models import ModelRegistry; "
            'assert "Qwen3OmniBatchedDecisionThinker" in ModelRegistry.get_supported_archs()',
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_disconnected_async_failure_still_marks_engine_unhealthy(monkeypatch):
    from vllm_omni.entrypoints.openai import serving_decision

    monkeypatch.setattr(serving_decision, "parse_audio", lambda body: None)

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()

        class Engine:
            is_async = True

            async def decide(self, *a, **kw):
                entered.set()
                await release.wait()
                raise RuntimeError("worker failed")

            def shutdown(self):
                pass

        handler = DecisionServing(Engine(), "test", 1)
        request = asyncio.create_task(handler.infer({}, "auto", 0.95))
        await entered.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release.set()
        await handler.close()
        assert handler.errored and handler.pending == 0

    asyncio.run(run())


def test_upstream_executor_shutdown_drains_async_jobs(monkeypatch):
    from vllm_omni.entrypoints.openai import serving_decision

    monkeypatch.setattr(serving_decision, "parse_audio", lambda body: None)

    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        stopped = []

        class Engine:
            is_async = True

            async def decide(self, *a, **kw):
                entered.set()
                await release.wait()
                assert not stopped
                return {"finished": True}

            def shutdown(self):
                stopped.append(True)

        handler = DecisionServing(Engine(), "test", 1)
        request = asyncio.create_task(handler.infer({}, "auto", 0.95))
        await entered.wait()
        shutdown = asyncio.create_task(asyncio.to_thread(handler.shutdown, timeout=2))
        for _ in range(100):
            if handler.closed:
                break
            await asyncio.sleep(0.01)
        assert handler.closed and not stopped
        assert (await handler.infer({}, "auto", 0.95)).error.code == 503
        release.set()
        assert await request == {"finished": True}
        await shutdown
        await handler.close()
        assert stopped == [True] and handler.pending == 0

    asyncio.run(run())


def test_audio_encoder_mixed_lengths_preserve_standalone_padding(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3OmniMoeAudioEncoder

    class Linear(nn.Linear):
        def forward(self, x):
            return super().forward(x), None

    # Real biased convolutions/GELU reproduce the boundary effect without a
    # GPU, model checkpoint, or vLLM tensor-parallel process group.
    torch.manual_seed(33)
    tower = Qwen3OmniMoeAudioEncoder.__new__(Qwen3OmniMoeAudioEncoder)
    nn.Module.__init__(tower)
    tower.n_window = 16
    tower.n_window_infer = 64
    tower.conv_chunksize = 500
    tower.conv2d1 = nn.Conv2d(1, 4, 3, 2, padding=1)
    tower.conv2d2 = nn.Conv2d(4, 4, 3, 2, padding=1)
    tower.conv2d3 = nn.Conv2d(4, 4, 3, 2, padding=1)
    tower.conv_out = Linear(4, 8)
    tower.positional_embedding = SimpleNamespace(positional_embedding=torch.zeros(8, 8))
    tower.layers = nn.ModuleList()
    tower.ln_post = nn.Identity()
    tower.proj1 = Linear(8, 8)
    tower.act = nn.GELU()
    tower.proj2 = Linear(8, 8)
    tower.compute_attn_mask_seqlen = lambda cu: int((cu[1:] - cu[:-1]).max())
    lengths = torch.tensor([8, 19, 64, 8, 35])
    output_lengths = (lengths + 7) // 8
    features = torch.randn(8, int(lengths.sum()))
    with torch.inference_mode():
        individual = torch.cat(
            [
                tower(x, lengths[i : i + 1], output_lengths[i : i + 1])
                for i, x in enumerate(features.split(lengths.tolist(), -1))
            ]
        )
        combined = tower(features, lengths, output_lengths)
        legacy = tower._forward_same_padding(features, lengths, output_lengths)
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "0")
        default = tower(features, lengths, output_lengths)
    torch.testing.assert_close(combined, individual, rtol=1e-5, atol=1e-7)
    assert float((legacy - individual).abs().max()) > 1e-5
    torch.testing.assert_close(default, legacy, rtol=0, atol=0)


def test_idle_engine_death_reaches_health_and_admission():
    async def run():
        engine = SimpleNamespace(is_async=True, errored=False, shutdown=lambda: None)
        handler = DecisionServing(engine, "test", 1)
        assert handler.is_running and not handler.errored
        engine.errored = True
        assert handler.errored and not handler.is_running
        with pytest.raises(RuntimeError):
            await handler.check_health()
        assert (await handler.infer({}, "auto", 0.95)).error.code == 503
        await handler.close()

    asyncio.run(run())


def test_invariant_audio_convolution_matches_conv2d_math(monkeypatch):
    import torch.nn.functional as functional
    from vllm.model_executor.determinism import batch_invariant

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import audio_conv2d_batch_invariant

    # CPU checks layout, padding, stride, dilation and bias. The GPU regression
    # exercises the actual invariant kernel rather than this CPU substitute.
    monkeypatch.setattr(batch_invariant, "linear_batch_invariant", functional.linear)
    torch.manual_seed(34)
    for shape in ((1, 2, 8, 7), (3, 2, 9, 11)):
        for dilation in (1, 2):
            conv = nn.Conv2d(2, 4, 3, stride=2, padding=1, dilation=dilation).double()
            inputs = torch.randn(shape, dtype=torch.float64)
            torch.testing.assert_close(audio_conv2d_batch_invariant(inputs, conv), conv(inputs), atol=1e-12, rtol=1e-12)


def test_worker_rejects_measured_audio_count_mismatch():
    requests = [{"request_id": "a", "input_tokens": 1, "mode": "auto", "threshold": 0.95}]
    model, calls = fake_model(requests)
    model.batch_audio_items = 0
    with pytest.raises(RuntimeError, match="freshly encoded audio"):
        model.forward(None, torch.arange(1), inputs_embeds=torch.zeros(1, 2))
    assert not calls


def test_shutdown_timeout_still_disposes_engine(monkeypatch):
    from vllm_omni.entrypoints.openai import serving_decision

    # Python 3.10's futures exception is distinct from built-in TimeoutError.
    class SimulatedFuturesTimeoutError(Exception):
        pass

    events = []

    class Drain:
        def result(self, timeout):
            events.append(("wait", timeout))
            raise SimulatedFuturesTimeoutError()

        def cancel(self):
            events.append("cancel")

    def submit(coroutine, loop):
        coroutine.close()
        return Drain()

    monkeypatch.setattr(serving_decision, "FuturesTimeoutError", SimulatedFuturesTimeoutError)
    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", submit)
    engine = SimpleNamespace(is_async=True, shutdown=lambda: events.append("shutdown"))
    handler = DecisionServing(engine, "test")
    handler.loop = SimpleNamespace(is_running=lambda: True)
    handler.shutdown(timeout=0.01)
    handler.shutdown(timeout=0.01)
    assert events == [("wait", 0.01), "cancel", "shutdown"]
    assert handler.closed and handler._stopped


@pytest.mark.parametrize(
    "failure", ["cancel_forward", "forward_error", "cancel_take", "take_error", "take_error_after_pop"]
)
def test_async_engine_abort_then_take_cleans_completed_results(failure):
    from vllm_omni.entrypoints.audio_decision.async_engine import AsyncDecisionEngine

    async def run():
        entered = asyncio.Event()
        never = asyncio.Event()
        events = []
        results = {}
        aborted = set()

        class LLM:
            async def generate(self, prompt, params, request_id):
                results[request_id] = {"request_id": request_id, "action": "respond"}
                events.append("stored")
                if failure in ("cancel_forward", "forward_error"):
                    entered.set()
                    if failure == "cancel_forward":
                        await never.wait()
                    raise RuntimeError("forward failed after storing readout")
                yield SimpleNamespace(outputs=[SimpleNamespace(token_ids=[2])])

            async def collective_rpc(self, method, args):
                assert method == "take_decision_result"
                key = args[0]
                if key in aborted:
                    events.append("cleanup_take")
                    return [results.pop(key, None)]
                events.append("take")
                entered.set()
                if failure == "cancel_take":
                    await never.wait()
                if failure == "take_error_after_pop":
                    results.pop(key)
                raise RuntimeError("result retrieval failed")

            async def abort(self, request_id):
                events.append("abort")
                # Simulate asynchronous engine-core acknowledgement. Cleanup
                # must await it before issuing the result-removal RPC.
                await asyncio.sleep(0)
                aborted.add(request_id)

        engine = AsyncDecisionEngine.__new__(AsyncDecisionEngine)
        engine.llm = LLM()
        engine.rendered = "task"
        engine.token_ids = [1, 2, 3]
        engine.config = {"actions": ["keep_listening", "respond", "insufficient_evidence"]}
        task = asyncio.create_task(engine.decide([0.0] * 160, 0.95))
        await asyncio.wait_for(entered.wait(), timeout=2)
        if failure.startswith("cancel"):
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(RuntimeError, match="failed"):
                await task
        assert events[-2:] == ["abort", "cleanup_take"]
        assert len(aborted) == 1 and not results

    asyncio.run(run())
