# SPDX-License-Identifier: Apache-2.0
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from vllm_omni.entrypoints.audio_decision import server


def wav():
    f = io.BytesIO()
    sf.write(f, np.zeros(1600), 16000, format="WAV")
    return f.getvalue()


def test_auth_bad_input_and_queue_recovery(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class FakeEngine:
        def __init__(self, *args):
            pass

        def decide(self, wave, threshold):
            entered.set()
            assert release.wait(5)
            return {"action": "keep_listening", "threshold": threshold}

    monkeypatch.setattr(server, "DecisionEngine", FakeEngine)
    app = server.create_app("unused", "unused", api_key="test-only", max_pending=1)
    headers = {"Authorization": "Bearer test-only"}
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"]
        assert client.post("/v1/decide?threshold=.95", content=wav()).status_code == 401
        assert client.post("/v1/decide?threshold=2", headers=headers, content=wav()).status_code == 422
        assert client.post("/v1/decide?threshold=.95", headers=headers, content=b"bad").status_code == 422
        with ThreadPoolExecutor(1) as pool:
            first = pool.submit(client.post, "/v1/decide?threshold=.95", headers=headers, content=wav())
            assert entered.wait(5)
            assert client.post("/v1/decide?threshold=.8", headers=headers, content=wav()).status_code == 429
            release.set()
            assert first.result().status_code == 200
        assert client.post("/v1/decide?threshold=.8", headers=headers, content=wav()).status_code == 200


@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
def test_frozen_waveform_preprocessing_is_preserved(rate):
    import math

    from scipy.signal import resample_poly

    samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(rate // 2) / rate)).astype(np.float32)
    stereo = np.stack((samples, samples * 0.5), axis=1)
    encoded = io.BytesIO()
    sf.write(encoded, stereo, rate, format="WAV", subtype="PCM_16")
    expected, _ = sf.read(io.BytesIO(encoded.getvalue()), dtype="float32", always_2d=True)
    expected = expected.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        expected = resample_poly(expected, 16000 // divisor, rate // divisor).astype(np.float32)
    np.testing.assert_array_equal(server.decode_audio(encoded.getvalue()), expected)


@pytest.mark.parametrize("samples,rate", [(np.zeros(0), 16000), (np.array([np.nan]), 16000), (np.zeros(100), 4000)])
def test_invalid_waveform_rejected_after_shared_decode(samples, rate):
    encoded = io.BytesIO()
    sf.write(encoded, samples, rate, format="WAV", subtype="FLOAT")
    with pytest.raises(ValueError):
        server.decode_audio(encoded.getvalue())
