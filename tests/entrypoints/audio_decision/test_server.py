# SPDX-License-Identifier: Apache-2.0
import io
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
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
