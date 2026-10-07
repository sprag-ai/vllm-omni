# SPDX-License-Identifier: Apache-2.0
import hashlib
import io
import json

import numpy as np
import pytest
import soundfile as sf

from vllm_omni.entrypoints.audio_decision.engine import validate_wave, verify_bundle
from vllm_omni.entrypoints.audio_decision.server import decode_audio


@pytest.mark.parametrize(
    "wave,threshold,rate",
    [
        ([], 0.95, 16000),
        ([float("nan")], 0.95, 16000),
        ([[0]], 0.95, 16000),
        ([0], float("nan"), 16000),
        ([0], True, 16000),
        ([0], 1.1, 16000),
        ([0], 0.95, 8000),
    ],
)
def test_invalid_inputs(wave, threshold, rate):
    with pytest.raises(ValueError):
        validate_wave(wave, threshold, rate)


def test_duration_and_decode():
    wave = np.linspace(-0.1, 0.1, 320000, dtype=np.float32)
    assert len(validate_wave(wave, 0.95)) == 320000
    with pytest.raises(ValueError):
        validate_wave(np.zeros(480001), 0.95)
    f = io.BytesIO()
    sf.write(f, np.stack([wave, wave], axis=1), 16000, format="WAV", subtype="FLOAT")
    assert np.array_equal(decode_audio(f.getvalue()), wave)


def test_resampling_and_oversize():
    f = io.BytesIO()
    sf.write(f, np.zeros(8000), 8000, format="WAV")
    assert len(decode_audio(f.getvalue())) == 16000
    f = io.BytesIO()
    sf.write(f, np.zeros(16000 * 31), 16000, format="WAV")
    with pytest.raises(ValueError):
        decode_audio(f.getvalue())


def test_bundle_tamper_and_path_escape(tmp_path):
    head = tmp_path / "head.npz"
    np.savez(head, mean=np.zeros(2), scale=np.ones(2), weight=np.zeros((2, 3)), bias=np.zeros(3))
    cfg = {
        "actions": ["keep_listening", "respond", "insufficient_evidence"],
        "files": {"head.npz": hashlib.sha256(head.read_bytes()).hexdigest()},
    }
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(cfg))
    verify_bundle(tmp_path)
    head.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(tmp_path)
    cfg["files"] = {"../escape": "x"}
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="contained"):
        verify_bundle(tmp_path)
