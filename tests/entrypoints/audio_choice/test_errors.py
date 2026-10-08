# SPDX-License-Identifier: Apache-2.0
import base64
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.audio_choice.errors import ChoiceInputError, log_input_error
from vllm_omni.entrypoints.audio_choice.media import decode_visual


def test_decoder_retains_original_cause_and_logs_locations_without_payload(monkeypatch, caplog):
    from vllm.multimodal.media import ImageMediaIO

    original = ValueError("PRIVATE MEDIA BYTES")

    def fail(*a):
        raise original

    monkeypatch.setattr(ImageMediaIO, "load_bytes", fail)
    request = SimpleNamespace(input_video=None, input_image=SimpleNamespace(data=base64.b64encode(b"private").decode()))
    with pytest.raises(ChoiceInputError) as caught:
        decode_visual(request, object())
    assert caught.value.__cause__ is original
    log_input_error(caught.value)
    assert "ValueError" in caplog.text and "decode_visual" in caplog.text and "in fail" in caplog.text
    assert "PRIVATE MEDIA BYTES" not in caplog.text
