# SPDX-License-Identifier: Apache-2.0
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.entrypoints.audio_choice.media import decode_visual


def test_image_loader_uses_bytes_not_container_allowlist(visual_processor):
    data = io.BytesIO()
    Image.new("RGB", (28, 28), (255, 0, 0)).save(data, format="PNG")
    request = SimpleNamespace(
        input_image=SimpleNamespace(data=base64.b64encode(data.getvalue()).decode(), format="unknown-hint"),
        input_video=None,
    )
    visual = decode_visual(request, visual_processor)
    assert visual.image.media.size == (28, 28)
    assert visual.image.media.getpixel((0, 0)) == (255, 0, 0)
    assert visual.content() == [{"type": "image", "image": "provided-array"}]


@pytest.mark.parametrize("modality", ["image", "video"])
def test_invalid_media_is_a_redacted_request_error(modality):
    request = SimpleNamespace(input_image=None, input_video=None)
    setattr(request, "input_" + modality, SimpleNamespace(data=base64.b64encode(b"PRIVATE EVIDENCE").decode()))
    with pytest.raises(ValueError, match="^Invalid or unsupported visual media$"):
        decode_visual(request, object())


def test_video_wrapper_is_a_single_item_for_native_parser():
    import numpy as np
    from vllm.multimodal.media import MediaWithBytes
    from vllm.multimodal.parse import MultiModalDataParser

    from vllm_omni.entrypoints.audio_choice.media import VisualEvidence

    frames = np.zeros((2, 28, 28, 3), dtype=np.uint8)
    metadata = {"fps": 1.0, "duration": 2.0, "total_num_frames": 2, "frames_indices": [0, 1]}
    visual = VisualEvidence(video=MediaWithBytes((frames, metadata), b"fixture"))
    items = MultiModalDataParser(video_needs_metadata=True).parse_mm_data(visual.multimodal_data())
    assert items["video"].get_count() == 1


def media_request(modality, data):
    request = SimpleNamespace(input_image=None, input_video=None)
    setattr(request, "input_" + modality, SimpleNamespace(data=base64.b64encode(data).decode()))
    return request


def test_real_video_processor_and_expansion(video_bytes, visual_processor):
    visual = decode_visual(media_request("video", video_bytes), visual_processor)
    assert len(visual.video.media[0]) == 4
    assert visual.expansion >= 0
    features = visual_processor.video_processor(
        videos=[visual.video.media[0]], do_sample_frames=False, return_tensors="pt"
    )
    assert (
        visual.expansion + 1 == int(features["video_grid_thw"].prod()) // visual_processor.image_processor.merge_size**2
    )


@pytest.mark.parametrize("limits", [{"max_video_seconds": 1}, {"max_video_frames": 3}])
def test_video_budget_rejects_before_frame_decode(video_bytes, visual_processor, monkeypatch, limits):
    from vllm.multimodal.media import VideoMediaIO

    monkeypatch.setattr(VideoMediaIO, "load_bytes", lambda *a: pytest.fail("Over-budget video reached frame decode"))
    with pytest.raises(ValueError, match="^Invalid or unsupported visual media$"):
        decode_visual(media_request("video", video_bytes), visual_processor, **limits)


def test_empty_video_is_rejected_before_processor(video_bytes, monkeypatch):
    import numpy as np
    from vllm.multimodal.media import MediaWithBytes, VideoMediaIO

    monkeypatch.setattr(VideoMediaIO, "load_bytes", lambda *a: MediaWithBytes((np.zeros((0, 28, 28, 3)), {}), b"x"))
    with pytest.raises(ValueError, match="^Invalid or unsupported visual media$"):
        decode_visual(media_request("video", video_bytes), object())


@pytest.mark.parametrize("grid", [[0, 2, 2], [1, 1, 1]])
def test_visual_expansion_requires_one_token(grid):
    import numpy as np

    from vllm_omni.entrypoints.audio_choice.media import _expansion

    with pytest.raises(ValueError, match="at least one token"):
        _expansion(np.array([grid]), 2)


def test_processor_exception_is_redacted():
    data = io.BytesIO()
    Image.new("RGB", (28, 28)).save(data, format="PNG")

    def fail(**kw):
        raise ValueError("PRIVATE PROCESSOR INPUT")

    with pytest.raises(ValueError, match="^Invalid or unsupported visual media$"):
        decode_visual(media_request("image", data.getvalue()), SimpleNamespace(image_processor=fail))


def test_video_metadata_rejects_invalid_values_and_releases_capture(monkeypatch):
    import cv2

    from vllm_omni.entrypoints.audio_choice.media import _probe_video

    for count, fps in [(0, 30), (30, 0), (float("nan"), 30), (30, float("inf"))]:
        released = []
        cap = SimpleNamespace(
            isOpened=lambda: True,
            get=lambda prop: count if prop == cv2.CAP_PROP_FRAME_COUNT else fps,
            release=lambda: released.append(True),
        )
        monkeypatch.setattr(cv2, "VideoCapture", lambda *a: cap)
        with pytest.raises(ValueError, match="finite positive"):
            _probe_video(b"video", 60, 1800)
        assert released == [True]
