# SPDX-License-Identifier: Apache-2.0
import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.entrypoints.audio_choice.media import decode_visual


def test_image_loader_uses_bytes_not_container_allowlist():
    data = io.BytesIO()
    Image.new("RGB", (28, 28), (255, 0, 0)).save(data, format="PNG")
    request = SimpleNamespace(
        input_image=SimpleNamespace(data=base64.b64encode(data.getvalue()).decode(), format="unknown-hint"),
        input_video=None,
    )
    visual = decode_visual(request)
    assert visual.image.media.size == (28, 28)
    assert visual.image.media.getpixel((0, 0)) == (255, 0, 0)
    assert visual.content() == [{"type": "image", "image": "provided-array"}]


@pytest.mark.parametrize("modality", ["image", "video"])
def test_invalid_media_is_a_redacted_request_error(modality):
    request = SimpleNamespace(input_image=None, input_video=None)
    setattr(request, "input_" + modality, SimpleNamespace(data=base64.b64encode(b"PRIVATE EVIDENCE").decode()))
    with pytest.raises(ValueError, match="^Invalid or unsupported visual media$"):
        decode_visual(request)


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
