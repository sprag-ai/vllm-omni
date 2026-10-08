# SPDX-License-Identifier: Apache-2.0
"""Decode inline visual evidence using the serving runtime's media loaders."""

import base64
from dataclasses import dataclass
from typing import Any


@dataclass
class VisualEvidence:
    image: Any = None
    video: Any = None

    def multimodal_data(self):
        return {key: [value] for key, value in (("image", self.image), ("video", self.video)) if value is not None}

    def content(self):
        return [{"type": key, key: "provided-array"} for key in self.multimodal_data()]

    def token_expansion(self, processor):
        expansion = 0
        if self.image is not None:
            features = processor.image_processor(images=[self.image.media], return_tensors="pt")
            expansion += int(features["image_grid_thw"].prod()) // processor.image_processor.merge_size**2 - 1
        if self.video is not None:
            frames, _ = self.video.media
            features = processor.video_processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
            expansion += int(features["video_grid_thw"].prod()) // processor.image_processor.merge_size**2 - 1
        return expansion


def decode_visual(request):
    from vllm.multimodal.media import ImageMediaIO, VideoMediaIO

    image_io = ImageMediaIO()
    result = VisualEvidence()
    try:
        if request.input_image is not None:
            result.image = image_io.load_bytes(base64.b64decode(request.input_image.data, validate=True))
        if request.input_video is not None:
            result.video = VideoMediaIO(image_io).load_bytes(base64.b64decode(request.input_video.data, validate=True))
    except Exception as exc:
        # Decoder diagnostics can contain untrusted file metadata.
        raise ValueError("Invalid or unsupported visual media") from exc
    return result
