# SPDX-License-Identifier: Apache-2.0
"""Decode inline visual evidence using the serving runtime's media loaders."""

import base64
import math
from contextlib import nullcontext
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Any


@dataclass
class VisualEvidence:
    image: Any = None
    video: Any = None
    expansion: int = 0

    def multimodal_data(self):
        return {key: [value] for key, value in (("image", self.image), ("video", self.video)) if value is not None}

    def content(self):
        return [{"type": key, key: "provided-array"} for key in self.multimodal_data()]

    def token_expansion(self, processor):
        expansion = 0
        if self.image is not None:
            features = processor.image_processor(images=[self.image.media], return_tensors="pt")
            expansion += _expansion(features["image_grid_thw"], processor.image_processor.merge_size)
        if self.video is not None:
            frames, _ = self.video.media
            features = processor.video_processor(videos=[frames], do_sample_frames=False, return_tensors="pt")
            expansion += _expansion(features["video_grid_thw"], processor.image_processor.merge_size)
        return expansion


def _expansion(grid, merge_size):
    tokens = int(grid.prod()) // merge_size**2
    if tokens < 1:
        raise ValueError("Visual media must produce at least one token")
    return tokens - 1


def _probe_video(data, max_seconds, max_frames):
    import cv2

    # The pinned OpenCV stream plugin can crash after a failed BytesIO open.
    # Probe a server-owned file before handing validated media to the native loader.
    with NamedTemporaryFile() as source:
        source.write(data)
        source.flush()
        cap = cv2.VideoCapture(source.name, cv2.CAP_FFMPEG)
        try:
            if not cap.isOpened():
                raise ValueError("Cannot read video metadata")
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(frames) or not math.isfinite(fps) or frames < 1 or fps <= 0:
                raise ValueError("Video requires finite positive frame count and frame rate")
            if frames > max_frames or frames / fps > max_seconds:
                raise ValueError("Video exceeds the configured source frame or duration limit")
        finally:
            cap.release()


def decode_visual(request, processor, processor_lock=None, max_video_seconds=60, max_video_frames=1800):
    from vllm.multimodal.media import ImageMediaIO, VideoMediaIO

    image_io = ImageMediaIO()
    result = VisualEvidence()
    try:
        if request.input_image is not None:
            result.image = image_io.load_bytes(base64.b64decode(request.input_image.data, validate=True))
        if request.input_video is not None:
            data = base64.b64decode(request.input_video.data, validate=True)
            _probe_video(data, max_video_seconds, max_video_frames)
            # Keep the decoder consistent with the metadata probe; callers cannot select a backend.
            result.video = VideoMediaIO(image_io, video_backend="opencv", backend="opencv").load_bytes(data)
            if len(result.video.media[0]) == 0:
                raise ValueError("Video contains no decodable frames")
        # Separate visual processor serialization from text/audio preparation.
        with processor_lock if processor_lock is not None else nullcontext():
            result.expansion = result.token_expansion(processor)
    except Exception:
        # Decoder diagnostics can contain untrusted file metadata.
        raise ValueError("Invalid or unsupported visual media") from None
    return result
