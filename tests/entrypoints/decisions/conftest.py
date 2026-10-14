# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import io
from fractions import Fraction

import numpy as np
import pytest


@pytest.fixture
def video_bytes():
    import av

    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 64
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), 32 + index, dtype=np.uint8), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()


@pytest.fixture
def video_factory(tmp_path):
    import av

    def encode(format_name="mp4", codec="mpeg4"):
        path = tmp_path / ("video." + format_name)
        options = {"movflags": "+faststart"} if format_name == "mp4" else {}
        with av.open(str(path), mode="w", format=format_name, options=options) as container:
            stream = container.add_stream(codec, rate=4)
            stream.width = stream.height = 64
            stream.pix_fmt = "yuv420p"
            rng = np.random.default_rng(42)
            for index in range(8):
                frame = av.VideoFrame.from_ndarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8), format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, 4)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path.read_bytes()

    return encode


@pytest.fixture
def audio_only_mp4():
    import av

    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("aac", rate=48000)
        frame = av.AudioFrame.from_ndarray(np.zeros((1, 2048), dtype=np.float32), format="fltp", layout="mono")
        frame.sample_rate = 48000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output.getvalue()
