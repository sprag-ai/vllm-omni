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
