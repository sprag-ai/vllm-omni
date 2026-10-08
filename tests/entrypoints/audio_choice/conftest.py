# SPDX-License-Identifier: Apache-2.0
import os

import pytest


@pytest.fixture(scope="session")
def visual_processor():
    path = os.environ.get("CHOICE_TEST_MODEL")
    if not path:
        pytest.skip("Set CHOICE_TEST_MODEL for real Qwen visual processor checks")
    from transformers import Qwen3OmniMoeProcessor

    return Qwen3OmniMoeProcessor.from_pretrained(path, local_files_only=True)


@pytest.fixture
def video_bytes(tmp_path):
    import cv2
    import numpy as np

    path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 2, (56, 56))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(np.zeros((56, 56, 3), dtype=np.uint8))
    writer.release()
    return path.read_bytes()
