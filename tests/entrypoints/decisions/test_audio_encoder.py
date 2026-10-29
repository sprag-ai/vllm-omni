# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio encoding grouped by standalone chunk padding."""

import pytest
import torch

from vllm_omni.model_executor.models.spev.audio import encode_audio

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingTower:
    n_window = 50

    def __init__(self):
        self.calls = []

    def _forward_same_padding(self, features, feature_lens, aftercnn_lens):
        self.calls.append(feature_lens.tolist())
        recordings = features.split(feature_lens.tolist(), dim=-1)
        return torch.cat([r[0, :1].expand(n)[:, None] for r, n in zip(recordings, aftercnn_lens.tolist())])


def test_audio_groups_share_standalone_padding_and_keep_order():
    lengths = torch.tensor([80, 300, 120, 80, 100])
    features = torch.cat([torch.full((2, n), float(i)) for i, n in enumerate(lengths.tolist())], dim=-1)
    tower = RecordingTower()
    outputs = encode_audio(tower, features, lengths, lengths // 10)
    assert sorted(tower.calls) == [[80, 80], [300, 120, 100]]
    assert [output.shape[0] for output in outputs] == [8, 30, 12, 8, 10]
    assert [output.unique().item() for output in outputs] == [0, 1, 2, 3, 4]
