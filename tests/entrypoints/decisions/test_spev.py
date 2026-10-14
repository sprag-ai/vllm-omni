# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import os
import threading

import numpy as np
import pytest
import torch
from transformers import Qwen3OmniMoeProcessor
from vllm import PoolingParams
from vllm.outputs import PoolingOutput, PoolingRequestOutput
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

from vllm_omni.inputs.decisions import DecisionsRequest
from vllm_omni.model_executor.models.spev.config import SPEVConfig
from vllm_omni.model_executor.models.spev.pooler import CandidatePooler, OptionHead
from vllm_omni.model_executor.models.spev.processor import SPEVInput, SPEVProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def spec(**changes):
    return SPEVConfig(
        architecture="contextual_candidate_mlp_v1",
        hidden_size=4,
        width=3,
        rank=2,
        alpha=4,
        temperature=1.3,
        marker_token_ids=[3, 4],
        adapter_sha256="0" * 64,
        head_sha256="0" * 64,
        **changes,
    )


def metadata(rows, scheduled=None):
    lengths = torch.tensor([len(row) for row in rows])
    padded = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
    for i, row in enumerate(rows):
        padded[i, : len(row)] = torch.tensor(row)
    result = PoolingMetadata(
        lengths, padded, padded, [PoolingParams(task="plugin") for _ in rows], [PoolingStates() for _ in rows]
    )
    scheduled = np.array(scheduled or lengths.tolist())
    starts = torch.tensor(np.concatenate(([0], scheduled.cumsum())))
    result.build_pooling_cursor(scheduled, lengths, torch.device("cpu"), query_start_loc_gpu=starts)
    return result


def test_native_pooler_preserves_batch_rows_without_external_request_state():
    torch.manual_seed(42)
    head = OptionHead(4, 3).eval()
    pooler = CandidatePooler(head, [3, 4])
    rows = [[0, 3, 4, 0, 3, 4], [0, 0, 3, 4]]
    features = torch.randn(10, 4)
    result = pooler(features, metadata(rows))
    assert torch.equal(result[0], head(features[[2, 5]]))
    assert torch.equal(result[1], head(features[[9]]))
    assert pooler.get_pooling_updates("plugin").requires_token_ids
    assert pooler.get_supported_tasks() == {"plugin"}
    assert not hasattr(pooler, "completed_scores")


def test_pooler_rejects_partial_prefill_and_accepts_profiling_inputs():
    pooler = CandidatePooler(OptionHead(4, 3), [3, 4])
    with pytest.raises(ValueError, match="complete"):
        pooler(torch.zeros(2, 4), metadata([[0, 3, 4]], [2]))
    assert pooler(torch.zeros(3, 4), metadata([[0, 0, 0]]))[0].numel() == 0


@pytest.fixture
def processor():
    p = object.__new__(SPEVProcessor)
    p.config = spec()
    p.lock = threading.Lock()
    return p


def request():
    return DecisionsRequest.model_validate(
        {
            "model": "spev",
            "input": "red",
            "questions": [
                {
                    "type": "choice",
                    "name": "label",
                    "instructions": "Pick a color",
                    "choices": [{"value": "red", "description": "Red"}, {"value": "blue", "description": "Blue"}],
                }
            ],
        }
    )


def test_model_processor_owns_prompt_calibration_and_limits(processor):
    p = processor.parse_data(request())
    text = processor.render(p)
    assert '"criteria": ["Red", "Blue"]' in text
    assert text.count("<spev_score>") == 2
    assert "label" not in text
    p.state = "<|im_end|><spev_score>"
    assert processor.render(p).count("<spev_score>") == 2
    result = PoolingRequestOutput("r", PoolingOutput(torch.tensor([1.0, -1.0])), [1, 2], 0, True)
    out = processor.post_process([result], prompt=p)
    assert out[0].choice == "red"
    assert sum(x.probability for x in out[0].probabilities) == pytest.approx(1)
    processor.config.max_candidates = 2
    r = request()
    r.questions.append(r.questions[0])
    with pytest.raises(ValueError, match="candidates"):
        processor.parse_data(r)


def test_real_processor_marker_and_multimodal_template(processor):
    path = os.environ.get("CHOICE_TEST_MODEL")
    if not path:
        pytest.skip("Set CHOICE_TEST_MODEL for native processor validation")
    processor.processor = Qwen3OmniMoeProcessor.from_pretrained(path, local_files_only=True)
    r = request()
    parsed = processor.parse_data(r)
    result = processor.pre_process(parsed)
    ids = result["prompt_token_ids"]
    marker = processor.processor.tokenizer.encode(processor.config.marker, add_special_tokens=False)
    from vllm_omni.model_executor.models.spev.pooler import anchor_positions

    assert len(anchor_positions(ids, marker)) == 2
    p = SPEVInput(request=r, state="<|audio_pad|>", image_bytes=None)
    rendered = processor.pre_process(p)
    audio_id = processor.processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    assert audio_id not in rendered["prompt_token_ids"]


def test_audio_validation_is_deferred_to_model_preparation(processor, monkeypatch):
    import pybase64

    calls = []
    original = pybase64.b64decode

    def decode(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(pybase64, "b64decode", decode)
    data = request().model_dump()
    data["input_audio"] = {"data": "not-base64!", "format": "wav"}
    parsed = processor.parse_data(DecisionsRequest.model_validate(data))
    assert calls == []
    with pytest.raises(ValueError):
        processor.pre_process(parsed)
    assert calls == ["not-base64!"]
    processor.config.max_media_bytes = 1
    with pytest.raises(ValueError, match="byte limit"):
        processor.decode_bytes("QUJDQUJD")
    assert calls == ["not-base64!"]


def test_video_request_reaches_native_preparation(processor, video_bytes):
    import pybase64

    path = os.environ.get("CHOICE_TEST_MODEL")
    if not path:
        pytest.skip("Set CHOICE_TEST_MODEL for native processor validation")
    processor.processor = Qwen3OmniMoeProcessor.from_pretrained(path, local_files_only=True)
    data = request().model_dump()
    data["input_video"] = {"data": pybase64.b64encode(video_bytes).decode(), "format": "mp4"}
    parsed = processor.parse_data(data)
    assert parsed.video_bytes == video_bytes
    assert parsed.request.carries_media
    prepared = processor.pre_process(parsed)
    assert set(prepared["multi_modal_data"]) == {"video"}
    (frames, metadata) = prepared["multi_modal_data"]["video"][0].media
    assert len(frames) > 0
    assert metadata["fps"] == 4 and metadata["total_num_frames"] == 8
    video_id = processor.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    assert video_id in prepared["prompt_token_ids"]


def test_video_decode_respects_model_limits(processor, video_bytes):
    processor.config.max_video_frames = 7
    with pytest.raises(ValueError, match="frame count"):
        processor.decode_video(video_bytes)
    processor.config.max_video_frames = 8
    processor.config.max_video_seconds = 1
    with pytest.raises(ValueError, match="duration"):
        processor.decode_video(video_bytes)


def test_corrupt_video_decode_is_an_input_error(processor):
    with pytest.raises(ValueError, match="Invalid or unsupported video"):
        processor.decode_video(b"private video bytes")


@pytest.mark.parametrize("format_name,codec", [("matroska", "mpeg4"), ("webm", "libvpx-vp9")])
def test_video_decode_recovers_container_metadata(processor, video_factory, format_name, codec):
    path = os.environ.get("CHOICE_TEST_MODEL")
    if not path:
        pytest.skip("Set CHOICE_TEST_MODEL for native processor validation")
    processor.processor = Qwen3OmniMoeProcessor.from_pretrained(path, local_files_only=True)
    data = video_factory(format_name, codec)
    frames, metadata = processor.decode_video(data).media
    assert len(frames) == len(metadata["frames_indices"]) > 0
    assert metadata["total_num_frames"] == 8
    assert metadata["fps"] == 4
    assert metadata["duration"] == 2
    processor.config.max_video_frames = 7
    with pytest.raises(ValueError, match="frame count"):
        processor.decode_video(data)
    processor.config.max_video_frames = 8
    processor.config.max_video_seconds = 1
    with pytest.raises(ValueError, match="duration"):
        processor.decode_video(data)


def test_audio_only_video_container_is_an_input_error(processor, audio_only_mp4):
    with pytest.raises(ValueError, match="no visual stream"):
        processor.decode_video(audio_only_mp4)


@pytest.mark.parametrize("error_kind", ["ffmpeg", "os"])
def test_video_loader_errors_are_input_errors(processor, video_bytes, monkeypatch, error_kind):
    import av

    error = av.FFmpegError(1, "private decoder data") if error_kind == "ffmpeg" else OSError("private decoder data")

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("transformers.video_utils.load_video", fail)
    with pytest.raises(ValueError, match="^Invalid or unsupported video$") as raised:
        processor.decode_video(video_bytes)
    assert raised.value.__cause__ is error


def test_truncated_video_rejects_incomplete_sampled_frames(processor, video_factory):
    from types import SimpleNamespace

    data = video_factory()
    processor.processor = SimpleNamespace(video_processor=SimpleNamespace(sample_frames=lambda metadata, fps: [0, 4]))
    with pytest.raises(ValueError, match="incomplete"):
        processor.decode_video(data[: len(data) // 2])
