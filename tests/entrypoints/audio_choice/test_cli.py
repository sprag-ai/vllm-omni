# SPDX-License-Identifier: Apache-2.0
import pytest

from vllm_omni.entrypoints.audio_decision.cli_args import validate_decision_args
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser


def parse(*flags):
    parser = TrackingArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers(dest="subparser"))
    return parser.parse_args(["serve", "/model", "--omni", "--choice-bundle", "/bundle", *flags])


def test_native_choice_cli_tracks_supported_settings():
    args = parse("--max-num-seqs", "8", "--max-num-batched-tokens", "16384")
    validate_decision_args(args)
    assert {"choice_bundle", "max_num_seqs", "max_num_batched_tokens"} <= args.explicit_keys


@pytest.mark.parametrize("flags", [["--dtype", "float16"], ["--max-model-len", "16384"], ["--seed", "42"]])
def test_native_choice_cli_rejects_ignored_settings(flags):
    with pytest.raises(ValueError, match="Unsupported options with --choice-bundle"):
        validate_decision_args(parse(*flags))


def test_choice_and_fixed_decision_modes_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_decision_args(parse("--decision-bundle", "/old-bundle"))


@pytest.mark.parametrize("flag", ["--tensor-parallel-size", "--pipeline-parallel-size"])
def test_choice_rejects_parallelism_at_cli_and_direct_launcher(flag, monkeypatch):
    import asyncio

    from vllm_omni.entrypoints.openai import serving_choice

    args = parse(flag, "2")
    for validate in (validate_decision_args, OmniServeCommand().validate):
        with pytest.raises(ValueError, match="TP=1 and PP=1"):
            validate(args)

    def unexpected(*a, **kw):
        pytest.fail("Invalid parallelism reached engine setup")

    monkeypatch.setattr(serving_choice, "AsyncChoiceEngine", unexpected)
    with pytest.raises(ValueError, match="TP=1 and PP=1"):
        asyncio.run(serving_choice.run_choice_server(args, None))


def test_choice_config_rejects_parallelism(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("tensor-parallel-size: 2\n")
    args = parse("--config", str(config))
    assert "tensor_parallel_size" in args.explicit_keys
    with pytest.raises(ValueError, match="TP=1 and PP=1"):
        validate_decision_args(args)


def test_vision_requires_opt_in_and_tracks_limits():
    assert not parse().choice_enable_vision
    args = parse("--choice-enable-vision", "--choice-max-video-seconds", "30", "--choice-max-video-frames", "900")
    validate_decision_args(args)
    assert args.choice_enable_vision and args.choice_max_video_seconds == 30 and args.choice_max_video_frames == 900
    assert {"choice_enable_vision", "choice_max_video_seconds", "choice_max_video_frames"} <= args.explicit_keys


@pytest.mark.parametrize(
    "flags",
    [
        ["--choice-max-video-seconds", "0"],
        ["--choice-max-video-seconds", "nan"],
        ["--choice-max-video-seconds", "inf"],
        ["--choice-max-video-frames", "0"],
    ],
)
def test_invalid_visual_limits_rejected(flags):
    with pytest.raises(ValueError, match="positive and finite"):
        validate_decision_args(parse("--choice-enable-vision", *flags))


def test_limits_without_vision_are_rejected():
    with pytest.raises(ValueError, match="require --choice-enable-vision"):
        validate_decision_args(parse("--choice-max-video-frames", "100"))


def test_video_config_tracks_opt_in_and_limits(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("choice-enable-vision: true\nchoice-max-video-seconds: 15\nchoice-max-video-frames: 450\n")
    args = parse("--config", str(config))
    validate_decision_args(args)
    assert args.choice_enable_vision and args.choice_max_video_seconds == 15 and args.choice_max_video_frames == 450


@pytest.mark.parametrize("decision", [False, True])
def test_visual_flags_require_choice_bundle(decision):
    args = parse("--choice-enable-vision")
    args.choice_bundle = None
    args.decision_bundle = "/old" if decision else None
    with pytest.raises(ValueError, match="require --choice-bundle"):
        validate_decision_args(args)
