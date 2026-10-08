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
