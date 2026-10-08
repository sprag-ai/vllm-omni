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
