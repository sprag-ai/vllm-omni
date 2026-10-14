# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import argparse

import pytest

from vllm_omni.entrypoints.cli.pooling import DEFAULT_MAX_REQUEST_BODY_BYTES, validate_pooling_args
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def parser():
    parser = TrackingArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers(dest="subparser"))
    return parser


def parse(parser, *options):
    return parser.parse_args(["serve", "/unused/model", "--omni", "--runner", "pooling", *options])


@pytest.mark.parametrize(
    "options",
    [
        ["--deploy-config", "stages.yaml"],
        ["--strategy-config", "strategy.yaml"],
        ["--stage-overrides", '{"0": {"max_num_seqs": 2}}'],
        ["--stage-id", "0"],
        ["--omni-master-address", "localhost"],
        ["--omni-master-port", "1234"],
        ["--omni-replica-address", "localhost"],
        ["--async-chunk"],
        ["--no-async-chunk"],
        ["--headless"],
        ["--api-server-count", "2"],
        ["--choice-bundle", "bundle"],
        ["--decision-bundle", "bundle"],
    ],
)
def test_pooling_rejects_ignored_stage_options_before_startup(parser, options):
    args = parse(parser, *options)
    for validate in (validate_pooling_args, OmniServeCommand().validate, OmniServeCommand.cmd):
        with pytest.raises(ValueError, match="single-stage"):
            validate(args)


def test_pooling_config_and_body_limit(parser, tmp_path):
    args = parse(parser)
    validate_pooling_args(args)
    assert args.max_request_body_bytes == DEFAULT_MAX_REQUEST_BODY_BYTES
    validate_pooling_args(parse(parser, "--max-request-body-bytes", "4096"))
    for size in ("0", "-1"):
        with pytest.raises(ValueError, match="positive"):
            OmniServeCommand.cmd(parse(parser, "--max-request-body-bytes", size))
    config = tmp_path / "serve.yaml"
    config.write_text("deploy-config: stages.yaml\n")
    with pytest.raises(ValueError, match="--deploy-config"):
        OmniServeCommand.cmd(parse(parser, "--config", str(config)))


def test_stage_configuration_remains_supported_without_pooling():
    validate_pooling_args(argparse.Namespace(runner="generate", deploy_config="stages.yaml"))
    with pytest.raises(ValueError, match="requires --runner pooling"):
        validate_pooling_args(argparse.Namespace(runner="generate", explicit_keys={"max_request_body_bytes"}))
