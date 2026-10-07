# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import asyncio

import pytest

from vllm_omni.entrypoints.audio_decision.cli_args import validate_decision_args
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def parser():
    # Same tracking parser/subparser path used by the installed `vllm --omni` CLI.
    result = TrackingArgumentParser()
    command = OmniServeCommand()
    command.subparser_init(result.add_subparsers(dest="subparser")).set_defaults(dispatch_function=command.cmd)
    return result


def parse(parser, *options):
    return parser.parse_args(["serve", "/unused/model", "--omni", "--decision-bundle", "/unused/bundle", *options])


@pytest.mark.parametrize(
    "options, key",
    [
        (["--max-model-len", "8192"], "max_model_len"),
        (["--dtype=float16"], "dtype"),
        (["--dtype", "bfloat16"], "dtype"),  # Even a matching frozen value is not a forwarded option.
        (["--seed", "17"], "seed"),
        (["--enforce-eager"], "enforce_eager"),
        (["--enable-prefix-caching"], "enable_prefix_caching"),
        (["--no-enable-prefix-caching"], "enable_prefix_caching"),
        (["--tokenizer", "/other/tokenizer"], "tokenizer"),
        (["--deploy-config", "/unused/config.yaml"], "deploy_config"),
        (["--hf-overrides", '{"architectures": ["OtherModel"]}'], "hf_overrides"),
    ],
)
def test_real_cli_tracks_and_rejects_ignored_flags(parser, options, key):
    args = parse(parser, *options)
    assert key in args.explicit_keys
    for validate in (OmniServeCommand().validate, OmniServeCommand.cmd):
        with pytest.raises(ValueError, match="Unsupported options with --decision-bundle") as exc:
            validate(args)
        assert "--" + key.replace("_", "-") in str(exc.value)


def test_reviewer_example_reports_both_flags(parser):
    args = parse(parser, "--max-model-len", "8192", "--dtype", "float16")
    with pytest.raises(ValueError) as exc:
        OmniServeCommand().validate(args)
    assert "--dtype, --max-model-len" in str(exc.value)


def test_config_engine_options_are_tracked_and_rejected(parser, tmp_path):
    config = tmp_path / "serve.yaml"
    config.write_text("dtype: float16\nmax-model-len: 8192\n")
    args = parse(parser, "--config", str(config))
    assert {"dtype", "max_model_len"} <= args.explicit_keys
    with pytest.raises(ValueError, match="--dtype, --max-model-len"):
        OmniServeCommand().validate(args)


def test_defaults_and_supported_options_are_allowed(parser, monkeypatch, tmp_path):
    from vllm_omni.diffusion.utils import hf_utils

    monkeypatch.setattr(hf_utils, "is_diffusion_model", lambda model: False)
    OmniServeCommand().validate(parse(parser))
    config = tmp_path / "http.yaml"
    config.write_text("port: 8917\ngpu-memory-utilization: 0.8\n")
    args = parse(
        parser,
        "--config",
        str(config),
        "--host",
        "127.0.0.1",
        "--api-key",
        "test-only",
        "--served-model-name",
        "decision",
        "--decision-max-pending",
        "4",
        "--disable-uvicorn-access-log",
        "--enable-request-id-headers",
        "--ssl-keyfile",
        "/unused/key",
        "--ssl-certfile",
        "/unused/cert",
        "-tp",
        "1",
        "-pp",
        "1",
        "--api-server-count",
        "1",
    )
    OmniServeCommand().validate(args)
    assert args.gpu_memory_utilization == 0.8
    assert args.port == 8917


@pytest.mark.parametrize("flag", ["-tp", "-pp"])
def test_parallelism_stays_single_gpu(parser, flag):
    with pytest.raises(ValueError, match="TP=1 and PP=1"):
        OmniServeCommand().validate(parse(parser, flag, "2"))


def test_tracking_is_required_only_for_decision_mode():
    validate_decision_args(argparse.Namespace(decision_bundle=None))
    with pytest.raises(ValueError, match="requires explicit argument tracking"):
        validate_decision_args(argparse.Namespace(decision_bundle="/bundle"))


def test_non_decision_engine_options_remain_supported(parser, monkeypatch):
    from vllm_omni.diffusion.utils import hf_utils

    monkeypatch.setattr(hf_utils, "is_diffusion_model", lambda model: False)
    args = parser.parse_args(["serve", "/unused/model", "--omni", "--dtype", "float16", "--max-model-len", "8192"])
    OmniServeCommand().validate(args)


def test_direct_server_rejects_before_socket_or_model_setup(parser, monkeypatch):
    from vllm_omni.entrypoints.openai import api_server

    def unexpected_setup(*args, **kwargs):
        pytest.fail("invalid decision CLI reached server/model setup")

    monkeypatch.setattr(api_server, "setup_openai_server", unexpected_setup)
    with pytest.raises(ValueError, match="--dtype"):
        asyncio.run(api_server.omni_run_server(parse(parser, "--dtype", "float16")))
