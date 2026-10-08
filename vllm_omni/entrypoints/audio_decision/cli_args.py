# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the deliberately restricted native-audio decision CLI."""

import argparse
import math

# Only options consumed by the decision engine, its HTTP launcher, or the CLI
# parser belong here. Never infer support from an upstream parser default: most
# engine options are intentionally fixed by DecisionEngine and are not forwarded.
SUPPORTED_DECISION_ARGS = frozenset(
    {
        "subparser",
        "config",
        "omni",
        "model",
        "model_tag",
        "decision_bundle",
        "choice_bundle",
        "decision_max_pending",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "api_server_count",
        "host",
        "port",
        "uds",
        "api_key",
        "served_model_name",
        "root_path",
        "allowed_origins",
        "allowed_methods",
        "allowed_headers",
        "allow_credentials",
        "middleware",
        "enable_request_id_headers",
        "disable_fastapi_docs",
        "enable_offline_docs",
        "log_error_stack",
        "uvicorn_log_level",
        "disable_uvicorn_access_log",
        "ssl_keyfile",
        "ssl_certfile",
        "ssl_ca_certs",
        "ssl_cert_reqs",
        "ssl_ciphers",
        "enable_ssl_refresh",
    }
)


def validate_decision_args(args: argparse.Namespace) -> None:
    choice = getattr(args, "choice_bundle", None)
    decision = getattr(args, "decision_bundle", None)
    if choice and decision:
        raise ValueError("--choice-bundle and --decision-bundle are mutually exclusive")
    visual_args = {"choice_enable_vision", "choice_max_video_seconds", "choice_max_video_frames"}
    if not choice and visual_args & (getattr(args, "explicit_keys", None) or set()):
        raise ValueError("Choice visual options require --choice-bundle")
    if not choice and not decision:
        return
    explicit_keys = getattr(args, "explicit_keys", None)
    if explicit_keys is None:
        raise ValueError("Decision serving requires explicit argument tracking; use the Omni TrackingArgumentParser")
    supported = SUPPORTED_DECISION_ARGS | visual_args if choice else SUPPORTED_DECISION_ARGS
    unsupported = sorted(explicit_keys - supported)
    if unsupported:
        flags = ", ".join("--" + key.replace("_", "-") for key in unsupported)
        raise ValueError(
            f"Unsupported options with {'--choice-bundle' if choice else '--decision-bundle'}: {flags}. "
            "Engine settings are fixed (BF16, eager, seed=17; context 2048 for decisions, 8192 for Choice); "
            "configurable engine options are --gpu-memory-utilization, --max-num-seqs and "
            "--max-num-batched-tokens. Remove the unsupported options."
        )
    if getattr(args, "tensor_parallel_size", 1) != 1 or getattr(args, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Decision mode supports TP=1 and PP=1 only")
    if getattr(args, "headless", False) or (getattr(args, "api_server_count", None) or 1) != 1:
        raise ValueError("Decision serving requires one API server and cannot run headless")
    if choice:
        seconds = getattr(args, "choice_max_video_seconds", 60)
        frames = getattr(args, "choice_max_video_frames", 1800)
        if not math.isfinite(seconds) or seconds <= 0 or frames < 1:
            raise ValueError("Choice video limits must be positive and finite")
        if explicit_keys & {"choice_max_video_seconds", "choice_max_video_frames"} and not getattr(
            args, "choice_enable_vision", False
        ):
            raise ValueError("Choice video limits require --choice-enable-vision")
