# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate the deliberately restricted native-audio decision CLI."""

import argparse

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
        "decision_max_pending",
        "gpu_memory_utilization",
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
    if not getattr(args, "decision_bundle", None):
        return
    explicit_keys = getattr(args, "explicit_keys", None)
    if explicit_keys is None:
        raise ValueError("Decision serving requires explicit argument tracking; use the Omni TrackingArgumentParser")
    unsupported = sorted(explicit_keys - SUPPORTED_DECISION_ARGS)
    if unsupported:
        flags = ", ".join("--" + key.replace("_", "-") for key in unsupported)
        raise ValueError(
            f"Unsupported options with --decision-bundle: {flags}. "
            "Decision engine settings are frozen (BF16, max-model-len=2048, eager, seed=17); "
            "only --gpu-memory-utilization is a configurable engine option. Remove the unsupported options."
        )
    if getattr(args, "tensor_parallel_size", 1) != 1 or getattr(args, "pipeline_parallel_size", 1) != 1:
        raise ValueError("Decision mode supports TP=1 and PP=1 only")
    if getattr(args, "headless", False) or (getattr(args, "api_server_count", None) or 1) != 1:
        raise ValueError("Decision serving requires one API server and cannot run headless")
