# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage pooling server arguments."""

from argparse import Namespace

DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024


def validate_pooling_args(args: Namespace) -> None:
    if getattr(args, "runner", None) != "pooling":
        if "max_request_body_bytes" in (getattr(args, "explicit_keys", None) or set()):
            raise ValueError("--max-request-body-bytes requires --runner pooling")
        return
    if getattr(args, "max_request_body_bytes", DEFAULT_MAX_REQUEST_BODY_BYTES) < 1:
        raise ValueError("--max-request-body-bytes must be positive")
    incompatible = (
        "deploy_config",
        "strategy_config",
        "stage_overrides",
        "stage_id",
        "omni_master_address",
        "omni_master_port",
        "omni_replica_address",
        "async_chunk",
        "decision_bundle",
        "choice_bundle",
    )
    offenders = [name for name in incompatible if getattr(args, name, None) is not None]
    if getattr(args, "headless", False):
        offenders.append("headless")
    if (getattr(args, "api_server_count", None) or 1) > 1:
        offenders.append("api_server_count")
    if offenders:
        flags = ", ".join("--" + name.replace("_", "-") for name in offenders)
        raise ValueError(f"--runner pooling uses a single-stage server and cannot be combined with {flags}")
