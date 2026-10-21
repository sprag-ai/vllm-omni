# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep the model's SDPA selection independent of runtime import side effects."""

from contextlib import contextmanager

from torch.nn.attention import SDPBackend, sdpa_kernel


@contextmanager
def checkpoint_attention():
    with sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION]
    ):
        yield
