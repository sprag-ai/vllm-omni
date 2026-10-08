# SPDX-License-Identifier: Apache-2.0
"""Redacted request errors with server-side failure locations."""

import logging
import traceback

logger = logging.getLogger(__name__)


class ChoiceInputError(ValueError):
    """Safe public message with the original exception retained as its cause."""


def log_input_error(error):
    # Native processor messages can embed media. Keep types and stack locations,
    # without formatting exception messages, source lines or frame locals.
    chain = []
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        locations = [f"{f.filename}:{f.lineno} in {f.name}" for f in traceback.extract_tb(error.__traceback__)]
        chain.append(f"{type(error).__name__}: " + ", ".join(locations))
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    logger.warning("Choice input rejected; exception chain: %s", " <- ".join(chain))
