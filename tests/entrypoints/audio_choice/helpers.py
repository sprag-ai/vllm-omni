# SPDX-License-Identifier: Apache-2.0
"""Fake renderer and generation client for Choice unit tests."""

from types import SimpleNamespace


async def render_inputs(prompts):
    return [dict(prompt, type="token") for prompt in prompts]


def fake_llm(**kwargs):
    kwargs.setdefault("errored", False)
    return SimpleNamespace(renderer=SimpleNamespace(render_cmpl_async=render_inputs), **kwargs)
