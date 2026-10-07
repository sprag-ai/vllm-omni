# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The streaming chat path reports ``system_fingerprint`` the way upstream vLLM does.

With ``stream_options.include_usage`` the final usage chunk carries it; without usage the chunk that
carries ``finish_reason`` does.
"""

from unittest.mock import MagicMock

import pytest

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_request,
    make_text_omni_output,
    parse_sse_chunks,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FINGERPRINT = "vllm-0.30.0-test"


async def _stream(include_usage: bool) -> list[dict]:
    serving_chat = build_serving_chat()
    serving_chat.system_fingerprint = FINGERPRINT
    request = make_request(modalities=["text"], include_usage=include_usage)

    async def result_generator():
        yield make_text_omni_output(text="he", token_ids=[10, 11], finish_reason=None)
        yield make_text_omni_output(text="llo", token_ids=[12], finish_reason="stop")

    raw_lines = await collect_stream(
        serving_chat.chat_completion_stream_generator(
            request=request,
            result_generator=result_generator(),
            request_id="test-req",
            model_name="test-model",
            conversation=[],
            tokenizer=MagicMock(),
            request_metadata=MagicMock(),
        )
    )
    return parse_sse_chunks(raw_lines)


@pytest.mark.asyncio
async def test_final_usage_chunk_carries_system_fingerprint():
    chunks = await _stream(include_usage=True)
    usage_chunks = [c for c in chunks if c.get("usage") and not c.get("choices")]
    assert len(usage_chunks) == 1
    assert usage_chunks[0]["system_fingerprint"] == FINGERPRINT


@pytest.mark.asyncio
async def test_finish_chunk_carries_system_fingerprint_without_usage():
    chunks = await _stream(include_usage=False)
    finishing = [c for c in chunks if any(ch.get("finish_reason") for ch in c.get("choices", []))]
    others = [c for c in chunks if c not in finishing]
    assert len(finishing) == 1
    assert finishing[0]["system_fingerprint"] == FINGERPRINT
    assert all("system_fingerprint" not in c for c in others)
