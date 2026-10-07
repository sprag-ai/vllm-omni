# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The streaming chat path reports ``system_fingerprint`` the way upstream vLLM does.

With ``stream_options.include_usage`` the final usage chunk carries it; without usage the chunk that
carries ``finish_reason`` does.
"""

from unittest.mock import MagicMock

import pytest
from vllm.outputs import CompletionOutput, RequestOutput

from tests.helpers.serving_chat import (
    build_serving_chat,
    collect_stream,
    make_request,
    make_text_omni_output,
    parse_sse_chunks,
)
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

FINGERPRINT = "vllm-0.30.0-test"


def _make_audio_omni_output(request_id: str = "test-req") -> OmniRequestOutput:
    completion = CompletionOutput(
        index=0, text="", token_ids=[], cumulative_logprob=0.0, logprobs=None, finish_reason="stop", stop_reason=None
    )
    res = RequestOutput(
        request_id=request_id,
        prompt="test",
        prompt_token_ids=[0, 1, 2],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    return OmniRequestOutput.from_stage_output(
        res, request_id=request_id, stage_id=None, replica_id=None, final_output_type="audio", finished=True
    )


async def _stream(include_usage: bool, modality: str = "text") -> list[dict]:
    serving_chat = build_serving_chat()
    serving_chat.system_fingerprint = FINGERPRINT
    request = make_request(modalities=[modality], include_usage=include_usage)

    async def result_generator():
        if modality == "audio":
            yield _make_audio_omni_output()
            return
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


@pytest.mark.asyncio
async def test_audio_finish_chunk_carries_system_fingerprint_without_usage():
    chunks = await _stream(include_usage=False, modality="audio")
    finishing = [c for c in chunks if any(ch.get("finish_reason") for ch in c.get("choices", []))]
    assert len(finishing) == 1
    assert finishing[0]["system_fingerprint"] == FINGERPRINT


def test_diffusion_instance_has_a_fingerprint_attribute():
    instance = OmniOpenAIServingChat.for_diffusion(MagicMock(), "diffusion-model")
    assert instance.system_fingerprint is None
