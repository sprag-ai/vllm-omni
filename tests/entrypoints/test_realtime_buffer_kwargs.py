# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session state reaches a model's realtime buffer only through the keyword arguments it declares."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from vllm.inputs import TokensPrompt
from vllm.model_executor.models.qwen3_asr_realtime import Qwen3ASRRealtimeGeneration

from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection, _declared_kwargs
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SESSION = {"tools": [{"type": "function"}], "speaker": "Ethan", "instructions": "Be brief.", "transcription": True}


def test_a_three_argument_buffer_receives_no_session_kwargs():
    assert _declared_kwargs(Qwen3ASRRealtimeGeneration.buffer_realtime_audio, SESSION) == {}


def test_the_omni_buffer_receives_every_session_kwarg():
    assert _declared_kwargs(Qwen3OmniMoeForConditionalGeneration.buffer_realtime_audio, SESSION) == SESSION


def test_a_buffer_receives_only_the_kwargs_it_declares():
    def buffer(audio_stream, input_stream, model_config, speaker=None):
        pass

    assert _declared_kwargs(buffer, SESSION) == {"speaker": "Ethan"}


def test_a_buffer_taking_var_kwargs_receives_every_session_kwarg():
    def buffer(audio_stream, input_stream, model_config, **kwargs):
        pass

    assert _declared_kwargs(buffer, SESSION) == SESSION


def _connection(mocker, buffer) -> RealtimeConnection:
    conn = RealtimeConnection.__new__(RealtimeConnection)
    conn.serving = mocker.Mock()
    conn.serving.model_config.is_encoder_decoder = False
    conn.serving.renderer.render_cmpl_async = mocker.AsyncMock(side_effect=lambda prompts: [dict(prompts[0])])
    conn.serving.model_cls.buffer_realtime_audio = buffer
    conn._tools = SESSION["tools"]
    conn._speaker = SESSION["speaker"]
    conn._instructions = SESSION["instructions"]
    conn._turn_prompt = None
    return conn


def test_a_session_on_a_three_argument_buffer_streams_its_prompts(mocker):
    prompt = TokensPrompt(prompt_token_ids=[1], multi_modal_data={"audio": np.zeros(8, dtype=np.float32)})

    async def buffer(audio_stream, input_stream, model_config):
        yield prompt

    conn = _connection(mocker, buffer)

    async def drain():
        return [item async for item in conn._buffer_realtime_audio_with_tools(None, None, True)]

    [streamed] = asyncio.run(drain())
    assert streamed.prompt["prompt_token_ids"] == [1]
