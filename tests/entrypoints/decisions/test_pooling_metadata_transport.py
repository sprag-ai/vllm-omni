# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch
from vllm import PoolingParams
from vllm.plugins.io_processors.interface import IOProcessor
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MetadataModel:
    supports_pooling_metadata = True

    def forward(self, *, pooling_metadata, **kwargs):
        return pooling_metadata


@dataclass
class InputBatchStub:
    params: list[PoolingParams]
    metadata: PoolingMetadata
    num_reqs: int = 1

    def get_pooling_params(self):
        return self.params

    def get_pooling_metadata(self):
        return self.metadata


@pytest.mark.parametrize(
    "pooling,opt_in,profiling",
    [
        (False, False, False),
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_pooling_metadata_is_opt_in_and_request_scoped(pooling, opt_in, profiling):
    params = [PoolingParams(task="plugin", extra_kwargs={"request": "first"})]
    ids = torch.tensor([[3, 4]])
    metadata = PoolingMetadata(torch.tensor([2]), ids, ids, params, [PoolingStates()])
    runner = object.__new__(GPUModelRunner)
    runner.lookback_token_ids = None
    runner.is_pooling_model = pooling
    runner.input_batch = InputBatchStub(params, metadata)
    model = MetadataModel() if opt_in else object()
    runner.get_model = lambda: model
    actual = runner._init_model_kwargs(num_reqs=0 if profiling else None)
    if not pooling or not opt_in:
        assert "pooling_metadata" not in actual
    else:
        assert actual["pooling_metadata"] is (None if profiling else metadata)
        other = PoolingMetadata(
            torch.tensor([2]),
            ids,
            ids,
            [PoolingParams(task="plugin", extra_kwargs={"request": "second"})],
            [PoolingStates()],
        )
        runner.input_batch.metadata = other
        assert runner._init_model_kwargs()["pooling_metadata"] is other
        assert actual["pooling_metadata"] is (None if profiling else metadata)


def test_dummy_pooling_metadata_survives_slicing():
    ids = torch.zeros(2, 4, dtype=torch.long)
    metadata = PoolingMetadata(
        torch.tensor([4, 4]),
        ids,
        ids,
        [PoolingParams(task="plugin")] * 2,
        [PoolingStates(), PoolingStates()],
        is_dummy=True,
    )
    assert metadata[:1].is_dummy is True


def test_default_contextual_parameters_delegate_to_existing_plugin():
    class Processor(IOProcessor):
        def pre_process(self, prompt, **kwargs):
            return prompt

        def post_process(self, outputs, **kwargs):
            return outputs

        def merge_pooling_params(self, params=None):
            return params or PoolingParams(task="plugin")

    processor = object.__new__(Processor)
    supplied = PoolingParams(task="plugin", extra_kwargs={"request": "first"})
    assert processor.prepare_pooling_params(object(), supplied) is supplied
    assert processor.prepare_pooling_params(object()).task == "plugin"


@pytest.mark.parametrize("online", [True, False])
def test_entrypoints_prepare_request_metadata_after_preprocessing(online):
    from vllm.config import ModelConfig
    from vllm.entrypoints.pooling.pooling.io_processor import (
        PluginWithIOProcessorPlugins,
    )
    from vllm.entrypoints.pooling.pooling.protocol import IOProcessorRequest
    from vllm.entrypoints.pooling.typing import (
        OfflinePluginInputsContext,
        PoolingServeContext,
    )
    from vllm.renderers.params import TokenizeParams

    class ContextProcessor(IOProcessor):
        def parse_data(self, data):
            return {"id": data, "ready": False}

        def pre_process(self, prompt, **kwargs):
            prompt["ready"] = True
            return b"rendered request"

        def prepare_pooling_params(self, prompt, params=None):
            assert prompt["ready"] is True
            params = params or PoolingParams(task="plugin")
            params.extra_kwargs = {**(params.extra_kwargs or {}), "id": prompt["id"]}
            return params

        def post_process(self, outputs, **kwargs):
            return outputs

    class Request(IOProcessorRequest):
        def build_tok_params(self, model_config):
            return TokenizeParams(max_total_tokens=64)

    class Renderer:
        default_cmpl_tok_params = TokenizeParams(max_total_tokens=64)

    processor = object.__new__(PluginWithIOProcessorPlugins)
    processor.io_processor = object.__new__(ContextProcessor)
    processor.model_config = object.__new__(ModelConfig)
    processor.renderer = Renderer()
    expected: dict[str, object]
    for request_id in ("first", "second"):
        if online:
            online_ctx = PoolingServeContext(
                request=Request(data=request_id),
                model_name="model",
                request_id=request_id,
                pooling_params=PoolingParams(task="plugin"),
                lora_request=None,
                priorities=None,
                prompt_extras=None,
            )
            (request,) = processor.get_request_factory_online(online_ctx)
            expected = {"id": request_id}
        else:
            supplied = PoolingParams(task="plugin", extra_kwargs={"existing": True})
            offline_ctx = OfflinePluginInputsContext(
                prompts={"data": request_id},
                pooling_params=supplied,
                pooling_task="plugin",
                tokenization_kwargs=None,
                lora_request=None,
                priorities=None,
            )
            factory, count = processor.get_request_factory_offline(offline_ctx)
            assert count == 1
            (request,) = factory()
            assert request["params"] is supplied
            expected = {"existing": True, "id": request_id}
        assert request["params"].extra_kwargs == expected
