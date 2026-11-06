# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The ``turns`` endpoint plugin: ``/v1/turns`` on a server that serves a one-label classifier.

Enable it by naming ``turns`` in ``VLLM_PLUGINS`` when serving a turn classifier checkpoint with ``vllm serve``.
"""

from __future__ import annotations

from argparse import Namespace

from fastapi import FastAPI, WebSocket
from starlette.datastructures import State
from vllm.engine.protocol import EngineClient
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

from vllm_omni.engine.duplex.vad import SileroStreamingVAD, SileroVADBackendProvider
from vllm_omni.entrypoints.openai.turns.connection import VAD_CONFIG, TurnsConnection
from vllm_omni.entrypoints.openai.turns.scorer import TURN_PROMPT_FILE, TurnPrompt, TurnScorer


async def turns_session(websocket: WebSocket) -> None:
    """Accept a ``/v1/turns`` WebSocket and serve it until it closes."""
    state = websocket.app.state
    await websocket.accept()
    vad = SileroStreamingVAD(VAD_CONFIG, backend_provider=state.turns_vad)
    await TurnsConnection(websocket, state.turns_scorer, vad).run()


class TurnsEndpointPlugin:
    """Serve ``/v1/turns`` from the server's classification handler.

    Attributes:
        name: Plugin name to list in ``VLLM_PLUGINS``.
        required_tasks: The plugin loads only on a server that serves classification.
    """

    name = "turns"
    required_tasks = ("classify",)

    def attach_router(self, app: FastAPI) -> None:
        """Register the ``/v1/turns`` WebSocket route."""
        app.add_api_websocket_route("/v1/turns", turns_session)

    async def init_state(self, engine_client: EngineClient | None, state: State, args: Namespace) -> None:
        """Build the shared scorer and VAD backend.

        Raises:
            ValueError: The server serves no one-label classifier, or the model ships no ``turn_prompt.json``.
            ServerVADUnavailableError: The Silero VAD artifact cannot be loaded.
        """
        classification = getattr(state, "serving_classification", None)
        if engine_client is None or classification is None:
            raise ValueError("the turns plugin requires a model served for classification")
        model_config = classification.model_config
        if getattr(model_config.hf_config, "num_labels", None) != 1:
            raise ValueError("the turns plugin requires a classifier with one label, the end-of-turn probability")
        raw = get_hf_file_to_dict(TURN_PROMPT_FILE, model_config.model, model_config.revision)
        if raw is None:
            raise ValueError(f"the turns plugin requires {TURN_PROMPT_FILE} next to the model weights")
        served = args.served_model_name[0] if args.served_model_name else model_config.model
        state.turns_scorer = TurnScorer(
            classification=classification, model=served, prompt=TurnPrompt.model_validate(raw)
        )
        state.turns_vad = SileroVADBackendProvider()
        state.turns_vad.get()
