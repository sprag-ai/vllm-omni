# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata

import pytest

from vllm_omni.entrypoints.openai.realtime import transcription
from vllm_omni.plugins import OMNI_REALTIME_TRANSCRIPTION_PLUGINS_GROUP, load_realtime_transcription_plugins

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _EntryPoint:
    def __init__(self, name, func):
        self.name = name
        self.value = f"tests:{name}"
        self._func = func

    def load(self):
        if isinstance(self._func, Exception):
            raise self._func
        return self._func


@pytest.fixture
def discovered(monkeypatch):
    plugins = []

    def entry_points(group):
        return list(plugins) if group == OMNI_REALTIME_TRANSCRIPTION_PLUGINS_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    transcription.transcription_plugins.cache_clear()
    yield plugins
    transcription.transcription_plugins.cache_clear()


class _Turns:
    def __init__(self, connection, config):
        self.connection = connection
        self.config = config


@pytest.fixture
def builtin(monkeypatch):
    monkeypatch.setattr(transcription, "TranscriptionTurns", _Turns)


def test_plugins_need_allowlist(discovered, monkeypatch):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    discovered.append(_EntryPoint("rollback", lambda c, cfg: object()))
    assert load_realtime_transcription_plugins() == []


def test_only_allowlisted_plugins_load(discovered, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "second")
    first, second = (lambda c, cfg: None), (lambda c, cfg: None)
    discovered.extend([_EntryPoint("first", first), _EntryPoint("second", second)])
    assert load_realtime_transcription_plugins() == [second]


def test_plugin_load_failure_stops_startup(discovered, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "broken")
    discovered.append(_EntryPoint("broken", ImportError("missing dependency")))
    with pytest.raises(ImportError, match="missing dependency"):
        load_realtime_transcription_plugins()


def test_no_plugin_builds_builtin_turns(discovered, builtin, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "anything")
    turns = transcription.make_transcription_turns("connection", "config")
    assert isinstance(turns, _Turns) and turns.connection == "connection" and turns.config == "config"


def test_first_claiming_plugin_wins(discovered, builtin, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "declines,claims,later")
    discovered.extend(
        [
            _EntryPoint("declines", lambda c, cfg: None),
            _EntryPoint("claims", lambda c, cfg: ("claims", c, cfg)),
            _EntryPoint("later", lambda c, cfg: ("later", c, cfg)),
        ]
    )
    assert transcription.make_transcription_turns("connection", "config") == ("claims", "connection", "config")


def test_all_declining_falls_back_to_builtin(discovered, builtin, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "declines")
    discovered.append(_EntryPoint("declines", lambda c, cfg: None))
    assert isinstance(transcription.make_transcription_turns("connection", "config"), _Turns)
