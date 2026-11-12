# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata

import pytest

from vllm_omni.plugins import OMNI_ENGINE_CLIENT_PLUGINS_GROUP, apply_engine_client_plugins

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
        return list(plugins) if group == OMNI_ENGINE_CLIENT_PLUGINS_GROUP else []

    monkeypatch.setattr(importlib.metadata, "entry_points", entry_points)
    return plugins


def test_no_plugins_returns_client(discovered, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "anything")
    client = object()
    assert apply_engine_client_plugins(client) is client


def test_plugins_need_allowlist(discovered, monkeypatch):
    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    calls = []
    discovered.append(_EntryPoint("wrap", lambda c: calls.append(c) or c))
    client = object()
    assert apply_engine_client_plugins(client) is client
    assert calls == []


def test_allowlisted_plugins_chain_in_order(discovered, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "first,second")
    discovered.append(_EntryPoint("first", lambda c: ("first", c)))
    discovered.append(_EntryPoint("skipped", lambda c: ("skipped", c)))
    discovered.append(_EntryPoint("second", lambda c: ("second", c)))
    assert apply_engine_client_plugins("client") == ("second", ("first", "client"))


def test_plugin_returning_none_fails(discovered, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "bad")
    discovered.append(_EntryPoint("bad", lambda c: None))
    with pytest.raises(TypeError, match="bad"):
        apply_engine_client_plugins(object())


def test_plugin_load_failure_stops_startup(discovered, monkeypatch):
    monkeypatch.setenv("VLLM_PLUGINS", "broken")
    discovered.append(_EntryPoint("broken", ImportError("missing dependency")))
    with pytest.raises(ImportError, match="missing dependency"):
        apply_engine_client_plugins(object())
