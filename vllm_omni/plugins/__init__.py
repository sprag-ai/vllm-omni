# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from collections.abc import Callable
from typing import Any

import vllm.envs as envs

logger = logging.getLogger(__name__)

# Default plugins group will be loaded in all processes (process0, engine core
# process and worker processes).
OMNI_DEFAULT_PLUGINS_GROUP = "vllm_omni.general_plugins"
# Platform plugins group will be loaded in all processes when
# `vllm_omni.platforms.current_omni_platform` is called and the value not
# initialized.
OMNI_PLATFORM_PLUGINS_GROUP = "vllm_omni.platform_plugins"
# Engine client plugins run once in the API server process, after the engine
# client is built and before any route handler takes a reference to it. Each
# entry point resolves to a callable that receives the engine client and
# returns the client to serve. They see every request on every route, so they
# load only when named in `VLLM_PLUGINS`, and a plugin that fails stops startup.
OMNI_ENGINE_CLIENT_PLUGINS_GROUP = "vllm_omni.engine_client_plugins"

# Make sure one process only loads plugins once.
omni_plugins_loaded = False


def load_omni_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]:
    from importlib.metadata import entry_points

    allowed_plugins = envs.VLLM_PLUGINS

    discovered_plugins = entry_points(group=group)
    if len(discovered_plugins) == 0:
        logger.debug("No plugins for group %s found.", group)
        return {}

    # Check if the only discovered plugin is the default one.
    is_default_group = group == OMNI_DEFAULT_PLUGINS_GROUP
    # Use INFO for non-default groups and DEBUG for the default group.
    log_level = logger.debug if is_default_group else logger.info

    log_level("Available plugins for group %s:", group)
    for plugin in discovered_plugins:
        log_level("- %s -> %s", plugin.name, plugin.value)

    if allowed_plugins is None:
        log_level("All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.")

    plugins: dict[str, Callable[[], Any]] = {}
    for plugin in discovered_plugins:
        if allowed_plugins is None or plugin.name in allowed_plugins:
            if allowed_plugins is not None:
                log_level("Loading plugin %s", plugin.name)

            try:
                func = plugin.load()
                plugins[plugin.name] = func
            except Exception:
                logger.exception("Failed to load plugin %s", plugin.name)

    return plugins


def load_omni_general_plugins() -> None:
    """WARNING: plugins can be loaded for multiple times in different
    processes. They should be designed in a way that they can be loaded
    multiple times without causing issues.
    """
    global omni_plugins_loaded
    if omni_plugins_loaded:
        return
    omni_plugins_loaded = True

    from vllm.plugins import load_general_plugins

    load_general_plugins()

    plugins = load_omni_plugins_by_group(group=OMNI_DEFAULT_PLUGINS_GROUP)
    # General plugins: we only need to execute the loaded functions.
    for func in plugins.values():
        func()


def apply_engine_client_plugins(engine_client: Any) -> Any:
    """Pass the engine client through each allowlisted engine client plugin, in discovery order."""
    from importlib.metadata import entry_points

    discovered = entry_points(group=OMNI_ENGINE_CLIENT_PLUGINS_GROUP)
    if not discovered:
        return engine_client

    allowed_plugins = envs.VLLM_PLUGINS
    if allowed_plugins is None:
        logger.warning(
            "Found engine client plugin(s) %s but VLLM_PLUGINS is not set. "
            "Engine client plugins see every request and must be allowlisted "
            "via VLLM_PLUGINS to be loaded.",
            [plugin.name for plugin in discovered],
        )
        return engine_client

    for plugin in discovered:
        if plugin.name not in allowed_plugins:
            continue
        result = plugin.load()(engine_client)
        if result is None:
            raise TypeError(f"Engine client plugin {plugin.name} returned None instead of an engine client")
        logger.info("Applied engine client plugin %s -> %s", plugin.name, plugin.value)
        engine_client = result
    return engine_client


__all__ = [
    "apply_engine_client_plugins",
    "load_omni_general_plugins",
    "OMNI_ENGINE_CLIENT_PLUGINS_GROUP",
    "OMNI_DEFAULT_PLUGINS_GROUP",
    "OMNI_PLATFORM_PLUGINS_GROUP",
]
