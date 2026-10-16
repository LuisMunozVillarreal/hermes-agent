"""Strict provider selection, independent of admission policy."""

import threading
from hermes_constants import hermes_home_key
from agent.workload_provider import WORKLOAD_API_VERSION, WorkloadUnavailable


_scopes = {}
_scope_lock = threading.RLock()


def _registry(scope=None):
    with _scope_lock:
        return _scopes.setdefault(scope or hermes_home_key(), WorkloadRegistry())


def snapshot_registration(name, *, scope=None):
    registry = _registry(scope)
    with registry._lock:
        return registry._providers.get(name)


def register_provider(provider, *, scope=None):
    _registry(scope).register(provider.name, provider)


def restore_registration(name, current, replacement, *, scope=None):
    registry = _registry(scope)
    with registry._lock:
        if registry._providers.get(name) is current:
            registry.unregister(name, current)
            if replacement is not None:
                registry.register(name, replacement)


def select_provider(config):
    return _registry().select(config)


def submit(config, spec):
    return _registry().submit(config, spec)


class WorkloadRegistry:
    def __init__(self):
        self._providers = {}
        self._lock = threading.RLock()

    def register(self, name, provider):
        with self._lock:
            if provider.api_version != WORKLOAD_API_VERSION or name in self._providers:
                raise WorkloadUnavailable("incompatible or duplicate workload provider")
            self._providers[name] = provider

    def unregister(self, name, provider):
        with self._lock:
            if self._providers.get(name) is provider:
                del self._providers[name]

    def select(self, config):
        mode = config.get("mode", "disabled")
        if mode == "disabled":
            return None
        if mode != "required":
            raise WorkloadUnavailable("invalid workload mode")
        with self._lock:
            provider = self._providers.get(config.get("provider"))
            if provider is None:
                raise WorkloadUnavailable("required workload provider missing")
            return provider

    def submit(self, config, spec):
        provider = self.select(config)
        if provider is None:
            return None
        try:
            return provider.submit(spec)
        except Exception as exc:
            raise WorkloadUnavailable("workload provider call failed") from exc
