"""Adapter factory — instantiates system adapters by name."""

from __future__ import annotations

from reqlens_eval.adapters.base import SystemAdapter
from reqlens_eval.adapters.baseline import BaselineAdapter
from reqlens_eval.adapters.reqinone import ReqInOneV1Adapter
from reqlens_eval.adapters.reqlens import ReqLensV2Adapter

_REGISTRY: dict[str, type[SystemAdapter]] = {
    "baseline": BaselineAdapter,
    "reqinone_v1": ReqInOneV1Adapter,
    "reqlens_v2": ReqLensV2Adapter,
}

AVAILABLE_SYSTEMS: list[str] = list(_REGISTRY.keys())


def get_adapter(system_id: str) -> SystemAdapter:
    """Return an instantiated adapter for the given system_id.

    Args:
        system_id: One of 'baseline', 'reqinone_v1', 'reqlens_v2'.

    Raises:
        ValueError: If system_id is not registered.
    """
    cls = _REGISTRY.get(system_id)
    if cls is None:
        raise ValueError(
            f"Unknown system '{system_id}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return cls()


def get_all_adapters() -> list[SystemAdapter]:
    """Return one instance of every registered adapter."""
    return [cls() for cls in _REGISTRY.values()]
