"""reqlens_eval.adapters — system adapter registry."""

from reqlens_eval.adapters.base import SystemAdapter
from reqlens_eval.adapters.baseline import BaselineAdapter
from reqlens_eval.adapters.factory import AVAILABLE_SYSTEMS, get_adapter, get_all_adapters
from reqlens_eval.adapters.reqinone import ReqInOneV1Adapter
from reqlens_eval.adapters.reqlens import ReqLensV2Adapter

__all__ = [
    "SystemAdapter",
    "BaselineAdapter",
    "ReqInOneV1Adapter",
    "ReqLensV2Adapter",
    "AVAILABLE_SYSTEMS",
    "get_adapter",
    "get_all_adapters",
]
