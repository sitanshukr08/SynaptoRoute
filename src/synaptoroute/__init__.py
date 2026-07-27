"""
SynaptoRoute
A high-throughput, local semantic routing engine.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("synaptoroute")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import BaseStorage, SQLiteStorage
from synaptoroute.models import DecisionReason, Route, RouteCandidate, RouterResult
from synaptoroute.durability import MutationReceipt
from synaptoroute.exceptions import StorageFlushError, StorageMutationError

__all__ = [
    "AdaptiveRouter",
    "BaseStorage",
    "DecisionReason",
    "Encoder",
    "MutationReceipt",
    "Route",
    "RouteCandidate",
    "RouterResult",
    "SQLiteStorage",
    "StorageFlushError",
    "StorageMutationError",
    "__version__",
]
