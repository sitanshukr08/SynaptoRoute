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
from synaptoroute.models import Route

__all__ = ["AdaptiveRouter", "Encoder", "BaseStorage", "SQLiteStorage", "Route", "__version__"]
