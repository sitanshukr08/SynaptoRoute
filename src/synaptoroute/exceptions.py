class SynaptoRouteError(Exception):
    """Base exception for SynaptoRoute."""
    pass

class RouteNotFoundError(SynaptoRouteError):
    """Raised when a specified route cannot be found."""
    pass

class ModelLoadError(SynaptoRouteError):
    """Raised when an embedding model fails to load."""
    pass

class RouterOverloadedError(SynaptoRouteError):
    """Raised when the dynamic batching queue is full (DDoS protection)."""
    pass
