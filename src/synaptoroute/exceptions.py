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

class RouterCapacityError(SynaptoRouteError):
    """Raised when the router's maximum capacity is exceeded."""
    pass


class StorageMutationError(SynaptoRouteError):
    """Raised when an asynchronously queued storage mutation fails."""

    def __init__(self, sequence: int, action: str, detail: str):
        self.sequence = sequence
        self.action = action
        self.detail = detail
        super().__init__(f"Storage mutation {sequence} ({action}) failed: {detail}")


class StorageFlushError(SynaptoRouteError):
    """Raised when a durable barrier observes one or more failed mutations."""

    def __init__(self, failures: list[tuple[int, str, str]]):
        self.failures = failures
        summary = "; ".join(
            f"{sequence}:{action}: {detail}" for sequence, action, detail in failures
        )
        super().__init__(f"{len(failures)} storage mutation(s) failed: {summary}")
