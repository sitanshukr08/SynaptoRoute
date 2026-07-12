"""Observable acknowledgement primitives for asynchronous storage mutations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from synaptoroute.exceptions import StorageMutationError


@dataclass
class MutationReceipt:
    """Tracks one mutation from in-memory acknowledgement to durable commit."""

    sequence: int
    action: str
    enqueued_at_ns: int = field(default_factory=time.perf_counter_ns)
    durable_at_ns: int | None = field(default=None, init=False)
    _error: BaseException | None = field(default=None, init=False, repr=False)
    _completed: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    @property
    def state(self) -> str:
        if not self._completed.is_set():
            return "queued"
        return "failed" if self._error is not None else "durable"

    @property
    def durable_latency_ms(self) -> float | None:
        if self.durable_at_ns is None:
            return None
        return (self.durable_at_ns - self.enqueued_at_ns) / 1_000_000.0

    def wait_durable(self, timeout: float | None = None) -> float:
        if not self._completed.wait(timeout):
            raise TimeoutError(
                f"Timed out waiting for mutation {self.sequence} ({self.action}) to become durable."
            )
        if self._error is not None:
            raise StorageMutationError(self.sequence, self.action, str(self._error)) from self._error
        latency = self.durable_latency_ms
        if latency is None:
            raise RuntimeError("durable mutation completed without a commit timestamp")
        return latency

    def _mark_durable(self) -> None:
        self.durable_at_ns = time.perf_counter_ns()
        self._completed.set()

    def _mark_failed(self, error: BaseException) -> None:
        self._error = error
        self._completed.set()


@dataclass(frozen=True)
class QueuedStorageMutation:
    action: str
    args: tuple[Any, ...]
    receipt: MutationReceipt
