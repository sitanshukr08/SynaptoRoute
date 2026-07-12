"""Child process used by the abrupt-restart durability benchmark."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from benchmarks.deterministic_encoder import DeterministicHashEncoder
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage


class DelayedSQLiteStorage(SQLiteStorage):
    def __init__(self, db_path: str, delay_seconds: float):
        super().__init__(db_path)
        self.delay_seconds = delay_seconds

    def save_route(self, route, embeddings=None):
        time.sleep(self.delay_seconds)
        return super().save_route(route, embeddings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--mode", choices=("memory", "durable"), required=True)
    parser.add_argument("--delay-ms", type=float, default=250.0)
    args = parser.parse_args()

    storage = DelayedSQLiteStorage(str(args.database), args.delay_ms / 1000.0)
    router = AdaptiveRouter(DeterministicHashEncoder(dim=8), storage)
    receipt = router.add_route(Route(name="crash_route", utterances=["persist me"]))
    if args.mode == "durable":
        receipt.wait_durable(timeout=max(5.0, args.delay_ms / 1000.0 + 2.0))
    args.marker.write_text(f"{args.mode}:{receipt.state}\n", encoding="utf-8")
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
