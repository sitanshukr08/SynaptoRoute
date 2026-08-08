"""Child process used by the abrupt-restart durability benchmark."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from benchmarks.deterministic_encoder import DeterministicHashEncoder
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage


class DelayedSQLiteStorage(SQLiteStorage):
    def __init__(self, db_path: str, delay_seconds: float, synchronous: str):
        super().__init__(db_path, synchronous=synchronous)
        self.delay_seconds = delay_seconds
        self.armed = False

    def _delay(self):
        if self.armed:
            time.sleep(self.delay_seconds)

    def save_route(self, route, embeddings=None, expected_version=None):
        self._delay()
        return super().save_route(route, embeddings, expected_version)

    def add_utterance(self, route_name, utterance, embedding=None, version=None):
        self._delay()
        return super().add_utterance(route_name, utterance, embedding, version)

    def update_threshold(self, route_name, threshold, version=None):
        self._delay()
        return super().update_threshold(route_name, threshold, version)

    def delete_route(self, route_name, expected_version=None):
        self._delay()
        return super().delete_route(route_name, expected_version)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--mode", choices=("memory", "durable"), required=True)
    parser.add_argument(
        "--mutation",
        choices=("add_route", "add_utterance", "update_threshold", "delete_route"),
        default="add_route",
    )
    parser.add_argument("--synchronous", choices=("FULL", "NORMAL"), default="FULL")
    parser.add_argument("--delay-ms", type=float, default=250.0)
    args = parser.parse_args()

    storage = DelayedSQLiteStorage(
        str(args.database),
        args.delay_ms / 1000.0,
        args.synchronous,
    )
    router = AdaptiveRouter(DeterministicHashEncoder(dim=8), storage)
    if args.mutation != "add_route":
        setup = router.add_route(Route(name="base_route", utterances=["base utterance"]))
        setup.wait_durable(timeout=5.0)

    storage.armed = True
    if args.mutation == "add_route":
        receipt = router.add_route(Route(name="crash_route", utterances=["persist me"]))
    elif args.mutation == "add_utterance":
        receipt = router.add_utterance("base_route", "target utterance")
    elif args.mutation == "update_threshold":
        receipt = router.update_threshold("base_route", 0.9)
    else:
        receipt = router.delete_route("base_route")
    if args.mode == "durable":
        receipt.wait_durable(timeout=max(5.0, args.delay_ms / 1000.0 + 2.0))
    args.marker.write_text(
        f"{args.mode}:{args.mutation}:{receipt.state}:{receipt.route_version}\n",
        encoding="utf-8",
    )
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
