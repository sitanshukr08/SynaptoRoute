"""
SynaptoRoute SQLite Persistence & Mutation Durability Example
===============================================================
Demonstrates dynamic route mutations (add, update, delete) backed by
durable SQLite WAL storage and mutation receipts.
"""

import os
from synaptoroute import AdaptiveRouter, Route, SQLiteStorage

DB_PATH = "data_example/routes.sqlite3"

def main():
    print(f"Initializing SQLiteStorage at '{DB_PATH}'...")

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    storage = SQLiteStorage(db_path=DB_PATH)
    router = AdaptiveRouter(storage=storage)

    # 1. Add a route and observe non-blocking mutation receipt
    auth_route = Route(
        name="auth_service",
        utterances=["login issues", "two factor auth code", "mfa reset"],
        threshold=0.80,
    )

    receipt = router.add_route(auth_route)
    print(f"\n[Mutation Enqueued] Sequence: {receipt.sequence}, Action: {receipt.action}, Initial State: {receipt.state}")

    # Wait for durable commit to SQLite disk
    latency_ms = receipt.wait_durable(timeout=5.0)
    print(f"[Mutation Durable] State: {receipt.state}, Disk Commit Latency: {latency_ms:.2f} ms")

    # 2. Add an utterance dynamically
    utt_receipt = router.add_utterance("auth_service", "sms verification code not arriving")
    utt_receipt.wait_durable()
    print(f"[Utterance Added] Utterances: {router._route_map['auth_service'].utterances}")

    # 3. Explicit durable barrier (flush all queued writes to disk)
    router.durable_barrier(timeout=5.0)
    print("[Durable Barrier] All pending writes flushed to SQLite WAL successfully.")

    # Close router instance
    router.close()
    print("[Router Closed]")

    # 4. Restart Recovery: Instantiate a new router from the existing SQLite database
    print("\n--- Restarting Router from SQLite Persistence ---")
    recovered_storage = SQLiteStorage(db_path=DB_PATH)
    recovered_router = AdaptiveRouter(storage=recovered_storage)

    # Verify recovered routes and utterances
    print(f"Recovered Routes: {list(recovered_router._route_map.keys())}")
    match = recovered_router.match("two factor auth code")
    print(f"Matched Recovered Route: '{match.route_name}' (Score: {match.score:.4f})")

    recovered_router.close()
    print("Recovered router closed.")

if __name__ == "__main__":
    main()
