"""
SynaptoRoute Asynchronous Routing Example
==========================================
Demonstrates non-blocking high-throughput async query evaluation with batching.
"""

import asyncio
from synaptoroute import AdaptiveRouter, Route

async def main():
    print("Initializing AdaptiveRouter for async workload...")
    # Bounded queue (1,000 queries max) and bounded in-flight batch workers (4)
    router = AdaptiveRouter(max_queue_size=1000, max_in_flight_batches=4)

    router.add_route(
        Route(
            name="password_reset",
            utterances=[
                "I forgot my password",
                "How to reset login password",
                "Locked out of my account",
                "Cannot sign in to dashboard",
            ],
            threshold=0.75,
        )
    )

    router.add_route(
        Route(
            name="shipping_status",
            utterances=[
                "Where is my package?",
                "Track my order shipment",
                "Estimated delivery time",
                "Package marked as delivered but not received",
            ],
            threshold=0.75,
        )
    )

    # Start the background batching worker
    await router.start()
    try:
        queries = [
            "I need to reset my login password",
            "Track package delivery for order #12345",
            "I am locked out and cannot log in",
            "What is the weather like today in Seattle?",
        ]

        # Execute concurrent async matching
        tasks = [router.amatch(q) for q in queries]
        results = await asyncio.gather(*tasks)

        print("\n--- Async Routing Results ---")
        for q, res in zip(queries, results):
            if res.matched:
                print(f"[MATCH] '{q}' -> {res.route_name} (score: {res.score:.4f}, margin: {res.margin})")
            else:
                print(f"[ABSTAIN] '{q}' -> Abstain ({res.decision_reason})")

    finally:
        # Graceful shutdown of worker queue and thread pools
        await router.stop()
        print("\nAsync router stopped successfully.")

if __name__ == "__main__":
    asyncio.run(main())
