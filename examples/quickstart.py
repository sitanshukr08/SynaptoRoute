"""
SynaptoRoute Quickstart Example
================================
Demonstrates basic synchronous intent classification using AdaptiveRouter.
"""

from synaptoroute import AdaptiveRouter, Route

def main():
    print("Initializing AdaptiveRouter...")
    # Initialize router with default fast local ONNX encoder
    router = AdaptiveRouter()

    # Define routes with sample utterances and confidence thresholds
    billing_route = Route(
        name="billing",
        utterances=[
            "I need a refund for my order",
            "Where is my receipt or invoice?",
            "Cancel my monthly subscription",
            "My payment was charged twice",
        ],
        threshold=0.75,
    )

    support_route = Route(
        name="technical_support",
        utterances=[
            "The app crashes on startup",
            "Cannot connect to the database",
            "Getting a 500 internal server error",
            "API request timed out",
        ],
        threshold=0.75,
    )

    # Register routes
    router.add_route(billing_route)
    router.add_route(support_route)

    # Query 1: Clear billing query
    query_1 = "How do I get my money back for this purchase?"
    result_1 = router.match(query_1)
    print(f"\nQuery: '{query_1}'")
    print(f"Matched: {result_1.matched}")
    if result_1.matched:
        print(f"Route: {result_1.route_name}")
        print(f"Score: {result_1.score:.4f}")
        print(f"Decision Reason: {result_1.decision_reason}")

    # Query 2: Out of domain / Below threshold query
    query_2 = "What is the capital of France?"
    result_2 = router.match(query_2)
    print(f"\nQuery: '{query_2}'")
    print(f"Matched: {result_2.matched}")
    print(f"Decision Reason: {result_2.decision_reason}")
    if result_2.candidates:
        print(f"Top candidate: '{result_2.candidates[0].route_name}' (score: {result_2.candidates[0].score:.4f}, threshold: {result_2.candidates[0].threshold})")

    # Clean shutdown
    router.close()
    print("\nRouter closed successfully.")

if __name__ == "__main__":
    main()
