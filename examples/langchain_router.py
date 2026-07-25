"""
SynaptoRoute LangChain Runnable Integration Example
=====================================================
Demonstrates adapting AdaptiveRouter as a pre-routing component in a LangChain pipeline.
"""

from typing import Any, Dict
from synaptoroute import AdaptiveRouter, Route

class SynaptoRouteLangChainAdapter:
    """LangChain-compatible router adapter using SynaptoRoute for local pre-routing."""

    def __init__(self, router: AdaptiveRouter):
        self.router = router

    def invoke(self, input_data: str | Dict[str, Any]) -> Dict[str, Any]:
        """Process input string or dict and return matched route name and metadata."""
        query = input_data if isinstance(input_data, str) else input_data.get("input", "")
        result = self.router.match(query)
        
        return {
            "query": query,
            "matched": result.matched,
            "route": result.route_name if result.matched else None,
            "score": result.score,
            "decision_reason": result.decision_reason,
            "candidates": [
                {"name": c.route_name, "score": c.score}
                for c in result.candidates
            ],
        }

def main():
    print("Initializing AdaptiveRouter for LangChain Integration...")
    router = AdaptiveRouter()

    router.add_route(
        Route(
            name="account_management",
            utterances=[
                "update my email address",
                "change account password",
                "delete my account",
                "profile settings update",
            ],
            threshold=0.75,
        )
    )

    router.add_route(
        Route(
            name="billing_support",
            utterances=[
                "view recent invoices",
                "update credit card payment details",
                "request refund for last charge",
            ],
            threshold=0.75,
        )
    )

    adapter = SynaptoRouteLangChainAdapter(router)

    # Simulate LangChain Runnable invocation
    test_inputs = [
        "I need to change my account email",
        "Where can I download my billing invoice?",
        "What is the airspeed velocity of an unladen swallow?",
    ]

    print("\n--- LangChain Pre-Routing Output ---")
    for text in test_inputs:
        output = adapter.invoke({"input": text})
        print(f"\nInput: '{output['query']}'")
        print(f"  Matched Route  : {output['route']}")
        print(f"  Confidence     : {output['score']:.4f}")
        print(f"  Decision Reason: {output['decision_reason']}")

    router.close()
    print("\nLangChain adapter demo completed.")

if __name__ == "__main__":
    main()
