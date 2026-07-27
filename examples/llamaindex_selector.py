"""
SynaptoRoute LlamaIndex Query Selector Example
===============================================
Demonstrates adapting AdaptiveRouter as a low-latency selector for LlamaIndex query engines.
"""

from typing import List, Dict, Any
from synaptoroute import AdaptiveRouter, Route

class SynaptoRouteLlamaIndexSelector:
    """LlamaIndex-compatible query selector backed by SynaptoRoute."""

    def __init__(self, router: AdaptiveRouter):
        self.router = router

    def select(self, query: str) -> List[Dict[str, Any]]:
        """Select relevant engine target(s) based on semantic matching."""
        result = self.router.match(query)
        if not result.matched:
            return []
        
        return [
            {
                "index": 0,
                "route_name": result.route_name,
                "score": result.score,
                "reason": result.decision_reason,
            }
        ]

def main():
    print("Initializing AdaptiveRouter for LlamaIndex Selector...")
    router = AdaptiveRouter()

    router.add_route(
        Route(
            name="vector_search_engine",
            utterances=[
                "search document knowledge base",
                "find articles about machine learning",
                "lookup documentation for python api",
            ],
            threshold=0.75,
        )
    )

    router.add_route(
        Route(
            name="sql_analytics_engine",
            utterances=[
                "calculate total revenue for q3",
                "how many active users logged in yesterday",
                "show monthly subscription churn rate",
            ],
            threshold=0.75,
        )
    )

    selector = SynaptoRouteLlamaIndexSelector(router)

    queries = [
        "What was our total revenue last month?",
        "Find technical documentation for the authentication API",
        "Who won the 1998 World Cup?",
    ]

    print("\n--- LlamaIndex Selector Routing Results ---")
    for q in queries:
        selections = selector.select(q)
        print(f"\nQuery: '{q}'")
        if selections:
            sel = selections[0]
            print(f"  Selected Engine Target: {sel['route_name']} (Score: {sel['score']:.4f})")
        else:
            print("  Selected Engine Target: None (Escalate to General LLM)")

    router.close()
    print("\nLlamaIndex selector demo completed.")

if __name__ == "__main__":
    main()
