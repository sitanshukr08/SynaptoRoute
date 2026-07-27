"""
SynaptoRoute Command Line Interface (CLI)
=========================================
Provides terminal commands:
  synaptoroute info      : Display environment, encoder, and storage details.
  synaptoroute match     : Evaluate a query against local SQLite route database.
  synaptoroute benchmark : Execute the verified CI benchmark suite.
"""

import argparse
import platform

def command_info():
    """Print system environment, Python runtime, and SynaptoRoute details."""
    print("\n--- SynaptoRoute System Information ---")
    print("  SynaptoRoute Version: 0.5.0")
    print(f"  Python Version      : {platform.python_version()}")
    print(f"  Operating System    : {platform.platform()}")
    print(f"  CPU Processor       : {platform.processor() or 'Standard CPU'}")
    
    try:
        from synaptoroute.encoder import FastEmbedEncoder
        enc = FastEmbedEncoder()
        print(f"  FastEmbed Encoder   : Active (Model: {getattr(enc, 'model_name', 'all-MiniLM-L6-v2')}, Dim: {enc.dim})")
    except Exception as e:
        print(f"  FastEmbed Encoder   : Unavailable ({e})")
    print()

def command_benchmark():
    """Execute the self-contained CI verified benchmark suite."""
    from benchmarks.run_verified_ci_benchmark import run_ci_benchmark
    run_ci_benchmark()

def command_match(query: str, db_path: str = ":memory:"):
    """Evaluate a text query against a local route database."""
    from synaptoroute import AdaptiveRouter, Route, SQLiteStorage

    print(f"\nEvaluating query: '{query}' against storage '{db_path}'...")
    storage = SQLiteStorage(db_path=db_path)
    router = AdaptiveRouter(storage=storage)

    if not router._route_map:
        print("Storage is empty. Adding default demo routes (billing, support)...")
        router.add_route(Route(name="billing", utterances=["my payment failed", "invoice status", "refund"]))
        router.add_route(Route(name="support", utterances=["app crashes", "database error", "api timeout"]))
        router.durable_barrier()

    res = router.match(query)
    print("\n--- Match Decision Output ---")
    print(f"  Matched Route  : {res.route_name}")
    print(f"  Matched        : {res.matched}")
    print(f"  Confidence     : {res.score:.4f}" if res.score is not None else "  Confidence     : N/A")
    reason_str = res.decision_reason.value if hasattr(res.decision_reason, "value") else str(res.decision_reason)
    print(f"  Decision Reason: {reason_str}")
    if res.candidates:
        print("\n  Top Candidates:")
        for c in res.candidates[:3]:
            print(f"    - {c.route_name}: score={c.score:.4f} (passed={c.passed_threshold})")
    print()
    router.close()

def main():
    parser = argparse.ArgumentParser(prog="synaptoroute", description="SynaptoRoute Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    subparsers.add_parser("info", help="Display system information and encoder status")

    # Benchmark command
    subparsers.add_parser("benchmark", help="Execute the verified CI benchmark suite")

    # Match command
    match_parser = subparsers.add_parser("match", help="Evaluate a query against routes")
    match_parser.add_argument("query", type=str, help="Text query to match")
    match_parser.add_argument("--db", type=str, default=":memory:", help="Path to SQLite database file")

    args = parser.parse_args()

    if args.command == "info":
        command_info()
    elif args.command == "benchmark":
        command_benchmark()
    elif args.command == "match":
        command_match(args.query, db_path=args.db)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
