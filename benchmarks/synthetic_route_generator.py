import json
import random
import argparse
from pathlib import Path

# Deterministic components for generation
ACTIONS = ["get", "set", "update", "delete", "fetch", "list", "create", "remove", "analyze", "process"]
ENTITIES = ["user", "account", "profile", "data", "report", "file", "image", "document", "settings", "metrics"]
CONTEXTS = ["in the cloud", "locally", "for the admin", "securely", "asynchronously", "via api", "from database"]

def generate_route_name(idx):
    return f"route_{idx}"

def generate_utterances(idx, num_utterances, rng):
    utterances = []
    for _ in range(num_utterances):
        action = rng.choice(ACTIONS)
        entity = rng.choice(ENTITIES)
        context = rng.choice(CONTEXTS)
        utterance = f"{action} the {entity} {context} {rng.randint(1, 1000)}"
        utterances.append(utterance)
    return utterances

def generate_synthetic_routes(output_path, num_routes=250000, seed=42, utterances_per_route=5):
    """
    Generates JSON routes procedurally and deterministically.
    Memory efficient: streams to file instead of holding in memory.
    """
    rng = random.Random(seed)
    
    with open(output_path, "w") as f:
        f.write('{\n  "routes": [\n')
        
        for i in range(num_routes):
            route = {
                "name": generate_route_name(i),
                "utterances": generate_utterances(i, utterances_per_route, rng)
            }
            
            # Convert to JSON string
            route_json = json.dumps(route, indent=4)
            # Indent for formatting
            route_json = "    " + route_json.replace("\n", "\n    ")
            
            if i < num_routes - 1:
                f.write(route_json + ",\n")
            else:
                f.write(route_json + "\n")
                
        f.write('  ]\n}\n')

def main():
    parser = argparse.ArgumentParser(description="Deterministic Synthetic Route Generator")
    parser.add_argument("--output", type=str, default="synthetic_routes.json", help="Output JSON file path")
    parser.add_argument("--count", type=int, default=10, help="Number of routes to generate (default 10 for quick testing, up to 250000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "benchmarks" / "datasets" / "synthetic"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = data_dir / args.output
    print(f"Generating {args.count} synthetic routes to {out_file}...")
    
    generate_synthetic_routes(out_file, num_routes=args.count, seed=args.seed)
    print("Generation complete.")

if __name__ == "__main__":
    main()
