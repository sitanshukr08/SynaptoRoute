import os
import json
import glob
import litellm
from pathlib import Path

def generate_utterances(route_name, seed_utterances, count=20, model="gpt-3.5-turbo"):
    prompt = f"Given the intent '{route_name}' and these seed examples:\n{json.dumps(seed_utterances)}\nGenerate {count} new, diverse examples."
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        content = response.choices[0].message.content
        return [line.strip("- *") for line in content.split("\n") if line.strip()]
    except Exception as e:
        print(f"LLM Error generating utterances for {route_name}: {e}")
        return [f"mock_utterance_{i}" for i in range(count)]

def generate_test_queries(routes, count=100, model="gpt-3.5-turbo"):
    route_names = [r["name"] for r in routes]
    prompt = f"Given these routes: {route_names}\nGenerate {count} test queries mapping to them. Output JSON format: [{{'query': '...', 'expected_route': '...'}}]"
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500
        )
        content = response.choices[0].message.content
        # Try to extract JSON from the output
        start = content.find("[")
        end = content.rfind("]")
        if start != -1 and end != -1:
            return json.loads(content[start:end+1])
        return []
    except Exception as e:
        print(f"LLM Error generating queries: {e}")
        return [{"query": f"mock_query_{i}", "expected_route": route_names[i % len(route_names)]} for i in range(count)]

def main():
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "benchmarks" / "datasets"
    intermediate_dir = dataset_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    seed_files = glob.glob(str(dataset_dir / "*.json"))
    
    # Check for API key to avoid hard failures
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Falling back to mock generation.")

    for seed_file in seed_files:
        print(f"Processing {seed_file}...")
        with open(seed_file, "r") as f:
            data = json.load(f)

        new_data = {"routes": [], "test_queries": []}
        
        # Generate 20-50 utterances
        for route in data.get("routes", []):
            new_utterances = generate_utterances(route["name"], route.get("utterances", []), count=30)
            new_data["routes"].append({
                "name": route["name"],
                "utterances": route.get("utterances", []) + new_utterances
            })
            
        # Generate 100 test queries
        new_queries = generate_test_queries(data.get("routes", []), count=100)
        new_data["test_queries"] = data.get("test_queries", []) + new_queries

        # Save to intermediate
        out_path = intermediate_dir / Path(seed_file).name
        with open(out_path, "w") as f:
            json.dump(new_data, f, indent=2)
            
        print(f"Saved generated candidates to {out_path}")

if __name__ == "__main__":
    main()
