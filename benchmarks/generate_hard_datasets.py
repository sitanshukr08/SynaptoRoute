import json
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv
import litellm

load_dotenv()

# We will use Groq for fast generation
MODEL = "groq/llama-3.1-8b-instant"

def generate_paraphrases(route_name, seed_utterances, count=20):
    prompt = f"""You are generating training data for an intent classification router.
The route is: '{route_name}'.
Here are some seed examples:
{json.dumps(seed_utterances, indent=2)}

Generate {count} NEW, diverse examples for this route. 
Do not just swap one word. Change the structure, length, and vocabulary.
Keep the same core intent.
Return ONLY a valid JSON list of strings. Do not use markdown wrapping or write any other text."""

    try:
        response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=1500)
        content = response.choices[0].message.content.strip()
        # Clean up potential markdown formatting
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
        
        new_utterances = json.loads(content)
        return list(set(seed_utterances + new_utterances))
    except Exception as e:
        print(f"Error parsing JSON for {route_name}: {e}")
        return seed_utterances

def generate_eval_queries(seeds, category, count=15):
    cat_seeds = [s for s in seeds if s.get("category") == category]
    if not cat_seeds:
        return []
        
    prompt = f"""You are generating evaluation data for a semantic router.
The category of these edge-case queries is: '{category}'.
Here are some seed examples:
{json.dumps(cat_seeds, indent=2)}

Generate {count} NEW, diverse examples for this category.
They must have the exact same 'category' and 'expected_route' behavior as the seeds.
Return ONLY a valid JSON list of objects with 'query', 'expected_route', and 'category' keys. Do not use markdown wrapping or write any other text."""

    try:
        response = litellm.completion(model=MODEL, messages=[{"role": "user", "content": prompt}], max_tokens=1500)
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
        
        new_queries = json.loads(content)
        return new_queries
    except Exception as e:
        print(f"Error parsing JSON for eval queries {category}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", type=str, default="datasets/seeds_stress.json")
    parser.add_argument("--output-file", type=str, default="datasets/stress_dataset.json")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    seed_path = base_dir / args.seed_file
    out_path = base_dir / args.output_file

    with open(seed_path, "r") as f:
        data = json.load(f)

    expanded_routes = []
    print("Expanding routes...")
    for route in data["routes"]:
        print(f"  - {route['name']}")
        expanded_utterances = generate_paraphrases(route["name"], route["utterances"], count=20)
        expanded_routes.append({
            "name": route["name"],
            "utterances": expanded_utterances
        })
        print("Sleeping 20s to avoid rate limits...")
        time.sleep(20)

    print("\nExpanding eval queries...")
    categories = set(s.get("category", "unknown") for s in data["eval_queries"])
    all_eval_queries = list(data["eval_queries"])
    
    for cat in categories:
        print(f"  - Category: {cat}")
        new_evals = generate_eval_queries(data["eval_queries"], cat, count=15)
        all_eval_queries.extend(new_evals)
        print("Sleeping 20s to avoid rate limits...")
        time.sleep(20)

    final_dataset = {
        "routes": expanded_routes,
        "eval_queries": all_eval_queries
    }

    with open(out_path, "w") as f:
        json.dump(final_dataset, f, indent=4)

    print(f"\nSaved expanded dataset to {out_path}")
    print(f"Total Routes: {len(expanded_routes)}")
    print(f"Total Eval Queries: {len(all_eval_queries)}")

if __name__ == "__main__":
    main()
