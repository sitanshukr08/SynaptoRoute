import json
import glob
import random
from pathlib import Path

# Provide a small library of synonyms for synthetic generation without an LLM
SYNONYMS = {
    "orchestrator_agent": {
        "verbs": ["delegate", "assign", "distribute", "coordinate", "plan", "manage", "figure out", "break down"],
        "objects": ["the task", "this project", "the workflow", "the team", "the subagents", "the objectives", "the work"],
        "modifiers": ["immediately", "for tomorrow", "efficiently", "across the team", "into subtasks", "carefully"]
    },
    "research_agent": {
        "verbs": ["search", "summarize", "gather", "scrape", "find", "look up", "read through", "analyze"],
        "objects": ["the web", "this pdf", "historical data", "the website", "statistics", "the population", "the article"],
        "modifiers": ["for the latest info", "and extract main points", "quickly", "and give me a brief", "in detail"]
    },
    "writer_agent": {
        "verbs": ["draft", "rewrite", "generate", "write", "summarize", "compose", "edit", "proofread"],
        "objects": ["a blog post", "this paragraph", "a cover letter", "a story", "these bullet points", "an email", "this text"],
        "modifiers": ["to sound professional", "for grammar", "creatively", "cohesively", "for my boss"]
    },
    # OOD synonyms (None route)
    "None": {
        "verbs": ["tell me", "how do I", "set", "play", "explain to me"],
        "objects": ["a joke", "cook a steak", "a timer", "the meaning of life", "some music", "how to tie a tie"],
        "modifiers": ["for 10 minutes", "in space", "right now", "please", "quickly"]
    },
    # Support
    "billing_issue": {
        "verbs": ["how to cancel", "refund", "I was charged", "dispute", "update", "why did you bill"],
        "objects": ["my account", "the premium package", "my credit card", "this invoice", "my subscription"],
        "modifiers": ["but I am locked", "twice", "by mistake", "for next month"]
    },
    "technical_support": {
        "verbs": ["fix", "I can't access", "it crashed", "reset", "help with", "troubleshoot"],
        "objects": ["my login", "the software", "the API", "the dashboard", "my password"],
        "modifiers": ["again", "error 404", "please help", "it's broken"]
    },
    "sales_inquiry": {
        "verbs": ["pricing for", "I want to buy", "do you offer", "talk to sales", "what is the cost of", "quote for"],
        "objects": ["50 users", "the premium package", "enterprise plan", "bulk licenses"],
        "modifiers": ["yearly", "monthly", "with a discount"]
    },
    # Coding
    "generate_code": {
        "verbs": ["build me", "write", "generate", "create", "implement"],
        "objects": ["a fastAPI server", "a react component", "a python script", "a rust backend", "a SQL query"],
        "modifiers": ["using best practices", "from scratch", "with async", "quickly"]
    },
    "debug_code": {
        "verbs": ["fix", "there's a bug in", "why is this failing", "debug", "resolve the error in"],
        "objects": ["my python script", "the fastAPI server", "this function", "the traceback", "my deployment"],
        "modifiers": ["it throws a TypeError", "please", "I am stuck"]
    },
    "explain_code": {
        "verbs": ["explain", "how does this work", "what does this do", "break down", "walk me through"],
        "objects": ["this python script", "the architecture", "this regex", "the codebase", "this loop"],
        "modifiers": ["step by step", "for a beginner", "in detail"]
    },
    # Travel & Logistics
    "book_flight": {
        "verbs": ["book", "find", "schedule", "reserve", "look up"],
        "objects": ["a flight", "tickets", "airfare", "a plane ticket"],
        "modifiers": ["to tokyo", "for next week", "from new york", "round trip", "cheaply"]
    },
    "book_hotel": {
        "verbs": ["book", "find", "reserve", "look for", "schedule"],
        "objects": ["a hotel", "a room", "an airbnb", "accommodation"],
        "modifiers": ["in paris", "for two nights", "with a pool", "near the airport"]
    },
    # Tools
    "get_weather": {
        "verbs": ["what's the weather", "check the weather", "is it raining", "forecast", "how hot is it"],
        "objects": ["in london", "outside", "today", "tomorrow"],
        "modifiers": ["right now", "please", "this evening"]
    },
    "get_stock_price": {
        "verbs": ["what is the price of", "check stock", "how is", "stock price for", "quote for"],
        "objects": ["AAPL", "TSLA", "the market", "tesla shares", "apple stock"],
        "modifiers": ["today", "right now", "doing today"]
    },
    "search_web": {
        "verbs": ["search for", "google", "look up", "find information on", "query the web for"],
        "objects": ["the latest news", "the current president", "the world cup", "python tutorials"],
        "modifiers": ["online", "on the internet", "please"]
    },
    "calculate_math": {
        "verbs": ["calculate", "what is", "solve", "math:", "compute"],
        "objects": ["2 + 2", "the square root of 144", "15% of 80", "12 times 12"],
        "modifiers": ["please", "exactly", "for me"]
    },
    # Conversation
    "greeting": {
        "verbs": ["hello", "hi", "hey", "good morning", "greetings"],
        "objects": ["there", "bot", "assistant", "friend"],
        "modifiers": ["how are you", "can you help me", "what's up"]
    },
    "goodbye": {
        "verbs": ["bye", "goodbye", "see you later", "exit", "quit"],
        "objects": ["for now", "thanks", "talk later"],
        "modifiers": ["have a good day", "stop", "close"]
    }
}

def generate_combinatorial(route_name, target_count=100):
    if route_name not in SYNONYMS:
        raise ValueError(f"Route '{route_name}' has no defined synonyms in expand_datasets_local.py! Cannot generate synthetic data without risking semantic corruption.")
    
    syn = SYNONYMS[route_name]
    generated = []
    
    # Generate random combinations
    for _ in range(target_count):
        v = random.choice(syn["verbs"])
        o = random.choice(syn["objects"])
        m = random.choice(syn["modifiers"])
        
        # 30% chance to drop modifier for variety
        if random.random() < 0.3:
            query = f"{v} {o}"
        else:
            query = f"{v} {o} {m}"
            
        generated.append({"query": query, "expected_route": route_name if route_name != "None" else None})
        
    return generated

def process_file(filepath, out_dir):
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    routes = [r["name"] for r in data.get("routes", [])]
    if "None" not in routes:
        routes.append("None")
        
    existing_queries = data.get("test_queries", [])
    
    # We want ~100 queries minimum per file. If there are e.g. 3 routes + None (4 total), 
    # we should generate 25 queries per route to reach 100.
    target_per_route = 30 
    
    new_queries = list(existing_queries)
    for route in routes:
        new_queries.extend(generate_combinatorial(route, target_count=target_per_route))
        
    data["test_queries"] = new_queries
    
    out_path = Path(out_dir) / Path(filepath).name
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Generated {len(new_queries)} queries for {Path(filepath).name}")

def main():
    base_dir = Path(__file__).parent.parent
    dataset_dir = base_dir / "benchmarks" / "datasets"
    intermediate_dir = dataset_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    
    for seed_file in glob.glob(str(dataset_dir / "*.json")):
        process_file(seed_file, intermediate_dir)

if __name__ == "__main__":
    main()
