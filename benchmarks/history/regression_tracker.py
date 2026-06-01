import json
import os
from typing import Dict, Any

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "benchmark_history.json")

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(history: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def track_and_compare(current_manifest: Dict[str, Any], current_metrics: Dict[str, float]):
    """
    Saves the current run and compares it with the most recent run for the same model/env.
    """
    history = load_history()
    
    # Find the most recent run with the same model
    model_name = current_manifest.get("embedding_model")
    env = current_manifest.get("env", "cpu")
    
    previous_run = None
    for run in reversed(history):
        if run.get("manifest", {}).get("embedding_model") == model_name and run.get("manifest", {}).get("env") == env:
            previous_run = run
            break
            
    print(f"\n=== Regression Tracking ({model_name} on {env.upper()}) ===")
    
    if previous_run:
        prev_metrics = previous_run.get("metrics", {})
        for metric, current_val in current_metrics.items():
            prev_val = prev_metrics.get(metric)
            if prev_val is not None and prev_val != 0:
                delta = ((current_val - prev_val) / prev_val) * 100
                
                # Determine if higher is better (like F1) or lower is better (like Latency)
                if "latency" in metric.lower() or "time" in metric.lower() or "memory" in metric.lower():
                    # Lower is better
                    if delta > 5.0:
                        status = "❌ REGRESSION"
                    elif delta < -5.0:
                        status = "✅ IMPROVEMENT"
                    else:
                        status = "➖ STABLE"
                else:
                    # Higher is better
                    if delta < -5.0:
                        status = "❌ REGRESSION"
                    elif delta > 5.0:
                        status = "✅ IMPROVEMENT"
                    else:
                        status = "➖ STABLE"
                        
                print(f"{metric:<25}: {current_val:>10.4f} (Prev: {prev_val:>10.4f}) | {delta:>+7.2f}% | {status}")
            else:
                print(f"{metric:<25}: {current_val:>10.4f} (NEW)")
    else:
        print("No previous runs found for this configuration. Storing baseline.")
        for metric, current_val in current_metrics.items():
            print(f"{metric:<25}: {current_val:>10.4f} (BASELINE)")
            
    # Save the current run
    run_record = {
        "manifest": current_manifest,
        "metrics": current_metrics
    }
    history.append(run_record)
    save_history(history)
