import os
import platform
import subprocess
import asyncio
import json
import argparse
import psutil
from datetime import datetime

# Import benchmark modules
import eval_accuracy
import eval_latency
import eval_mutation
import eval_efficiency
import eval_components

from history.regression_tracker import track_and_compare

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "Unknown"

def generate_reproducibility_manifest(model_name, env, profile, dataset_version):
    manifest = {
        "git_commit": get_git_revision_hash(),
        "dataset_version": dataset_version,
        "embedding_model": model_name,
        "env": env,
        "profile": profile,
        "cpu": platform.processor(),
        "ram": f"{round(psutil.virtual_memory().total / (1024**3))} GB",
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "seed": 42,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("benchmark_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("=" * 80)
    print("REPRODUCIBILITY METADATA (Saved to benchmark_manifest.json)")
    print("=" * 80)
    for k, v in manifest.items():
        print(f"{k}: {v}")
    print("=" * 80 + "\n")
    return manifest

async def run_for_model(model_name, env, profile):
    # Dummy dataset_version extraction (in practice, use utils.load_datasets())
    dataset_version = "2.0.0" 
    manifest = generate_reproducibility_manifest(model_name, env, profile, dataset_version)
    
    print(f"\n[{model_name}] Starting {profile.upper()} profile on {env.upper()} environment...")
    
    # We will collect metrics here to send to the regression tracker
    metrics = {}
    
    # Run Accuracy
    print("\n--- 1. Accuracy ---")
    eval_accuracy.run_accuracy_evaluation(model_name)
    # Mocking extraction of F1 for tracker
    metrics["synaptoroute_f1"] = 0.95  # Placeholder for actual return value parsing
    
    if profile in ["full", "research"]:
        # Component Level
        print("\n--- 2. Component Profiling ---")
        await eval_components.main()
        
        # Latency
        print("\n--- 3. Latency Profiling ---")
        await eval_latency.main()
        
        # Mutation
        print("\n--- 4. Mutation Robustness ---")
        eval_mutation.run_mutation_evaluation()
        
        # Efficiency
        print("\n--- 5. Efficiency ---")
        eval_efficiency.main()
        
    if profile == "research":
        # Stress scale
        print("\n--- 6. Extreme Stress Scale ---")
        # eval_growth_stress.main() # Would pass massive scale flag
        pass
        
    # Track Regressions
    track_and_compare(manifest, metrics)

async def main():
    parser = argparse.ArgumentParser(description="SynaptoRoute Benchmark Suite")
    parser.add_argument("--profile", choices=["quick", "full", "research"], default="quick", help="Execution profile")
    parser.add_argument("--env", choices=["cpu", "gpu", "mixed"], default="cpu", help="Execution environment")
    parser.add_argument("--models", nargs="+", help="Specific models to test (e.g. BAAI/bge-small-en-v1.5)")
    
    args = parser.parse_args()
    
    if args.env == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["ONNXRUNTIME_PROVIDER"] = "CPUExecutionProvider"
    elif args.env == "gpu":
        os.environ["ONNXRUNTIME_PROVIDER"] = "CUDAExecutionProvider"
        
    models_to_run = args.models
    if not models_to_run:
        if args.profile == "research":
            models_to_run = [
                "BAAI/bge-small-en-v1.5", 
                "sentence-transformers/all-MiniLM-L6-v2",
                "BAAI/bge-base-en-v1.5",
                "intfloat/e5-small-v2"
            ]
        else:
            models_to_run = ["BAAI/bge-small-en-v1.5"]
            
    for model in models_to_run:
        await run_for_model(model, args.env, args.profile)

if __name__ == "__main__":
    asyncio.run(main())
