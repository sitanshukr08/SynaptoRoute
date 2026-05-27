import os
for f in ["bench_classification.log", "bench_performance.log", "bench_scalability.log", "bench_adversarial.log"]:
    try:
        if os.path.exists(f):
            with open(f, "r", encoding="utf-16le") as infile:
                text = infile.read()
            with open(f + ".utf8.txt", "w", encoding="utf-8") as outfile:
                outfile.write(text)
            print(f"Converted {f}")
        else:
            print(f"Not found: {f}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
