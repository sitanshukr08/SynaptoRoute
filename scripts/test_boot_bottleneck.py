import time
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage

import onnxruntime

db_path = "data/extreme_bench.sqlite"

# Detect available providers and prefer CUDA when present.
_available = onnxruntime.get_available_providers()
_providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in _available else ["CPUExecutionProvider"]
print(f"Using providers: {_providers}")

print("Test 1: Cold Booting existing extreme scale database (10k vectors)...")
# Note: The extreme_bench database might be 50k vectors if we left it there.
start_time = time.time()
storage = SQLiteStorage(db_path)
encoder = Encoder(providers=_providers)
router = AdaptiveRouter(encoder, storage)
duration_pass1 = time.time() - start_time
print(f"Pass 1 Boot Time (Backfilling/Computing): {duration_pass1:.2f}s")
router.storage.close()

print("\nTest 2: Fast Rebooting with Cached BLOBs...")
start_time2 = time.time()
storage2 = SQLiteStorage(db_path)
router2 = AdaptiveRouter(encoder, storage2)
duration_pass2 = time.time() - start_time2
print(f"Pass 2 Boot Time (Mapping from SQLite BLOBs): {duration_pass2:.2f}s")
router2.storage.close()

if duration_pass2 < (duration_pass1 / 2):
    print("\nSUCCESS: BLOB Caching works! The router booted significantly faster.")
else:
    print("\nWARNING: BLOB caching might not have improved boot speed.")
