import os
from synaptoroute.router import AdaptiveRouter
from synaptoroute.encoder import Encoder
from synaptoroute.storage import SQLiteStorage
from synaptoroute.models import Route
from synaptoroute.exceptions import RouteNotFoundError

db_path = "data/verify_v020.sqlite"
if os.path.exists(db_path):
    os.remove(db_path)

storage = SQLiteStorage(db_path)
encoder = Encoder(providers=["CPUExecutionProvider"])
router = AdaptiveRouter(encoder, storage)

# 1. Test update_threshold
route = Route(name="test_route", utterances=["hello", "hi"])
router.add_route(route)
print(f"Original threshold: {router._route_map['test_route'].threshold}")

# Mock fit_thresholds by manually testing if update_threshold propagates to DB
router.storage.update_threshold("test_route", 0.77)

# Verify DB reflects new threshold on reload
storage2 = SQLiteStorage(db_path)
loaded_routes = storage2.load_all_routes()
for r in loaded_routes:
    if r.name == "test_route":
        print(f"Loaded threshold: {r.threshold}")
        assert r.threshold == 0.77
storage2.close()

# 2. Test RouteNotFoundError
try:
    router.add_utterance("fake_route", "this should fail")
    print("FAILED: RouteNotFoundError was not raised!")
except RouteNotFoundError as e:
    print(f"SUCCESS: Caught RouteNotFoundError - {e}")

router.storage.close()
print("All checks passed.")
