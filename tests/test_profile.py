import os
from synaptoroute.profile import get_profile, ProfileType
from synaptoroute.router import AdaptiveRouter
from synaptoroute.storage import SQLiteStorage

def test_throughput_profile_defaults():
    profile = get_profile(ProfileType.THROUGHPUT)
    assert profile.type == ProfileType.THROUGHPUT
    assert profile.threads == 1
    assert profile.batch_size == 32
    assert profile.batch_timeout == 0.005

def test_latency_profile_defaults():
    profile = get_profile(ProfileType.LATENCY)
    assert profile.type == ProfileType.LATENCY
    # Should use multiple threads but max 1 less than total CPU
    expected_threads = max(1, (os.cpu_count() or 4) - 1)
    assert profile.threads == expected_threads
    assert profile.batch_size == 1
    assert profile.batch_timeout == 0.0

def test_router_inherits_profile(tmp_path, encoder):
    storage = SQLiteStorage(str(tmp_path / "test.db"))
    
    # Test Latency Profile
    latency_profile = get_profile(ProfileType.LATENCY)
    latency_router = AdaptiveRouter(encoder, storage, profile=latency_profile)
    assert latency_router.batch_size == 1
    assert latency_router.batch_timeout == 0.0
    
    # Test Throughput Profile (default)
    default_router = AdaptiveRouter(encoder, storage)
    assert default_router.batch_size == 32
    assert default_router.batch_timeout == 0.005
