import os
from enum import Enum
from dataclasses import dataclass

class ProfileType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"

@dataclass
class OptimizationProfile:
    type: ProfileType
    threads: int
    batch_size: int
    batch_timeout: float

def get_profile(profile_type: ProfileType = ProfileType.THROUGHPUT) -> OptimizationProfile:
    cpu_count = os.cpu_count() or 4
    if profile_type == ProfileType.LATENCY:
        return OptimizationProfile(
            type=ProfileType.LATENCY,
            threads=max(1, cpu_count - 1),
            batch_size=1,
            batch_timeout=0.0
        )
    else:
        return OptimizationProfile(
            type=ProfileType.THROUGHPUT,
            threads=1,
            batch_size=32,
            batch_timeout=0.005
        )
