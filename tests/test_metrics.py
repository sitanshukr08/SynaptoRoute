import pytest
from unittest.mock import patch
from synaptoroute.metrics import MetricsRegistry

def test_metrics_registry_mock_fallback():
    with patch('synaptoroute.metrics.HAS_PROMETHEUS', False):
        registry = MetricsRegistry()
        
        registry.inference_latency_seconds.observe(0.5)
        registry.inference_latency_seconds.observe(1.5)
        registry.batch_size.observe(32)
        
        registry.queue_depth.set(10)
        registry.queue_depth.inc(5)
        registry.queue_depth.dec(2)
        registry.capacity_usage.set(500)
        
        output = registry.export_metrics()
        
        assert "inference_latency_seconds: count=2 avg=1.0" in output
        assert "batch_size: count=1 avg=32.0" in output
        assert "queue_depth: value=13.0" in output
        assert "capacity_usage: value=500.0" in output
