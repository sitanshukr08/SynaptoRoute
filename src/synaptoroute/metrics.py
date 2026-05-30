try:
    from prometheus_client import Histogram, Gauge, generate_latest
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class _MockHistogram:
    def __init__(self, name):
        self.name = name
        self.observations = []

    def observe(self, amount):
        self.observations.append(amount)


class _MockGauge:
    def __init__(self, name):
        self.name = name
        self.value = 0.0

    def set(self, value):
        self.value = float(value)

    def inc(self, amount=1.0):
        self.value += float(amount)

    def dec(self, amount=1.0):
        self.value -= float(amount)


class MetricsRegistry:
    def __init__(self):
        if HAS_PROMETHEUS:
            self.inference_latency_seconds = Histogram("inference_latency_seconds", "Latency of routing")
            self.batch_size = Histogram("batch_size", "Batch size of queries processed")
            self.queue_depth = Gauge("queue_depth", "Depth of the query queue")
            self.capacity_usage = Gauge("capacity_usage", "Number of vectors stored in capacity")
        else:
            self.inference_latency_seconds = _MockHistogram("inference_latency_seconds")
            self.batch_size = _MockHistogram("batch_size")
            self.queue_depth = _MockGauge("queue_depth")
            self.capacity_usage = _MockGauge("capacity_usage")

    def export_metrics(self) -> str:
        if HAS_PROMETHEUS:
            return generate_latest().decode("utf-8")
        else:
            lines = []
            for name, metric in vars(self).items():
                if isinstance(metric, _MockHistogram):
                    obs = metric.observations
                    avg = sum(obs) / len(obs) if obs else 0
                    lines.append(f"{name}: count={len(obs)} avg={avg}")
                elif isinstance(metric, _MockGauge):
                    lines.append(f"{name}: value={metric.value}")
            return "\n".join(lines)
