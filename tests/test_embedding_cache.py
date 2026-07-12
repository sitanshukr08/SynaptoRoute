import numpy as np

from benchmarks.embedding_cache import MemoizingEncoder


def test_memoizing_encoder_preserves_order_and_avoids_duplicate_work(fake_encoder, monkeypatch):
    calls = []
    original = fake_encoder.encode_batch

    def tracked(texts):
        calls.append(list(texts))
        return original(texts)

    monkeypatch.setattr(fake_encoder, "encode_batch", tracked)
    encoder = MemoizingEncoder(fake_encoder)

    first = encoder.encode_batch(["hello", "bye", "hello"])
    second = encoder.encode_batch(["bye", "hello"])

    assert calls == [["hello", "bye"]]
    assert np.array_equal(first[[1, 0]], second)
    assert encoder.cache_stats() == {"entries": 2, "hits": 3, "misses": 2}


def test_memoizing_encoder_returns_copies(fake_encoder):
    encoder = MemoizingEncoder(fake_encoder)
    first = encoder.encode("hello")
    first[:] = 0.0

    second = encoder.encode("hello")

    assert not np.all(second == 0.0)
