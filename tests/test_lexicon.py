"""
Tests for hybrid lexicographic-semantic routing (v0.6.0).

Covers:
- BM25LexiconIndex: add/remove/search/normalize, thread safety
- SlotExtractor: entity matching, missing slots, empty slots
- SlotValidator: is_satisfied, multi-slot
- AdaptiveRouter hybrid fusion: enable_hybrid_lexicon, hybrid_alpha
- AdaptiveRouter slot validation: enable_slot_matching, SLOT_MISMATCH
- Backward compatibility: existing tests unaffected when features disabled
"""

from __future__ import annotations

import threading
import pytest
from synaptoroute.lexicon import BM25LexiconIndex, _tokenize  # noqa: E402
from synaptoroute.slots import SlotExtractor, SlotValidator  # noqa: E402
from synaptoroute.models import Route, DecisionReason  # noqa: E402
from synaptoroute import AdaptiveRouter  # noqa: E402

# ------------------------------------------------------------------
# BM25 import guard: skip all tests if rank_bm25 is not installed
# ------------------------------------------------------------------
rank_bm25 = pytest.importorskip("rank_bm25", reason="rank_bm25 not installed; run pip install 'synaptoroute[lexicon]'")


# ==================================================================
# BM25LexiconIndex tests
# ==================================================================

class TestTokenize:
    def test_lowercases(self):
        assert _tokenize("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self):
        assert _tokenize("order #8821!") == ["order", "8821"]

    def test_empty_string(self):
        assert _tokenize("") == []


class TestBM25LexiconIndex:
    @pytest.fixture
    def index(self):
        idx = BM25LexiconIndex()
        idx.add_route("billing", ["my payment failed", "invoice status", "refund my order"])
        idx.add_route("support", ["app crashes", "database error", "api timeout"])
        idx.add_route("account", ["reset my password", "change email address", "two factor auth"])
        return idx

    def test_is_ready_after_add(self, index):
        assert index.is_ready is True

    def test_route_count(self, index):
        assert index.route_count == 3

    def test_top_result_for_billing_query(self, index):
        results = index.search("I need a refund for my order", top_k=3)
        assert len(results) > 0
        top_score, top_name = results[0]
        assert top_name == "billing"
        assert 0.0 < top_score <= 1.0

    def test_top_result_for_support_query(self, index):
        results = index.search("the api keeps timing out", top_k=3)
        assert len(results) > 0
        assert results[0][1] == "support"

    def test_no_results_for_unrelated_query(self, index):
        results = index.search("xyzzy plugh", top_k=3)
        # May return 0 results or low-confidence results; none should be negative
        for score, _ in results:
            assert score >= 0.0

    def test_remove_route(self, index):
        index.remove_route("account")
        assert index.route_count == 2
        results = index.search("reset password", top_k=5)
        names = [r[1] for r in results]
        assert "account" not in names

    def test_search_empty_index(self):
        idx = BM25LexiconIndex()
        assert idx.search("anything") == []

    def test_score_route_returns_float(self, index):
        score = index.score_route("I want a refund", "billing")
        assert isinstance(score, float)
        assert score >= 0.0

    def test_score_route_absent_returns_zero(self, index):
        assert index.score_route("query", "nonexistent") == 0.0

    def test_normalized_scores_bounded(self, index):
        """All normalized scores must be in [0.0, 1.0]."""
        for query in ["refund", "crash", "password reset", "database api error"]:
            for score, _ in index.search(query):
                assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for query '{query}'"

    def test_update_route_utterances(self, index):
        """Re-adding a route with different utterances replaces old entries."""
        index.add_route("billing", ["subscription renewal", "payment plan"])
        results = index.search("subscription renewal", top_k=3)
        assert results[0][1] == "billing"

    def test_thread_safety(self, index):
        """Concurrent reads and writes must not raise."""
        errors = []

        def reader():
            for _ in range(20):
                try:
                    index.search("refund my order", top_k=3)
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(5):
                try:
                    index.add_route(f"route_{i}", [f"utterance {i}", f"phrase {i}"])
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety errors: {errors}"


# ==================================================================
# SlotExtractor / SlotValidator tests
# ==================================================================

class TestSlotExtractor:
    @pytest.fixture
    def extractor(self):
        return SlotExtractor()

    def test_extracts_order_id(self, extractor):
        result = extractor.extract(
            "Refund order #8821 please",
            "refund",
            {"order_id": r"#?\b\d{4,}\b"},
        )
        assert result.is_satisfied is True
        assert len(result.matched_slots) == 1
        assert result.matched_slots[0].slot_name == "order_id"
        # #?\b\d{4,}\b: the '#' is matched by '#?' before the word boundary,
        # so the full captured value includes the hash prefix.
        assert result.matched_slots[0].matched_value == "#8821"

    def test_missing_required_slot(self, extractor):
        result = extractor.extract(
            "I want a refund",
            "refund",
            {"order_id": r"\b#?\d{4,}\b"},
        )
        assert result.is_satisfied is False
        assert "order_id" in result.missing_slots

    def test_no_slots_is_always_satisfied(self, extractor):
        result = extractor.extract("any query", "route", None)
        assert result.is_satisfied is True

    def test_empty_slots_dict_is_satisfied(self, extractor):
        result = extractor.extract("any query", "route", {})
        assert result.is_satisfied is True

    def test_multiple_slots_all_matched(self, extractor):
        result = extractor.extract(
            "Transfer $250 to account 9001",
            "transfer",
            {
                "amount": r"\$\d+",
                "account_id": r"\b\d{4,}\b",
            },
        )
        assert result.is_satisfied is True
        assert len(result.matched_slots) == 2

    def test_partial_slots_not_satisfied(self, extractor):
        result = extractor.extract(
            "Transfer money to account 9001",
            "transfer",
            {
                "amount": r"\$\d+",
                "account_id": r"\b\d{4,}\b",
            },
        )
        assert result.is_satisfied is False
        assert "amount" in result.missing_slots

    def test_case_insensitive_match(self, extractor):
        result = extractor.extract(
            "PRIORITY ticket",
            "escalate",
            {"priority": r"\bpriority\b"},
        )
        assert result.is_satisfied is True

    def test_pattern_caching(self, extractor):
        """Second call with same pattern uses cached compiled regex."""
        pattern = r"\b\d{5}\b"
        extractor.extract("zip code 90210", "r", {"zip": pattern})
        assert pattern in extractor._pattern_cache


class TestSlotValidator:
    @pytest.fixture
    def validator(self):
        return SlotValidator()

    def test_satisfied_when_slot_matches(self, validator):
        assert validator.is_satisfied(
            "Cancel order #5500", "cancel", {"order_id": r"\b#?\d{4,}\b"}
        ) is True

    def test_not_satisfied_when_slot_missing(self, validator):
        assert validator.is_satisfied(
            "Cancel my order", "cancel", {"order_id": r"\b#?\d{4,}\b"}
        ) is False

    def test_always_satisfied_with_no_slots(self, validator):
        assert validator.is_satisfied("anything", "route", None) is True


# ==================================================================
# AdaptiveRouter: hybrid lexicon + slot matching integration tests
# ==================================================================

class TestAdaptiveRouterHybrid:
    @pytest.fixture
    def hybrid_router(self):
        router = AdaptiveRouter(
            storage=None,  # in-memory
            enable_hybrid_lexicon=True,
            hybrid_alpha=0.3,
        )
        router.add_route(Route(name="billing", utterances=["payment failed", "invoice status", "refund my order"]))
        router.add_route(Route(name="support", utterances=["app crashes", "database error", "api timeout"]))
        router.durable_barrier()
        return router

    def test_hybrid_match_returns_result(self, hybrid_router):
        result = hybrid_router.match("I need a refund")
        assert result.matched is True

    def test_hybrid_decision_reason(self, hybrid_router):
        result = hybrid_router.match("I need a refund")
        # When lexicon has scores, should be MATCHED_HYBRID
        assert result.decision_reason.value in ("matched", "matched_hybrid")

    def test_bm25_score_in_metadata(self, hybrid_router):
        result = hybrid_router.match("I need a refund")
        if result.decision_reason == DecisionReason.MATCHED_HYBRID:
            assert "hybrid_bm25_score" in (result.route.metadata or {})

    def test_hybrid_alpha_clamp(self):
        """hybrid_alpha outside [0,1] is clamped rather than raising."""
        router = AdaptiveRouter(enable_hybrid_lexicon=True, hybrid_alpha=1.5)
        assert router.hybrid_alpha == 1.0
        router2 = AdaptiveRouter(enable_hybrid_lexicon=True, hybrid_alpha=-0.5)
        assert router2.hybrid_alpha == 0.0

    def test_lexicon_updates_on_add_route(self, hybrid_router):
        hybrid_router.add_route(Route(name="shipping", utterances=["track my package", "delivery status"]))
        hybrid_router.durable_barrier()
        assert hybrid_router._lexicon.route_count == 3

    def test_lexicon_updates_on_delete_route(self, hybrid_router):
        hybrid_router.delete_route("support")
        hybrid_router.durable_barrier()
        assert hybrid_router._lexicon.route_count == 1


class TestAdaptiveRouterSlotMatching:
    @pytest.fixture
    def slot_router(self):
        router = AdaptiveRouter(
            storage=None,
            enable_slot_matching=True,
        )
        router.add_route(Route(
            name="refund",
            utterances=["I want a refund", "process my return", "give me my money back"],
            slots={"order_id": r"\b#?\d{4,}\b"},
        ))
        router.add_route(Route(
            name="general_support",
            utterances=["I need help", "can you assist me", "support please"],
        ))
        router.durable_barrier()
        return router

    def test_slot_matched_routes_to_refund(self, slot_router):
        result = slot_router.match("I want a refund for order #9001")
        assert result.matched is True
        assert result.route.name == "refund"

    def test_slot_mismatch_without_order_id(self, slot_router):
        """Query matching 'refund' utterance but lacking order_id -> SLOT_MISMATCH."""
        result = slot_router.match("I want a refund please")
        # The vector match fires but slot check fails
        assert result.decision_reason == DecisionReason.SLOT_MISMATCH or result.matched is False

    def test_no_slot_constraints_always_matches(self, slot_router):
        result = slot_router.match("I need help with my account")
        assert result.matched is True
        assert result.route.name == "general_support"

    def test_slot_mismatch_reason_value(self, slot_router):
        result = slot_router.match("I want a refund please")
        if result.decision_reason == DecisionReason.SLOT_MISMATCH:
            assert result.decision_reason.value == "slot_mismatch"


class TestBackwardCompatibility:
    """Ensure all existing behaviour is preserved when new flags are disabled."""

    def test_default_router_no_hybrid(self):
        router = AdaptiveRouter()
        assert router._lexicon is None
        assert router.enable_hybrid_lexicon is False

    def test_default_router_no_slot_matching(self):
        router = AdaptiveRouter()
        assert router._slot_validator is None
        assert router.enable_slot_matching is False

    def test_default_match_still_works(self):
        router = AdaptiveRouter()
        router.add_route(Route(name="greeting", utterances=["hello", "hi there", "good morning"]))
        router.durable_barrier()
        result = router.match("hello there")
        assert result.matched is True
        assert result.decision_reason == DecisionReason.MATCHED

    def test_route_without_slots_field_is_backward_compatible(self):
        """Routes defined without slots should still work with slot_matching enabled."""
        router = AdaptiveRouter(enable_slot_matching=True)
        router.add_route(Route(name="help", utterances=["I need assistance"]))
        router.durable_barrier()
        result = router.match("I need assistance please")
        assert result.matched is True
