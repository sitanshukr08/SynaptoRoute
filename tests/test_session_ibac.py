"""
Tests for session-aware routing and Intent-Based Access Control (IBAC).
"""
from __future__ import annotations

import time
import pytest

from synaptoroute import AdaptiveRouter, Route
from synaptoroute.models import DecisionReason
from synaptoroute.session import SessionContext, SessionStore, apply_session_boost


# ================================================================== #
# SessionContext tests                                                 #
# ================================================================== #

class TestSessionContext:
    def test_record_and_weights(self):
        ctx = SessionContext("s1", window=5)
        ctx.record("billing", 0.9)
        ctx.record("billing", 0.85)
        weights = ctx.recency_weights()
        assert "billing" in weights
        assert weights["billing"] == pytest.approx(1.0)

    def test_exponential_decay(self):
        ctx = SessionContext("s2", window=5)
        ctx.record("billing", 0.9)
        ctx.record("refund", 0.85)
        weights = ctx.recency_weights()
        # "refund" most recent => weight 1.0; "billing" position 1 => weight 0.5
        assert weights["refund"] == pytest.approx(1.0)
        assert weights["billing"] == pytest.approx(0.5)

    def test_not_expired_immediately(self):
        ctx = SessionContext("s3", ttl_seconds=3600.0)
        assert ctx.is_expired() is False

    def test_expired_after_ttl(self):
        ctx = SessionContext("s4", ttl_seconds=0.01)
        time.sleep(0.05)
        assert ctx.is_expired() is True

    def test_window_cap(self):
        ctx = SessionContext("s5", window=3)
        for i in range(10):
            ctx.record(f"route_{i}", 0.8)
        weights = ctx.recency_weights()
        # Only 3 most recent entries should be in weights
        assert len(weights) <= 3


# ================================================================== #
# SessionStore tests                                                   #
# ================================================================== #

class TestSessionStore:
    def test_creates_new_session(self):
        store = SessionStore()
        ctx = store.get_or_create("sess_a")
        assert ctx is not None
        assert ctx.session_id == "sess_a"

    def test_returns_same_context(self):
        store = SessionStore()
        ctx1 = store.get_or_create("sess_b")
        ctx1.record("billing", 0.9)
        ctx2 = store.get_or_create("sess_b")
        assert ctx2.recency_weights().get("billing", 0.0) == pytest.approx(1.0)

    def test_recency_weights_empty_for_unknown(self):
        store = SessionStore()
        weights = store.recency_weights("nonexistent_session")
        assert weights == {}

    def test_record_and_retrieve(self):
        store = SessionStore()
        store.record("sess_c", "shipping", 0.88)
        weights = store.recency_weights("sess_c")
        assert "shipping" in weights

    def test_clear_session(self):
        store = SessionStore()
        store.record("sess_d", "cancel", 0.75)
        store.clear("sess_d")
        weights = store.recency_weights("sess_d")
        assert weights == {}

    def test_expired_session_recreated(self):
        store = SessionStore(default_ttl=0.01)
        store.record("sess_e", "billing", 0.9)
        time.sleep(0.05)
        # Expired context should not return weights
        weights = store.recency_weights("sess_e")
        assert weights == {}

    def test_active_count(self):
        store = SessionStore()
        store.record("s1", "billing", 0.9)
        store.record("s2", "refund", 0.8)
        assert store.active_count() >= 2


# ================================================================== #
# apply_session_boost tests                                           #
# ================================================================== #

class TestApplySessionBoost:
    def _make_candidates(self):
        """Create a simple best_by_route dict with dummy Route-like objects."""
        class FakeRoute:
            def __init__(self, name): self.name = name
        return {
            "billing": (0.80, FakeRoute("billing")),
            "refund":  (0.78, FakeRoute("refund")),
        }

    def test_boost_applied(self):
        candidates = self._make_candidates()
        weights = {"billing": 1.0}
        result = apply_session_boost(candidates, weights, session_alpha=0.05)
        # billing should be boosted above its base 0.80
        assert result["billing"][0] > 0.80
        # refund should be unchanged
        assert result["refund"][0] == pytest.approx(0.78)

    def test_boost_capped_at_1(self):
        candidates = {"billing": (0.99, None)}
        weights = {"billing": 1.0}
        result = apply_session_boost(candidates, weights, session_alpha=0.10)
        assert result["billing"][0] <= 1.0

    def test_zero_alpha_no_change(self):
        candidates = self._make_candidates()
        weights = {"billing": 1.0}
        result = apply_session_boost(candidates, weights, session_alpha=0.0)
        assert result["billing"][0] == pytest.approx(0.80)

    def test_alpha_clamped_to_015(self):
        """alpha > 0.15 should be clamped internally."""
        candidates = {"billing": (0.80, None)}
        weights = {"billing": 1.0}
        result_clamped = apply_session_boost(candidates, weights, session_alpha=1.0)
        result_max = apply_session_boost(candidates, weights, session_alpha=0.15)
        assert result_clamped["billing"][0] == pytest.approx(result_max["billing"][0])

    def test_empty_weights_no_change(self):
        candidates = self._make_candidates()
        result = apply_session_boost(candidates, {}, session_alpha=0.05)
        assert result["billing"][0] == pytest.approx(0.80)


# ================================================================== #
# AdaptiveRouter: session routing integration                         #
# ================================================================== #

class TestAdaptiveRouterSession:
    @pytest.fixture
    def session_router(self):
        router = AdaptiveRouter(
            storage=None,
            enable_session_routing=True,
            session_alpha=0.05,
            session_window=5,
        )
        router.add_route(Route(name="billing", utterances=["payment issue", "invoice problem", "billing query"]))
        router.add_route(Route(name="support", utterances=["app crash", "system down", "api error"]))
        router.durable_barrier()
        return router

    def test_cold_session_matches(self, session_router):
        result = session_router.match("payment issue", session_id="user_001")
        assert result.matched is True

    def test_session_recorded_after_match(self, session_router):
        session_router.match("payment issue", session_id="user_002")
        weights = session_router._session_store.recency_weights("user_002")
        assert "billing" in weights

    def test_no_session_id_no_boost(self, session_router):
        """match() without session_id behaves identically to old API."""
        result = session_router.match("payment issue")
        assert result.matched is True
        # No session metadata injected
        assert "session_recency_weight" not in (result.route.metadata or {})

    def test_backward_compat_no_session_routing(self):
        """enable_session_routing=False (default) has no session store."""
        router = AdaptiveRouter(storage=None)
        assert router._session_store is None
        router.add_route(Route(name="billing", utterances=["billing query"]))
        router.durable_barrier()
        result = router.match("billing query")
        assert result.matched is True
        assert result.decision_reason == DecisionReason.MATCHED

    def test_session_alpha_clamped(self):
        router = AdaptiveRouter(enable_session_routing=True, session_alpha=99.0)
        assert router.session_alpha <= 0.15


# ================================================================== #
# AdaptiveRouter: IBAC (Intent-Based Access Control)                  #
# ================================================================== #

class TestAdaptiveRouterIBAC:
    @pytest.fixture
    def ibac_router(self):
        router = AdaptiveRouter(storage=None)
        router.add_route(Route(
            name="billing_admin",
            utterances=["view all invoices", "export billing data", "bulk refund"],
            required_permissions=["billing:read", "billing:admin"],
        ))
        router.add_route(Route(
            name="profile_view",
            utterances=["view my profile", "show account info", "my details"],
            required_permissions=["profile:read"],
        ))
        router.add_route(Route(
            name="general_help",
            utterances=["help me please", "I need assistance", "support"],
        ))
        router.durable_barrier()
        return router

    def test_permitted_caller_matches(self, ibac_router):
        result = ibac_router.match(
            "view all invoices",
            caller_permissions={"billing:read", "billing:admin"},
        )
        assert result.matched is True
        assert result.route_name == "billing_admin"

    def test_underpermissioned_caller_denied(self, ibac_router):
        result = ibac_router.match(
            "view all invoices",
            caller_permissions={"billing:read"},  # missing billing:admin
        )
        assert result.decision_reason == DecisionReason.PERMISSION_DENIED
        assert result.matched is False

    def test_no_permissions_denied(self, ibac_router):
        result = ibac_router.match(
            "view all invoices",
            caller_permissions=set(),
        )
        assert result.decision_reason == DecisionReason.PERMISSION_DENIED

    def test_route_without_permissions_always_accessible(self, ibac_router):
        """Routes with no required_permissions should not be gated."""
        result = ibac_router.match(
            "I need assistance",
            caller_permissions=set(),
        )
        assert result.matched is True
        assert result.route_name == "general_help"

    def test_no_caller_permissions_arg_skips_ibac(self, ibac_router):
        """When caller_permissions is None, IBAC is not applied at all."""
        result = ibac_router.match("view all invoices")
        # IBAC skipped, route matched on semantic similarity
        assert result.matched is True

    def test_permission_denied_reason_value(self, ibac_router):
        result = ibac_router.match("view all invoices", caller_permissions=set())
        if result.decision_reason == DecisionReason.PERMISSION_DENIED:
            assert result.decision_reason.value == "permission_denied"
