import pytest

from synaptoroute import AdaptiveRouter, DecisionReason, Route, RouterResult
from synaptoroute.storage import SQLiteStorage


def test_match_exposes_scored_unique_candidates(fake_encoder):
    router = AdaptiveRouter(fake_encoder, SQLiteStorage(":memory:"))
    router.add_route(Route(name="greeting", utterances=["hello", "hi"], threshold=0.5))
    router.add_route(Route(name="farewell", utterances=["bye"], threshold=0.5))

    result = router.match("hello")

    assert isinstance(result, RouterResult)
    assert result.route_name == "greeting"
    assert result.matched is True
    assert result.decision_reason is DecisionReason.MATCHED
    assert result.score == pytest.approx(1.0)
    assert [candidate.route_name for candidate in result.candidates].count("greeting") == 1
    assert result.candidates[0].passed_threshold is True
    router.close()


def test_match_reports_empty_and_below_threshold_rejections(fake_encoder):
    empty_router = AdaptiveRouter(fake_encoder, SQLiteStorage(":memory:"))
    empty_result = empty_router.match("hello")
    assert empty_result.route is None
    assert empty_result.decision_reason is DecisionReason.EMPTY_INDEX
    empty_router.close()

    router = AdaptiveRouter(fake_encoder, SQLiteStorage(":memory:"))
    router.add_route(Route(name="support", utterances=["help"], threshold=0.9))
    rejected = router.match("unrelated words")
    assert rejected.route is None
    assert rejected.score is not None
    assert rejected.decision_reason is DecisionReason.BELOW_THRESHOLD
    assert rejected.candidates[0].passed_threshold is False
    router.close()


@pytest.mark.asyncio
async def test_match_and_amatch_return_equivalent_margin_rejections(fake_encoder):
    router = AdaptiveRouter(fake_encoder, SQLiteStorage(":memory:"), margin=0.1)
    router.add_route(Route(name="first", utterances=["hello"], threshold=0.5))
    router.add_route(Route(name="second", utterances=["hi"], threshold=0.5))

    sync_result = router.match("hello")
    await router.start()
    async_result = await router.amatch("hello")

    assert sync_result.route is None
    assert async_result.route is None
    assert sync_result.decision_reason is DecisionReason.AMBIGUOUS_MARGIN
    assert async_result.decision_reason is DecisionReason.AMBIGUOUS_MARGIN
    assert async_result.score == pytest.approx(sync_result.score)
    assert async_result.margin == pytest.approx(sync_result.margin)
    assert await router.aquery("hello") is None
    await router.stop()


class SelectingReranker:
    def __init__(self, selected_name=None):
        self.selected_name = selected_name

    def rerank(self, query, candidates):
        del query
        for _, route in candidates:
            if route.name == self.selected_name:
                return route
        return None


def test_reranker_decisions_remain_observable(fake_encoder):
    storage = SQLiteStorage(":memory:")
    router = AdaptiveRouter(fake_encoder, storage, reranker=SelectingReranker("farewell"))
    router.add_route(Route(name="greeting", utterances=["hello"], threshold=0.5))
    router.add_route(Route(name="farewell", utterances=["bye"], threshold=0.5))

    accepted = router.match("hello")
    assert accepted.route_name == "farewell"
    assert accepted.decision_reason is DecisionReason.MATCHED_RERANKER

    router.reranker = SelectingReranker()
    rejected = router.match("hello")
    assert rejected.route is None
    assert rejected.decision_reason is DecisionReason.RERANKER_REJECTED
    router.close()


def test_router_result_serializes_for_prediction_artifacts(fake_encoder):
    router = AdaptiveRouter(fake_encoder, SQLiteStorage(":memory:"))
    router.add_route(Route(name="greeting", utterances=["hello"], threshold=0.5))

    payload = router.match("hello").model_dump(mode="json")

    assert payload["route"]["name"] == "greeting"
    assert payload["decision_reason"] == "matched"
    assert payload["candidates"][0]["route_name"] == "greeting"
    router.close()
