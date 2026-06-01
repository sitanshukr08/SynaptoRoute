import pytest
from synaptoroute.models import Route
from synaptoroute.exceptions import RouteNotFoundError

def test_route_creation():
    route = Route(name="greeting", utterances=["hello", "hi"])
    assert route.name == "greeting"
    assert route.utterances == ["hello", "hi"]
    assert route.threshold == 0.5
    assert route.metadata is None

def test_route_with_metadata_and_threshold():
    route = Route(
        name="goodbye", 
        utterances=["bye", "cya"], 
        threshold=0.5, 
        metadata={"category": "closing"}
    )
    assert route.threshold == 0.5
    assert route.metadata == {"category": "closing"}
