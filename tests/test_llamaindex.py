import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

pytest.importorskip("llama_index.core")

from llama_index.core.schema import QueryBundle
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.selectors import SelectorResult

from synaptoroute.integrations.llamaindex import SynaptoRouteSelector
from synaptoroute.models import Route

def test_synaptoroute_selector_select():
    mock_router = MagicMock()
    # Mock the returned object from query
    mock_route = Route(name="route_b", utterances=["test"])
    mock_router.return_value = mock_route
    
    selector = SynaptoRouteSelector(router=mock_router)
    
    choices = [
        ToolMetadata(name="route_a", description="Route A"),
        ToolMetadata(name="route_b", description="Route B"),
    ]
    query = QueryBundle(query_str="test query")
    
    result = selector._select(choices, query)
    
    assert isinstance(result, SelectorResult)
    assert len(result.selections) == 1
    assert result.selections[0].index == 1
    mock_router.assert_called_once_with("test query")

@pytest.mark.asyncio
async def test_synaptoroute_selector_aselect():
    mock_router = MagicMock()
    mock_result = Mock()
    mock_result.name = "route_a"
    mock_router.aquery = AsyncMock(return_value=mock_result)
    
    selector = SynaptoRouteSelector(router=mock_router)
    
    choices = [
        ToolMetadata(name="route_a", description="Route A"),
        ToolMetadata(name="route_b", description="Route B"),
    ]
    query = QueryBundle(query_str="test async query")
    
    result = await selector._aselect(choices, query)
    
    assert isinstance(result, SelectorResult)
    assert len(result.selections) == 1
    assert result.selections[0].index == 0
    mock_router.aquery.assert_called_once_with("test async query")

def test_synaptoroute_selector_no_match():
    mock_router = MagicMock()
    mock_result = Mock()
    mock_result.name = "route_c"
    mock_router.return_value = mock_result
    
    selector = SynaptoRouteSelector(router=mock_router)
    
    choices = [
        ToolMetadata(name="route_a", description="Route A"),
        ToolMetadata(name="route_b", description="Route B"),
    ]
    query = QueryBundle(query_str="test query")
    
    with pytest.raises(ValueError, match="No matching choice found for route: route_c"):
        selector._select(choices, query)
