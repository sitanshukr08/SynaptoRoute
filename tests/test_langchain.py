import pytest
from unittest.mock import AsyncMock, MagicMock
from synaptoroute.models import Route

pytest.importorskip("langchain_core")

from synaptoroute.integrations.langchain import SynaptoRouteChain

def test_synaptoroute_chain_invoke_with_route():
    mock_router = MagicMock()
    mock_router.return_value = Route(name="math_route", utterances=["test"])
    
    chain = SynaptoRouteChain(router=mock_router)
    result = chain.invoke("What is 2+2?")
    
    mock_router.assert_called_once_with("What is 2+2?")
    assert result == "math_route"

def test_synaptoroute_chain_invoke_no_route():
    mock_router = MagicMock()
    mock_router.return_value = None
    
    chain = SynaptoRouteChain(router=mock_router)
    result = chain.invoke("Hello there")
    
    mock_router.assert_called_once_with("Hello there")
    assert result == "default"

@pytest.mark.asyncio
async def test_synaptoroute_chain_ainvoke_with_route():
    mock_router = MagicMock()
    mock_router.aquery = AsyncMock(return_value=Route(name="science_route", utterances=["test"]))
    
    chain = SynaptoRouteChain(router=mock_router)
    result = await chain.ainvoke("Tell me about atoms")
    
    mock_router.aquery.assert_awaited_once_with("Tell me about atoms")
    assert result == "science_route"

@pytest.mark.asyncio
async def test_synaptoroute_chain_ainvoke_no_route():
    mock_router = MagicMock()
    mock_router.aquery = AsyncMock(return_value=None)
    
    chain = SynaptoRouteChain(router=mock_router)
    result = await chain.ainvoke("Random text")
    
    mock_router.aquery.assert_awaited_once_with("Random text")
    assert result == "default"
