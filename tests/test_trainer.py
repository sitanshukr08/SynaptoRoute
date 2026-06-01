import pytest
from unittest.mock import AsyncMock, MagicMock
from synaptoroute.router import AdaptiveRouter
from synaptoroute.trainer import SyntheticTuner, SyntheticResponse

@pytest.fixture
def mock_router():
    router = MagicMock(spec=AdaptiveRouter)
    return router

@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    # Mock beta.chat.completions.parse
    mock_parse = AsyncMock()
    
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.parsed = SyntheticResponse(
        positive=["pos1", "pos2"],
        negative=["neg1", "neg2"]
    )
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    
    mock_parse.return_value = mock_response
    client.beta.chat.completions.parse = mock_parse
    return client

@pytest.mark.asyncio
async def test_synthetic_tuner_tune_route(mock_router, mock_openai_client):
    tuner = SyntheticTuner(router=mock_router, client=mock_openai_client)
    
    await tuner.tune_route(
        route_name="test_route", 
        description="A test route",
        num_samples=2
    )
    
    # Check that OpenAI was called
    mock_openai_client.beta.chat.completions.parse.assert_called_once()
    kwargs = mock_openai_client.beta.chat.completions.parse.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["response_format"] == SyntheticResponse
    assert "test_route" in kwargs["messages"][1]["content"]
    assert "A test route" in kwargs["messages"][1]["content"]
    
    # Check that router.fit_thresholds was called with correct arguments
    mock_router.fit_thresholds.assert_called_once_with(
        ["pos1", "pos2", "neg1", "neg2"],
        ["test_route", "test_route", "_NEGATIVE_", "_NEGATIVE_"]
    )
