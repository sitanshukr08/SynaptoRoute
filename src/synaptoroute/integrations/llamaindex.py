from typing import Sequence, Any

try:
    from llama_index.core.schema import QueryBundle
    from llama_index.core.selectors import BaseSelector, SelectorResult, SingleSelection
    from llama_index.core.tools.types import ToolMetadata
except ImportError:
    BaseSelector = object
    QueryBundle = Any
    SelectorResult = Any
    SingleSelection = Any
    ToolMetadata = Any

from synaptoroute.router import AdaptiveRouter

class SynaptoRouteSelector(BaseSelector):
    """
    A selector that uses SynaptoRoute's AdaptiveRouter to select a route.
    """
    router: Any

    def __init__(self, router: AdaptiveRouter):
        super().__init__()
        if BaseSelector is object:
            raise ImportError("llama_index is not installed. Please install it using `pip install llama-index-core`")
        self.router = router

    def _get_prompts(self) -> dict:
        return {}
    
    def _update_prompts(self, prompts: dict) -> None:
        pass

    def _select(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        result = self.router(query.query_str)
        route_name = result.name if result else "default"
        
        for i, choice in enumerate(choices):
            if choice.name == route_name:
                return SelectorResult(
                    selections=[SingleSelection(index=i, reason=f"Matched route: {route_name}")]
                )
        
        return SelectorResult(
            selections=[SingleSelection(index=0, reason=f"No matching choice found for route: {route_name}, falling back to choice 0")]
        )

    async def _aselect(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        result = await self.router.aquery(query.query_str)
        route_name = result.name if result else "default"
        
        for i, choice in enumerate(choices):
            if choice.name == route_name:
                return SelectorResult(
                    selections=[SingleSelection(index=i, reason=f"Matched route: {route_name}")]
                )
                
        return SelectorResult(
            selections=[SingleSelection(index=0, reason=f"No matching choice found for route: {route_name}, falling back to choice 0")]
        )
