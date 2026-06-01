from typing import Sequence, Any

from llama_index.core.schema import QueryBundle
from llama_index.core.selectors import BaseSelector, SelectorResult, SingleSelection
from llama_index.core.tools.types import ToolMetadata

from synaptoroute.router import AdaptiveRouter

class SynaptoRouteSelector(BaseSelector):
    """
    A selector that uses SynaptoRoute's AdaptiveRouter to select a route.
    """
    router: Any

    def __init__(self, router: AdaptiveRouter):
        super().__init__()
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
        
        raise ValueError(f"No matching choice found for route: {route_name}")

    async def _aselect(self, choices: Sequence[ToolMetadata], query: QueryBundle) -> SelectorResult:
        result = await self.router.aquery(query.query_str)
        route_name = result.name if result else "default"
        
        for i, choice in enumerate(choices):
            if choice.name == route_name:
                return SelectorResult(
                    selections=[SingleSelection(index=i, reason=f"Matched route: {route_name}")]
                )
                
        raise ValueError(f"No matching choice found for route: {route_name}")
