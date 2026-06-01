from typing import Any, Optional
from langchain_core.runnables import Runnable, RunnableConfig

from synaptoroute.router import AdaptiveRouter

class SynaptoRouteChain(Runnable):
    """
    A LangChain Runnable that routes string inputs using SynaptoRoute.
    Returns the route name if matched, else 'default'.
    """
    
    def __init__(self, router: AdaptiveRouter):
        self.router = router

    def invoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        route = self.router(input)
        if route and hasattr(route, 'name'):
            return route.name
        return "default"
        
    async def ainvoke(self, input: str, config: Optional[RunnableConfig] = None, **kwargs: Any) -> str:
        route = await self.router.aquery(input)
        if route and hasattr(route, 'name'):
            return route.name
        return "default"
