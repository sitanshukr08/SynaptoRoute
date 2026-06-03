from typing import Any, Optional
try:
    from langchain_core.runnables import Runnable, RunnableConfig
except ImportError:
    Runnable = object
    RunnableConfig = Any

from synaptoroute.router import AdaptiveRouter

class SynaptoRouteChain(Runnable):
    """
    A LangChain Runnable that routes string inputs using SynaptoRoute.
    Returns the route name if matched, else 'default'.
    """
    
    def __init__(self, router: AdaptiveRouter):
        if Runnable is object:
            raise ImportError("langchain_core is not installed. Please install it using `pip install langchain-core`")
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
