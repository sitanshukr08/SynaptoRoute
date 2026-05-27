import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Optional

from synaptoroute import AdaptiveRouter, Route, Encoder, SQLiteStorage

encoder = None
storage = None
router = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global encoder, storage, router
    
    encoder = Encoder()
    db_path = os.environ.get("DB_PATH", "routes.db")
    storage = SQLiteStorage(db_path)
    router = AdaptiveRouter(encoder=encoder, storage=storage)
    
    await router.start()
    
    yield
    
    if router:
        await router.stop()

app = FastAPI(lifespan=lifespan, title="Synaptoroute API Server")

class QueryRequest(BaseModel):
    query: str

class RouteResponse(BaseModel):
    intent: Optional[str]

class RouteCreateRequest(BaseModel):
    name: str
    utterances: List[str]

class UtteranceCreateRequest(BaseModel):
    utterance: str

@app.post("/route", response_model=RouteResponse)
async def route_query(request: QueryRequest = Body(...)):
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        route = await router.aquery(request.query)
        return RouteResponse(intent=route.name if route else None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/routes")
async def add_route(request: RouteCreateRequest = Body(...)):
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        route = Route(name=request.name, utterances=request.utterances)
        router.add_route(route)
        return {"status": "success", "route": request.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/routes/{name}/utterances")
async def add_utterance(name: str, request: UtteranceCreateRequest = Body(...)):
    if not router:
        raise HTTPException(status_code=503, detail="Router not initialized")
    try:
        router.add_utterance(name, request.utterance)
        return {"status": "success", "route": name, "utterance": request.utterance}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
