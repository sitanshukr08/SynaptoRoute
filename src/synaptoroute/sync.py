import asyncio
import json
import uuid
try:
    import redis.asyncio as redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
from typing import Optional

class BaseSyncManager:
    async def start(self):
        pass

    async def stop(self):
        pass

    def broadcast(self, action: str, payload: dict, embeddings: bytes = None):
        pass

    def register(self, router):
        pass

class RedisSyncManager(BaseSyncManager):
    def __init__(self, redis_url: str, sync_worker_count: int = 4, sync_queue_size: int = 1000):
        super().__init__()
        self.sync_worker_count = sync_worker_count
        self.sync_queue_size = sync_queue_size
        if not HAS_REDIS:
            raise ImportError("redis is not installed. Please install it using 'pip install synaptoroute[redis]'.")
        self.redis_url = redis_url
        self.sender_id = str(uuid.uuid4())
        self.router = None
        self._pubsub = None
        self._redis_client = None
        self._listener_task: Optional[asyncio.Task] = None
        self._publisher_task: Optional[asyncio.Task] = None
        self._outbound_queue: Optional[asyncio.Queue] = None
        self._inbound_queue: Optional[asyncio.Queue] = None
        self._dispatch_workers: list[asyncio.Task] = []
        self._channel = "synaptoroute:sync"
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def register(self, router):
        self.router = router

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._outbound_queue = asyncio.Queue()
        self._inbound_queue = asyncio.Queue(maxsize=self.sync_queue_size)
        self._redis_client = redis.from_url(self.redis_url)
        self._pubsub = self._redis_client.pubsub()
        await self._pubsub.subscribe(self._channel)

        self._listener_task = self._loop.create_task(self._listener_loop())
        self._publisher_task = self._loop.create_task(self._publisher_loop())
        self._dispatch_workers = [
            self._loop.create_task(self._dispatch_worker_loop()) 
            for _ in range(self.sync_worker_count)
        ]
        
        self.broadcast("request_full_sync", {})

    async def stop(self):
        if self._outbound_queue:
            try:
                await asyncio.wait_for(self._outbound_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._publisher_task:
            self._publisher_task.cancel()
            try:
                await self._publisher_task
            except asyncio.CancelledError:
                pass
                
        for worker in self._dispatch_workers:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
            
        if self._pubsub:
            await self._pubsub.unsubscribe(self._channel)
            await self._pubsub.close()
            
        if self._redis_client:
            await self._redis_client.aclose()

    async def _listener_loop(self):
        while True:
            try:
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            sender_id = data.get("sender_id")
                            target_id = data.get("target_id")
                            if sender_id != self.sender_id:
                                if target_id is None or target_id == self.sender_id:
                                    await self._inbound_queue.put(data)
                        except Exception:
                            pass
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(1.0)
                try:
                    await self._pubsub.subscribe(self._channel)
                except Exception:
                    pass

    async def _publisher_loop(self):
        try:
            while True:
                msg = await self._outbound_queue.get()
                await self._redis_client.publish(self._channel, json.dumps(msg))
                self._outbound_queue.task_done()
        except asyncio.CancelledError:
            pass

    async def _dispatch_worker_loop(self):
        try:
            while True:
                data = await self._inbound_queue.get()
                try:
                    await asyncio.to_thread(self._dispatch, data)
                except Exception:
                    import traceback
                    traceback.print_exc()
                finally:
                    self._inbound_queue.task_done()
        except asyncio.CancelledError:
            pass

    def broadcast(self, action: str, payload: dict, embeddings: bytes = None, target_id: str = None):
        if self._outbound_queue is None or self._loop is None:
            return
            
        import base64
        if embeddings is not None:
            payload["_embeddings_b64"] = base64.b64encode(embeddings).decode('ascii')
            
        msg = {
            "sender_id": self.sender_id,
            "target_id": target_id,
            "action": action,
            "payload": payload,
            "encoder_model": getattr(
                getattr(self.router, 'encoder', None),
                'model_name', None
            )
        }
        
        # Thread-safe queue insertion
        self._loop.call_soon_threadsafe(self._outbound_queue.put_nowait, msg)

    def _dispatch(self, data: dict):
        if not self.router:
            return
            
        incoming_model = data.get("encoder_model")
        local_model = getattr(
            getattr(self.router, 'encoder', None),
            'model_name', None
        )
        if incoming_model and local_model and incoming_model != local_model:
            import logging
            logging.getLogger(__name__).warning(
                f"Rejecting sync message: sender used encoder "
                f"'{incoming_model}', local encoder is "
                f"'{local_model}'. Cluster encoder mismatch."
            )
            return

        import base64
        import numpy as np

        action = data.get("action")
        payload = data.get("payload", {})
        
        emb_b64 = payload.pop("_embeddings_b64", None)
        emb_bytes = base64.b64decode(emb_b64) if emb_b64 else None
        precomputed_embeddings = np.frombuffer(emb_bytes, dtype=np.float32) if emb_bytes else None
                
        if action == "request_full_sync":
            sender_id = data.get("sender_id")
            if sender_id:
                import time
                with self.router._route_map_lock:
                    routes_snapshot = list(self.router._route_map.values())
                
                for i, route in enumerate(routes_snapshot):
                    self.broadcast("add_route", route.model_dump(mode="json"), target_id=sender_id)
                    if i > 0 and i % 100 == 0:
                        time.sleep(0.01)
                    
        elif action == "add_route":
            payload.get("name")
            
            from synaptoroute.models import Route
            route = Route(**payload)
            # Reshape 1D array back to 2D based on number of utterances if we have embeddings
            if precomputed_embeddings is not None:
                precomputed_embeddings = precomputed_embeddings.reshape(len(route.utterances), -1)
            self.router.add_route(route, _broadcast=False, _precomputed_embeddings=precomputed_embeddings)
        elif action == "add_utterance":
            self.router.add_utterance(payload["route_name"], payload["utterance"], _broadcast=False, _precomputed_embedding=precomputed_embeddings)
        elif action == "delete_route":
            self.router.delete_route(payload["route_name"], _broadcast=False)
        elif action == "update_threshold":
            self.router.update_threshold(payload["route_name"], payload["threshold"], _broadcast=False)
