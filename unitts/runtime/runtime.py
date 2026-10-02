from __future__ import annotations

from collections.abc import AsyncIterator

from unitts.contracts import AudioChunk, AudioResult, InferenceContext, LoadContext, SynthesisRequest
from unitts.runtime.registry import EngineRegistry
from unitts.runtime.scheduler import GlobalAdmissionScheduler
from unitts.runtime.worker import EngineWorker


class UniTTSRuntime:
    """Single-machine vertical slice with replaceable remote worker transport."""

    def __init__(self, registry: EngineRegistry) -> None:
        self.registry = registry
        self.scheduler = GlobalAdmissionScheduler(registry)
        self._workers: dict[str, EngineWorker] = {}

    async def _worker(self, engine_id: str) -> EngineWorker:
        worker = self._workers.get(engine_id)
        if worker is None:
            worker = EngineWorker(self.registry.create(engine_id))
            device = "auto" if self.registry.get(engine_id).hardware.accelerator == "cuda" else "cpu"
            await worker.load(LoadContext(replica_id=f"{engine_id}-local-0", device=device))
            self._workers[engine_id] = worker
        return worker

    async def generate(self, request: SynthesisRequest, *, tenant_id: str = "local") -> AudioResult:
        plan = self.scheduler.plan(request, tenant_id=tenant_id)
        try:
            worker = await self._worker(plan.engine_id)
            return await worker.generate(request, InferenceContext(request_id=request.request_id, tenant_id=tenant_id))
        finally:
            self.scheduler.complete(plan)

    async def stream(self, request: SynthesisRequest, *, tenant_id: str = "local") -> AsyncIterator[AudioChunk]:
        plan = self.scheduler.plan(request, tenant_id=tenant_id)
        try:
            worker = await self._worker(plan.engine_id)
            async for chunk in worker.stream(request, InferenceContext(request_id=request.request_id, tenant_id=tenant_id)):
                yield chunk
        finally:
            self.scheduler.complete(plan)

    async def close(self) -> None:
        for worker in self._workers.values():
            await worker.unload()
        self._workers.clear()
