from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from unitts.contracts import EngineError, SynthesisRequest
from unitts.engines import DummyEngine
from unitts.runtime import UniTTSRuntime
from unitts.runtime.registry import EngineRegistry


def build_registry() -> EngineRegistry:
    registry = EngineRegistry()
    root = Path(__file__).resolve().parents[2] / "engines"
    if root.exists():
        registry.discover([root])
    else:
        registry.register_manifest(DummyEngine.manifest())
    registry.register_factory("dummy", DummyEngine)
    from unitts.engines.indextts import IndexTTSEngine
    registry.register_factory("indextts-2.5", IndexTTSEngine)
    return registry


@dataclass
class Job:
    job_id: str
    request: SynthesisRequest
    status: str = "queued"
    result: Any = None
    error: dict[str, Any] | None = None
    task: asyncio.Task[Any] | None = field(default=None, repr=False)


class JobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}

    def create(self, request: SynthesisRequest) -> Job:
        job = Job(job_id=str(uuid4()), request=request)
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError("generation not found") from exc


def create_app() -> Any:
    """Create the optional FastAPI service without making FastAPI a Core dependency."""
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import Response
    except ImportError as exc:
        raise ImportError("Install UniTTS with the 'service' extra to run the API") from exc

    registry = build_registry()
    runtime = UniTTSRuntime(registry)
    store = JobStore()

    @asynccontextmanager
    async def lifespan(_app: Any):
        yield
        await runtime.close()

    app = FastAPI(title="UniTTS", version="1.0", lifespan=lifespan)
    app.state.runtime = runtime
    app.state.registry = registry
    app.state.jobs = store

    async def run_job(job: Job) -> None:
        job.status = "running"
        try:
            job.result = await runtime.generate(job.request)
            job.status = "succeeded"
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except EngineError as exc:
            job.error = exc.as_dict()
            job.status = "failed"
        except Exception as exc:
            job.error = {"code": "internal_error", "message": str(exc)}
            job.status = "failed"

    def payload(job: Job) -> dict[str, Any]:
        result = None
        if job.result is not None:
            result = job.result.model_dump(exclude={"data"}, mode="json")
            result["audio_base64"] = base64.b64encode(job.result.data).decode("ascii")
        return {"job_id": job.job_id, "status": job.status, "result": result, "error": job.error}

    @app.post("/v1/speech/generations", status_code=202)
    async def create_generation(request: SynthesisRequest) -> dict[str, Any]:
        job = store.create(request)
        job.task = asyncio.create_task(run_job(job))
        return payload(job)

    @app.get("/v1/speech/generations/{job_id}")
    async def get_generation(job_id: str) -> dict[str, Any]:
        try:
            return payload(store.get(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/v1/speech/generations/{job_id}", status_code=202)
    async def cancel_generation(job_id: str) -> dict[str, Any]:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.task and not job.task.done():
            job.task.cancel()
        job.status = "cancelled"
        return payload(job)

    @app.websocket("/v1/speech/generations/{job_id}/stream")
    async def stream_generation(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        try:
            job = store.get(job_id)
            async for chunk in runtime.stream(job.request.model_copy(update={"streaming": True})):
                await websocket.send_json(chunk.model_dump(exclude={"data"}, mode="json"))
                if chunk.data:
                    await websocket.send_bytes(chunk.data)
        except (KeyError, WebSocketDisconnect):
            return
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    @app.get("/v1/engines")
    async def list_engines() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in registry.list()]

    @app.get("/v1/engines/{engine_id}")
    async def get_engine(engine_id: str) -> dict[str, Any]:
        try:
            return registry.get(engine_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/audio/speech")
    async def openai_speech(body: dict[str, Any]) -> Any:
        request = SynthesisRequest(text=str(body.get("input", "")), engine_id=str(body.get("model", "dummy")), voice=body.get("voice"))
        result = await runtime.generate(request)
        return Response(content=result.data, media_type="audio/wav")

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install UniTTS with the 'service' extra") from exc
    uvicorn.run("unitts.services.api:create_app", factory=True, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
