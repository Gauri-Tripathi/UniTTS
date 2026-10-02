"""Versioned JSON-lines control transport for one engine per OS process.

Large audio currently uses base64 in the response. The envelope deliberately keeps
transport descriptors open for a later shared-memory or gRPC implementation.
"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Any

from unitts.contracts import InferenceContext, LoadContext, ReplicaState, SynthesisRequest
from unitts.runtime.worker import EngineWorker
from unitts.services.api import build_registry
from unitts.workers.protocol import serve as serve_protocol


async def serve() -> None:
    engine_id = os.environ.get("UNITTS_ENGINE_ID", "dummy")
    replica_id = os.environ.get("UNITTS_REPLICA_ID", f"{engine_id}-0")
    registry = build_registry()
    worker = EngineWorker(registry.create(engine_id))

    async def dispatch(envelope: dict[str, Any]) -> dict[str, Any]:
        operation = envelope.get("operation")
        if operation == "load":
            await worker.load(LoadContext.model_validate(envelope.get("context", {
                "replica_id": replica_id, "device": os.environ.get("UNITTS_DEVICE", "auto"),
            })))
            return {"loaded": True}
        if operation not in {"generate", "health", "unload"}:
            raise ValueError(f"Unsupported operation: {operation}")
        if worker.state is ReplicaState.UNLOADED and operation != "unload":
            await worker.load(LoadContext(replica_id=replica_id, device=os.environ.get("UNITTS_DEVICE", "auto")))
        if operation == "generate":
            request = SynthesisRequest.model_validate(envelope["request"])
            result = await worker.generate(request, InferenceContext(request_id=request.request_id, tenant_id=envelope.get("tenant_id", "local")))
            payload: dict[str, Any] = result.model_dump(exclude={"data"}, mode="json")
            payload["audio_base64"] = base64.b64encode(result.data).decode("ascii")
            return payload
        if operation == "health":
            return (await worker.health()).model_dump(mode="json")
        await worker.unload()
        return {"unloaded": True}

    try:
        await serve_protocol(dispatch)
    finally:
        await worker.unload()


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
