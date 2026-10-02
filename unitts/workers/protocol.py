"""Shared JSON-lines protocol. Standard library only, including in model environments."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any
from uuid import uuid4

PROTOCOL_VERSION = "1.0"


def request_message(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "protocol_version": PROTOCOL_VERSION, "message_id": uuid4().hex}


def read_response(line: str, message_id: str) -> dict[str, Any]:
    response = json.loads(line)
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported worker protocol version")
    if response.get("message_id") != message_id:
        raise ValueError("Worker response does not match the current request")
    if not response.get("ok"):
        error = response.get("error", {})
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise RuntimeError(message)
    result = response["result"]
    if not isinstance(result, dict):
        raise ValueError("Worker result must be an object")
    return result


async def serve(handler: Callable[[dict[str, Any]], Any], *, stop_on_error: bool = False) -> None:
    """Dispatch sync/async handlers and isolate protocol output from model logs."""
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    stdout_fd = sys.stdout.fileno()
    os.dup2(sys.stderr.fileno(), stdout_fd)
    try:
        while line := await asyncio.to_thread(sys.stdin.readline):
            request: dict[str, Any] = {}
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    request = {}
                    raise ValueError("Worker request must be an object")
                if request.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("Unsupported worker protocol version")
                result = handler(request)
                if inspect.isawaitable(result):
                    result = await result
                response = {"ok": True, "result": result}
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                error = exc.as_dict() if hasattr(exc, "as_dict") else {"code": "worker_error", "message": f"{type(exc).__name__}: {exc}"}
                response = {"ok": False, "error": error}
            response.update(protocol_version=PROTOCOL_VERSION, message_id=request.get("message_id"))
            protocol.write(json.dumps(response) + "\n")
            if stop_on_error and not response["ok"]:
                break
    finally:
        sys.stdout.flush()
        os.dup2(protocol.fileno(), stdout_fd)
        protocol.close()
