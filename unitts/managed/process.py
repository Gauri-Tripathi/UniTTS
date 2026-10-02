"""Persistent JSON-line worker with bounded waits and explicit process cleanup."""

from __future__ import annotations

import atexit
import json
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

from unitts.managed.indextts import child_environment


class WorkerProcess:
    def __init__(self, command: list[str], *, cwd: Path, log_path: Path) -> None:
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command, cwd=cwd, env=child_environment(), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self._log, text=True, encoding="utf-8", bufsize=1,
            )
        except BaseException:
            self._log.close()
            raise
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.RLock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        atexit.register(self.close)

    def _read(self) -> None:
        try:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self._responses.put(line)
        finally:
            self._responses.put(None)

    def call(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        with self._lock:
            try:
                if self.process.poll() is not None:
                    raise RuntimeError("worker exited")
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps(payload) + "\n")
                self.process.stdin.flush()
                line = self._responses.get(timeout=timeout)
                if line is None:
                    raise RuntimeError("worker exited before responding")
                response = json.loads(line)
                if not response.get("ok"):
                    # A failed CUDA context must never be reused.
                    raise RuntimeError(response.get("error", "unknown worker error"))
                return response["result"]
            except Exception as exc:
                self.close()
                reason = f"timed out after {timeout:g}s" if isinstance(exc, queue.Empty) else str(exc)
                raise RuntimeError(f"IndexTTS {reason}. Worker stopped; next call starts a fresh worker. Log: {self.log_path}") from exc
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self._log.close()
            atexit.unregister(self.close)
