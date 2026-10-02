"""Reproducible IndexTTS installation outside the caller's Python environment."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from unitts.managed.process import child_environment as child_environment

UPSTREAM_URL = "https://github.com/index-tts/index-tts.git"
SOURCE_REVISION = "d9e41aac89fd00b3d71497fddb287b7f24613712"
MODEL_REPO = "IndexTeam/IndexTTS-2.5"
MODEL_REVISION = "c39ce5ba981572cb187443877ff559dfb246ce63"
WORKER_SCRIPT = Path(__file__).resolve().parents[1] / "workers" / "indextts.py"
logger = logging.getLogger(__name__)


class IndexTTSEnvironment:
    def __init__(self, cache_dir: str | Path | None = None, model_dir: str | Path | None = None) -> None:
        default_root = os.environ.get("UNITTS_CACHE_DIR") or Path.home() / ".cache" / "unitts"
        self.root = Path(cache_dir or default_root).expanduser().resolve()
        self.install_dir = self.root / "engines" / "indextts-2.5" / SOURCE_REVISION[:12]
        self.source = self.install_dir / "source"
        self.python = self.source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        self.model_dir = Path(model_dir).expanduser().resolve() if model_dir else self.root / "models" / "indextts-2.5" / MODEL_REVISION[:12]
        self.external_model = model_dir is not None
        self.log_path = self.install_dir / "setup.log"

    def status(self) -> dict[str, Any]:
        try:
            marker = json.loads((self.install_dir / "installed.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            marker = {}
        return {
            "installed": self.python.is_file() and marker.get("source_revision") == SOURCE_REVISION,
            "python": str(self.python), "source": str(self.source),
            "source_revision": SOURCE_REVISION, "model_revision": MODEL_REVISION,
            "model_dir": str(self.model_dir), "config_found": (self.model_dir / "config.yaml").is_file(),
            "setup_log": str(self.log_path),
        }

    def _run(self, command: list[str], *, timeout: float, cwd: Path | None = None) -> None:
        self.install_dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as log:
            try:
                subprocess.run(command, cwd=cwd, env=child_environment(), stdout=log, stderr=log, check=True, timeout=timeout)
            except (OSError, subprocess.SubprocessError) as exc:
                log.flush()
                with self.log_path.open("rb") as reader:
                    reader.seek(max(0, self.log_path.stat().st_size - 3000))
                    tail = reader.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"IndexTTS setup failed. Details: {self.log_path}.\n{tail}") from exc

    def ensure(self, *, device: str = "auto", timeout: float = 1800, install: bool = True) -> dict[str, Any]:
        if self.external_model and not (self.model_dir / "config.yaml").is_file():
            raise ValueError(f"IndexTTS model_dir has no config.yaml: {self.model_dir}")
        if not self.status()["installed"]:
            if not install:
                raise RuntimeError("IndexTTS is not set up. Run: unitts setup indextts")
            try:
                from filelock import FileLock
            except ImportError as exc:
                raise ImportError('Install managed setup support: python -m pip install "unitts[indextts]"') from exc
            self.install_dir.mkdir(parents=True, exist_ok=True)
            with FileLock(str(self.install_dir / "install.lock"), timeout=timeout):
                if not self.status()["installed"]:
                    self._install(timeout)
        # Run preflight inside the managed interpreter, never import host Torch.
        self._run([str(self.python), "-I", str(WORKER_SCRIPT), "--source", str(self.source), "--check", device], timeout=60)
        if not self.external_model:
            try:
                from filelock import FileLock
            except ImportError as exc:
                raise ImportError('Install managed setup support: python -m pip install "unitts[indextts]"') from exc
            with FileLock(str(self.install_dir / "install.lock"), timeout=timeout):
                complete = self.model_dir / ".unitts-complete"
                if not complete.is_file() or not (self.model_dir / "config.yaml").is_file():
                    logger.warning("Downloading IndexTTS weights and auxiliary models; details: %s", self.log_path)
                    self._run([
                        str(self.python), "-I", str(WORKER_SCRIPT), "--source", str(self.source),
                        "--download", str(self.model_dir), "--revision", MODEL_REVISION,
                    ], timeout=timeout)
                    complete.write_text(MODEL_REVISION, encoding="utf-8")
        return self.status()

    def _install(self, timeout: float) -> None:
        uv = shutil.which("uv")
        if uv:
            uv_command = [uv]
        elif importlib.util.find_spec("uv"):
            uv_command = [sys.executable, "-m", "uv"]
        else:
            raise ImportError('Install managed setup support: python -m pip install "unitts[indextts]"')
        if not shutil.which("git"):
            raise RuntimeError("IndexTTS setup needs Git. Install Git, then rerun: unitts setup indextts")
        logger.warning("Preparing isolated IndexTTS Python 3.11 environment; details: %s", self.log_path)
        if not (self.source / ".git").is_dir():
            self._run(["git", "init", str(self.source)], timeout=60)
        self._run(["git", "-C", str(self.source), "fetch", "--depth", "1", UPSTREAM_URL, SOURCE_REVISION], timeout=timeout)
        self._run(["git", "-C", str(self.source), "checkout", "--detach", SOURCE_REVISION], timeout=60)
        self._run(uv_command + ["python", "install", "3.11"], timeout=timeout)
        self._run(uv_command + ["sync", "--project", str(self.source), "--python", "3.11", "--frozen", "--no-dev"], timeout=timeout)
        # Validate imports before marking the installation reusable.
        self._run([str(self.python), "-I", str(WORKER_SCRIPT), "--source", str(self.source), "--check", "imports"], timeout=120)
        (self.install_dir / "installed.json").write_text(json.dumps({"source_revision": SOURCE_REVISION}), encoding="utf-8")
