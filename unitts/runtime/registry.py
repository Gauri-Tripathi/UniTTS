from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from unitts.contracts import EngineManifest
from unitts.engine_sdk import SpeechEngine


class EngineRegistry:
    """Manifest-first registry that never imports model code during discovery."""

    def __init__(self) -> None:
        self._manifests: dict[str, EngineManifest] = {}
        self._factories: dict[str, type[SpeechEngine]] = {}

    def register_manifest(self, manifest: EngineManifest) -> None:
        if manifest.engine_id in self._manifests:
            raise ValueError(f"Duplicate engine id: {manifest.engine_id}")
        self._manifests[manifest.engine_id] = manifest

    def register_factory(self, engine_id: str, factory: type[SpeechEngine]) -> None:
        manifest = factory.manifest()
        if manifest.engine_id != engine_id:
            raise ValueError(f"Factory engine id {manifest.engine_id} does not match {engine_id}")
        if engine_id not in self._manifests:
            self.register_manifest(manifest)
        else:
            # Executable code is authoritative; catalogue files must not override
            # the actual provider's capability declaration.
            self._manifests[engine_id] = manifest
        self._factories[engine_id] = factory

    def discover(self, roots: Iterable[Path]) -> None:
        for root in roots:
            for path in sorted(root.glob("*/manifest.yaml")):
                self.register_manifest(EngineManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))

    def get(self, engine_id: str) -> EngineManifest:
        try:
            return self._manifests[engine_id]
        except KeyError as exc:
            raise KeyError(f"Unknown engine: {engine_id}") from exc

    def list(self, *, public_only: bool = True) -> list[EngineManifest]:
        values: Iterable[EngineManifest] = self._manifests.values()
        if public_only:
            values = (item for item in values if not item.engine_id.startswith("research."))
        return sorted(values, key=lambda item: item.engine_id)

    def create(self, engine_id: str) -> SpeechEngine:
        factory = self._factories.get(engine_id)
        if factory is None:
            manifest = self.get(engine_id)
            if not manifest.entrypoint:
                raise RuntimeError(f"Engine {engine_id} has no runnable entrypoint (status={manifest.status})")
            module_name, object_name = manifest.entrypoint.split(":", 1)
            candidate: Any = getattr(importlib.import_module(module_name), object_name)
            factory = candidate
            self.register_factory(engine_id, factory)
        return factory()
