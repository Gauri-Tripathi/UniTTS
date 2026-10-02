"""Service-runtime adapter using the same managed worker as the Python SDK."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from unitts.contracts import (
    AudioChunk,
    AudioEncoding,
    AudioResult,
    EngineCapabilities,
    EngineHealth,
    EngineManifest,
    EngineStatus,
    InferenceContext,
    LicenseInfo,
    LoadContext,
    ReplicaState,
    ResourceEstimate,
    SynthesisRequest,
    UnsupportedCapabilityError,
)
from unitts.contracts.models import HardwareRequirements
from unitts.core.schemas import TTSRequest
from unitts.engine_sdk import SpeechEngine
from unitts.managed.indextts import MODEL_REPO, SOURCE_REVISION, UPSTREAM_URL
from unitts.providers.indextts import IndexTTSControls, IndexTTSProvider


class IndexTTSEngine(SpeechEngine):
    controls_model = IndexTTSControls

    def __init__(self) -> None:
        self._provider: IndexTTSProvider | None = None

    @classmethod
    def manifest(cls) -> EngineManifest:
        return EngineManifest(
            engine_id="indextts-2.5", family="IndexTTS-2.5", engine_version="0.1.0",
            protocol_versions=["1.0"], status=EngineStatus.EXPERIMENTAL,
            entrypoint="unitts.engines.indextts:IndexTTSEngine",
            upstream_code=f"{UPSTREAM_URL.removesuffix('.git')}/tree/{SOURCE_REVISION}",
            checkpoints=[MODEL_REPO],
            capabilities=EngineCapabilities(
                voice_cloning=True, languages=["en", "zh", "ja", "es", "ar"],
                streaming=False, execution_strategy="autoregressive",
            ),
            hardware=HardwareRequirements(accelerator="cuda", cuda_runtime="12.8 (upstream locked Torch build)"),
            license=LicenseInfo(code="LicenseRef-Bilibili-IndexTTS", weights="bilibili Model Use License Agreement", commercial_use="restricted",
                                notes="Upstream terms apply to separately downloaded code and weights; see upstream LICENSE."),
            limitations=["Experimental; real-GPU validation required", "WAV only; one local reference", "Streaming not implemented", "No guaranteed minimum VRAM profile"],
        )

    async def load(self, context: LoadContext) -> None:
        self.manifest().assert_protocol(context.protocol_version)
        settings: dict[str, Any] = dict(context.settings)
        if context.model_cache:
            settings.setdefault("cache_dir", context.model_cache)
        self._provider = IndexTTSProvider(device=context.device, **settings)

    async def generate(self, request: SynthesisRequest, context: InferenceContext) -> AudioResult:
        if self._provider is None:
            raise RuntimeError("IndexTTS engine is not loaded")
        self.validate_controls(request)
        if request.streaming or request.output_format != "wav":
            raise UnsupportedCapabilityError("IndexTTS supports non-streaming WAV output only")
        legacy = TTSRequest(
            text=request.text, language=request.language, voice=request.voice,
            reference_audio=[Path(request.reference.uri)] if request.reference else [],
            speed=request.speed, seed=request.seed, controls=request.controls,
            sample_rate=None,
        )
        result = await asyncio.to_thread(self._provider.synthesize, legacy)
        return AudioResult(
            request_id=request.request_id, data=result.audio, sample_rate=result.sample_rate or 22050,
            encoding=AudioEncoding.WAV, duration_seconds=result.duration_seconds,
            engine_id="indextts-2.5", metadata={**result.metadata, "warnings": result.warnings},
        )

    async def stream(self, request: SynthesisRequest, context: InferenceContext) -> AsyncIterator[AudioChunk]:
        raise UnsupportedCapabilityError("IndexTTS streaming is not implemented")
        yield

    async def unload(self) -> None:
        if self._provider:
            await asyncio.to_thread(self._provider.cleanup)
        self._provider = None

    def estimate_resources(self, request: SynthesisRequest) -> ResourceEstimate:
        return ResourceEstimate(token_count=max(1, len(request.text)))

    async def health(self) -> EngineHealth:
        worker = self._provider._worker if self._provider else None
        warm = worker is not None and worker.process.poll() is None
        return EngineHealth(
            healthy=self._provider is not None,
            state=ReplicaState.WARM if warm else ReplicaState.UNLOADED,
            message="Model loads on first synthesis" if self._provider and not warm else None,
        )
