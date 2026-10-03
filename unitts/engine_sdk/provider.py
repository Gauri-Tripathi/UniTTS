"""Thin service boundary: real speech uses the same SDK execution path everywhere."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import ClassVar

from pydantic import BaseModel

from unitts.contracts import (
    AudioChunk,
    AudioEncoding,
    AudioResult,
    EngineHealth,
    InferenceContext,
    LoadContext,
    ReplicaState,
    ResourceEstimate,
    SynthesisRequest,
    UnsupportedCapabilityError,
)
from unitts.core.engine import UniTTS
from unitts.core.provider import BaseProvider
from unitts.engine_sdk.engine import SpeechEngine


class ProviderEngine(SpeechEngine):
    """Service metadata stays engine-specific; execution and cleanup belong to UniTTS."""

    provider_class: ClassVar[type[BaseProvider]]

    def __init__(self) -> None:
        self._tts: UniTTS | None = None
        self._has_generated = False

    async def load(self, context: LoadContext) -> None:
        self.manifest().assert_protocol(context.protocol_version)
        await self.unload()
        settings = dict(context.settings)
        if context.model_cache:
            settings.setdefault("model_cache", context.model_cache)
        self._tts = UniTTS(provider=self.provider_class.name, device=context.device, **settings)

    def validate_controls(self, request: SynthesisRequest) -> BaseModel | dict[str, object]:
        controls_model = self.provider_class.controls_model
        if controls_model is None:
            return super().validate_controls(request)
        return controls_model.model_validate(request.controls)

    async def generate(self, request: SynthesisRequest, context: InferenceContext) -> AudioResult:
        if self._tts is None:
            raise RuntimeError("Engine is not loaded")
        if request.streaming:
            raise UnsupportedCapabilityError("Streaming is not implemented by the SDK")
        capabilities = self._tts.capabilities
        if request.output_format.value not in capabilities.output_formats:
            raise UnsupportedCapabilityError(f"Supported output formats: {', '.join(capabilities.output_formats)}")
        self.validate_controls(request)
        try:
            result = await asyncio.to_thread(self._tts.synthesize, request.to_tts_request())
        except BaseException:
            self._has_generated = False
            raise
        self._has_generated = True
        return AudioResult(
            request_id=request.request_id, data=result.audio,
            sample_rate=result.sample_rate or 24000, encoding=AudioEncoding(result.output_format.value),
            duration_seconds=result.duration_seconds, engine_id=self.manifest().engine_id,
            metadata={**result.metadata, "warnings": result.warnings},
        )

    async def stream(self, request: SynthesisRequest, context: InferenceContext) -> AsyncIterator[AudioChunk]:
        raise UnsupportedCapabilityError("Streaming is not implemented by the SDK")
        yield

    async def unload(self) -> None:
        if self._tts is not None:
            await asyncio.to_thread(self._tts.cleanup)
        self._tts = None
        self._has_generated = False

    def estimate_resources(self, request: SynthesisRequest) -> ResourceEstimate:
        memory = self.provider_class.capabilities.min_gpu_vram_gb or 0
        return ResourceEstimate(gpu_memory_mb=int(memory * 1024))

    async def health(self) -> EngineHealth:
        return EngineHealth(
            healthy=self._tts is not None,
            state=ReplicaState.WARM if self._has_generated else ReplicaState.UNLOADED,
            message="Model loads on first synthesis" if self._tts and not self._has_generated else None,
        )
