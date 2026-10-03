from __future__ import annotations

import io
import math
import struct
import wave
from collections.abc import AsyncIterator

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
)
from unitts.engine_sdk import SpeechEngine


class DummyEngine(SpeechEngine):
    """Deterministic sine-wave engine for GPU-free end-to-end testing."""

    sample_rate = 16_000

    def __init__(self) -> None:
        self.loaded = False

    @classmethod
    def manifest(cls) -> EngineManifest:
        return EngineManifest(
            engine_id="dummy", family="Dummy", engine_version="1.0.0",
            protocol_versions=["1.0"], status=EngineStatus.READY,
            entrypoint="unitts.engines.dummy:DummyEngine", upstream_code="first-party",
            capabilities=EngineCapabilities(streaming=True, execution_strategy="simple", safe_pause_boundaries=True, supports_seed=True),
            license=LicenseInfo(code="MIT", weights="none", commercial_use="allowed"),
        )

    async def load(self, context: LoadContext) -> None:
        self.manifest().assert_protocol(context.protocol_version)
        self.loaded = True

    def _pcm(self, request: SynthesisRequest) -> bytes:
        samples = max(800, len(request.text) * 320)
        frequency = 220 + (request.seed or 0) % 220
        return b"".join(struct.pack("<h", int(2500 * math.sin(2 * math.pi * frequency * i / self.sample_rate))) for i in range(samples))

    async def generate(self, request: SynthesisRequest, context: InferenceContext) -> AudioResult:
        if not self.loaded:
            raise RuntimeError("dummy engine is not loaded")
        pcm = self._pcm(request)
        output = io.BytesIO()
        with wave.open(output, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(self.sample_rate)
            writer.writeframes(pcm)
        return AudioResult(
            request_id=request.request_id, data=output.getvalue(), sample_rate=self.sample_rate,
            encoding=AudioEncoding.WAV, duration_seconds=len(pcm) / 2 / self.sample_rate,
            engine_id="dummy", metadata={"deterministic": True},
        )

    async def stream(self, request: SynthesisRequest, context: InferenceContext) -> AsyncIterator[AudioChunk]:
        if not self.loaded:
            raise RuntimeError("dummy engine is not loaded")
        pcm = self._pcm(request)
        chunk_bytes = 1600
        sequence = 0
        for offset in range(0, len(pcm), chunk_bytes):
            data = pcm[offset:offset + chunk_bytes]
            yield AudioChunk(
                request_id=request.request_id, sequence=sequence, start_sample=offset // 2,
                sample_rate=self.sample_rate, encoding=AudioEncoding.PCM_S16LE,
                data=data, final=offset + chunk_bytes >= len(pcm),
            )
            sequence += 1

    async def unload(self) -> None:
        self.loaded = False

    def estimate_resources(self, request: SynthesisRequest) -> ResourceEstimate:
        return ResourceEstimate(host_memory_mb=1, estimated_seconds=len(request.text) / 1000)

    async def health(self) -> EngineHealth:
        return EngineHealth(healthy=self.loaded, state=ReplicaState.WARM if self.loaded else ReplicaState.UNLOADED)
