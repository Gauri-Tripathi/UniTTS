"""Pydantic models used on the versioned engine-worker wire protocol."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from unitts.core.capabilities import EngineCapabilities as EngineCapabilities
from unitts.core.schemas import OutputFormat, TTSRequest
from unitts.workers.protocol import PROTOCOL_VERSION as PROTOCOL_VERSION

AudioFormat = OutputFormat
AudioEncoding = OutputFormat


class GenerationMode(StrEnum):
    REALTIME = "realtime"
    ASYNCHRONOUS = "asynchronous"


class EngineStatus(StrEnum):
    PLANNED = "planned"
    EXPERIMENTAL = "experimental"
    READY = "ready"
    RESEARCH_ONLY = "research_only"
    LICENSE_BLOCKED = "license_blocked"
    DEPRECATED = "deprecated"


class ReplicaState(StrEnum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    WARM = "WARM"
    ACTIVE = "ACTIVE"
    DRAINING = "DRAINING"
    EVICTING = "EVICTING"
    FAILED = "FAILED"


class VoiceReference(BaseModel):
    uri: str
    transcript: str | None = None
    content_type: str | None = None
    sha256: str | None = None


class SynthesisRequest(TTSRequest):
    """Service envelope around the SDK request; synthesis fields are inherited."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str = Field(min_length=1)
    engine_id: str | None = None
    reference: VoiceReference | None = None
    stream: bool = Field(default=False, validation_alias=AliasChoices("streaming", "stream"), serialization_alias="streaming")
    mode: GenerationMode = GenerationMode.REALTIME

    @property
    def streaming(self) -> bool:
        return self.stream

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        updates = dict(update or {})
        if "streaming" in updates:
            updates["stream"] = updates.pop("streaming")
        return super().model_copy(update=updates, deep=deep)

    @model_validator(mode="after")
    def _reference_shape(self) -> Self:
        if self.reference and self.reference_audio:
            raise ValueError("Use reference or reference_audio, not both")
        if self.reference and self.reference.transcript is not None and self.reference_text is not None:
            if self.reference.transcript != self.reference_text:
                raise ValueError("Conflicting reference transcripts")
        return self

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

    def to_tts_request(self) -> TTSRequest:
        """Translate the HTTP envelope once; providers only execute TTSRequest."""
        values = self.model_dump(include=set(TTSRequest.model_fields), exclude_unset=True)
        values.update(text=self.text, output_format=self.output_format.value, stream=self.streaming)
        if self.reference:
            values["reference_audio"] = self.reference.uri
            if self.reference.transcript is not None:
                values["reference_text"] = self.reference.transcript
        return TTSRequest.model_validate(values)


class AudioResult(BaseModel):
    request_id: str
    data: bytes
    sample_rate: int = Field(gt=0)
    encoding: AudioEncoding
    duration_seconds: float = Field(ge=0.0)
    engine_id: str
    artifact_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioChunk(BaseModel):
    request_id: str
    sequence: int = Field(ge=0)
    start_sample: int = Field(ge=0)
    sample_rate: int = Field(gt=0)
    encoding: AudioEncoding
    data: bytes
    final: bool = False


class LicenseInfo(BaseModel):
    code: str
    weights: str
    commercial_use: Literal["allowed", "restricted", "verify", "unknown"] = "unknown"
    notes: str | None = None


class HardwareRequirements(BaseModel):
    accelerator: Literal["cpu", "cuda", "remote", "any"] = "any"
    min_vram_gb: float = Field(default=0.0, ge=0.0)
    min_driver: str | None = None
    cuda_runtime: str | None = None
    compute_capability: str | None = None
    peak_load_memory_mb: int = Field(default=0, ge=0)


class EngineManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    engine_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    family: str
    engine_version: str
    protocol_versions: list[str]
    status: EngineStatus
    entrypoint: str | None = None
    upstream_code: str
    checkpoints: list[str] = Field(default_factory=list)
    capabilities: EngineCapabilities
    hardware: HardwareRequirements = Field(default_factory=HardwareRequirements)
    license: LicenseInfo
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_supported_protocol(self) -> "EngineManifest":
        if not self.protocol_versions:
            raise ValueError("at least one protocol version is required")
        if self.status is EngineStatus.READY and not self.entrypoint:
            raise ValueError("ready engines require an entrypoint")
        return self

    def assert_protocol(self, protocol_version: str) -> None:
        from unitts.contracts.errors import ProtocolVersionError

        if protocol_version not in self.protocol_versions:
            raise ProtocolVersionError(
                f"Engine {self.engine_id} does not support protocol {protocol_version}",
                details={"supported": self.protocol_versions},
            )


class ResourceEstimate(BaseModel):
    gpu_memory_mb: int = Field(default=0, ge=0)
    host_memory_mb: int = Field(default=0, ge=0)
    estimated_seconds: float = Field(default=0.0, ge=0.0)
    token_count: int | None = Field(default=None, ge=0)
    frame_count: int | None = Field(default=None, ge=0)


class ExecutionPlan(BaseModel):
    request_id: str
    engine_id: str
    replica_id: str
    strategy: str
    tenant_id: str
    priority: int = 0
    protocol_version: str = PROTOCOL_VERSION
    resource_estimate: ResourceEstimate


class LoadContext(BaseModel):
    replica_id: str
    device: str = "cpu"
    protocol_version: str = PROTOCOL_VERSION
    model_cache: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class InferenceContext(BaseModel):
    request_id: str
    tenant_id: str = "local"
    deadline_monotonic: float | None = None
    trace_id: str | None = None


class EngineHealth(BaseModel):
    healthy: bool
    state: ReplicaState
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
