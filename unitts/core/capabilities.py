"""Provider capability declarations."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ProviderType(StrEnum):
    """High-level provider runtime category."""

    API = "api"
    CPU = "cpu"
    GPU = "gpu"


_LEGACY_NAMES = {
    "supports_streaming": "streaming", "supports_voice_cloning": "voice_cloning",
    "supports_reference_audio": "voice_cloning", "supports_emotion": "emotion",
    "supports_emotion_list": "emotions", "supports_speaking_style": "speaking_style",
    "supports_ssml": "ssml", "supports_speed": "speed", "supports_pitch": "pitch",
    "supports_voice_description": "voice_description", "supports_dialogue": "dialogue",
    "supports_non_verbal": "non_verbal", "supports_breathing": "breathing",
    "supported_languages": "languages", "supported_formats": "output_formats",
}


class Capabilities(BaseModel):
    """Single capability definition for providers, manifests, CLI, and service API.

    Legacy public names are accepted on input and read as aliases. Serialization
    always uses canonical names so consumers cannot observe conflicting flags.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    provider_type: ProviderType = ProviderType.CPU
    name: str | None = None
    languages: list[str] = Field(default_factory=list)
    voices: bool = True
    voice_cloning: bool = False
    streaming: bool = False
    ssml: bool = False
    emotion: bool = False
    emotions: list[str] = Field(default_factory=list)
    speaking_style: bool = False
    speed: bool = False
    pitch: bool = False
    volume: bool = False
    multi_speaker: bool = False
    dialogue: bool = False
    output_formats: list[str] = Field(default_factory=lambda: ["wav"])
    generation_controls: bool = False
    voice_description: bool = False
    non_verbal: bool = False
    breathing: bool = False
    requires_api_key: bool = False
    requires_gpu: bool = False
    min_gpu_vram_gb: float | None = None
    max_text_length: int | None = None
    notes: str | None = None
    execution_strategy: Literal["autoregressive", "flow_matching", "remote", "simple"] = "simple"
    supports_seed: bool = False
    safe_pause_boundaries: bool = False

    @model_validator(mode="before")
    @classmethod
    def _legacy_names(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for old, canonical in _LEGACY_NAMES.items():
            if old in result:
                if canonical in result and result[canonical] != result[old]:
                    raise ValueError(f"Conflicting capability values: {old} and {canonical}")
                result[canonical] = result.pop(old)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in _LEGACY_NAMES:
            return getattr(self, _LEGACY_NAMES[name])
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(_LEGACY_NAMES.get(name, name), value)

    def unsupported_request_fields(self, request: object) -> list[str]:
        """Return request feature names that this provider will ignore."""

        unsupported: list[str] = []
        checks = {
            "ssml": (self.ssml, None),
            "reference_audio": (self.voice_cloning, []),
            "emotion": (self.emotion, None),
            "speaking_style": (self.speaking_style, None),
            "voice_description": (self.voice_description, None),
            "add_breathing": (self.breathing, False),
            "laughter": (self.non_verbal, False),
            "non_verbal_sounds": (self.non_verbal, []),
            "speed": (self.speed, 1.0),
            "pitch": (self.pitch, 0.0),
            "volume": (self.volume, 1.0),
            "stream": (self.streaming, False),
        }
        for field_name, supported in checks.items():
            supported, default = supported
            value = getattr(request, field_name, None)
            if value not in (None, False, [], "", default) and not supported:
                unsupported.append(field_name)
        return unsupported


# Import compatibility only: these are the same class, not independent schemas.
CapabilityMatrix = Capabilities
ProviderCapabilities = Capabilities
EngineCapabilities = Capabilities
