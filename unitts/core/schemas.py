"""Pydantic schemas for UniTTS requests and responses."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unitts.core.capabilities import ProviderCapabilities as ProviderCapabilities


class Emotion(StrEnum):
    """Common cross-provider emotion vocabulary."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    EXCITED = "excited"
    CALM = "calm"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"
    SURPRISED = "surprised"
    SERIOUS = "serious"
    FRIENDLY = "friendly"
    EMPATHETIC = "empathetic"
    WHISPER = "whisper"
    SHOUTING = "shouting"


class SpeakingStyle(StrEnum):
    """Common speaking style vocabulary."""

    CONVERSATIONAL = "conversational"
    NARRATION = "narration"
    NEWSCAST = "newscast"
    CUSTOMER_SERVICE = "customer-service"
    ASSISTANT = "assistant"
    POETRY = "poetry"
    STORYTELLING = "storytelling"
    ADVERTISEMENT = "advertisement"


class OutputFormat(StrEnum):
    """Audio formats shared by requests, responses, and worker transports."""

    WAV = "wav"
    PCM_S16LE = "pcm_s16le"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    OPUS = "opus"


AudioFormat = OutputFormat


class TTSRequest(BaseModel):
    """Universal text-to-speech request.

    Providers may support only a subset of these fields. Unsupported fields are
    reported in ``TTSResponse.warnings`` and otherwise ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | None = None
    ssml: str | None = None
    language: str | None = "en"
    phonemes: str | None = None

    voice: str | None = None
    voice_description: str | None = None
    speaker_id: int | str | None = None
    gender: str | None = None
    age: str | None = None

    reference_audio: list[Path] = Field(default_factory=list)
    reference_text: str | None = None
    clone_enhance: bool = False

    speed: float | None = Field(default=1.0, ge=0.1, le=5.0)
    pitch: float | None = Field(default=0.0, ge=-20.0, le=20.0)
    volume: float | None = Field(default=1.0, ge=0.0, le=2.0)
    energy: float | None = Field(default=None, ge=0.0, le=1.0)
    pause_between_sentences: float | None = Field(default=None, ge=0.0)
    word_gap: float | None = Field(default=None, ge=0.0)

    emotion: Emotion | None = None
    emotion_intensity: float | None = Field(default=0.5, ge=0.0, le=1.0)
    speaking_style: SpeakingStyle | None = None
    expressiveness: float | None = Field(default=0.5, ge=0.0, le=1.0)

    add_breathing: bool = False
    laughter: bool = False
    non_verbal_sounds: list[str] = Field(default_factory=list)

    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    seed: int | None = None
    repetition_penalty: float | None = Field(default=1.0, ge=0.0)
    max_length_seconds: float | None = Field(default=None, gt=0.0)
    cfg_scale: float | None = Field(default=None, ge=0.0)

    # Namespaced controls interpreted and validated by the selected provider.
    controls: dict[str, Any] = Field(default_factory=dict)

    output_format: OutputFormat = OutputFormat.WAV
    sample_rate: int | None = Field(default=24000, gt=0)
    normalize: bool = True
    denoise: bool = False
    trim_silence: bool = True

    stream: bool = False
    chunk_size: int = Field(default=4096, gt=0)

    @field_validator("reference_audio", mode="before")
    @classmethod
    def _coerce_reference_audio(cls, value: object) -> list[Path]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            return [Path(value)]
        return [Path(item) for item in value]  # type: ignore[arg-type]

    @model_validator(mode="after")
    def _text_or_ssml_required(self) -> "TTSRequest":
        if not any(value and value.strip() for value in (self.text, self.ssml, self.phonemes)):
            raise ValueError("One of text, ssml, or phonemes is required")
        return self


class TTSResponse(BaseModel):
    """Synthesized audio plus metadata and graceful-degradation warnings."""

    model_config = ConfigDict(populate_by_name=True)

    audio: bytes
    sample_rate: int | None = 24000
    duration_seconds: float = 0.0
    output_format: OutputFormat = OutputFormat.WAV
    provider: str | None = None
    voice_used: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def audio_bytes(self) -> bytes:
        """Compatibility alias for callers that use ``audio_bytes``."""

        return self.audio

    @property
    def format(self) -> OutputFormat:
        """Compatibility alias for callers that use ``format``."""

        return self.output_format

    @model_validator(mode="after")
    def _fill_common_metadata(self) -> "TTSResponse":
        if self.provider is None:
            provider = self.metadata.get("provider")
            self.provider = str(provider) if provider is not None else None
        if self.voice_used is None:
            voice = self.metadata.get("voice")
            self.voice_used = str(voice) if voice is not None else None
        return self


class VoiceInfo(BaseModel):
    """Provider voice metadata."""

    id: str
    name: str
    language: list[str]
    gender: str | None = None
    preview_url: str | None = None
    supports_cloning: bool = False


class DialogueTurn(BaseModel):
    """Single speaker turn for dialogue synthesis."""

    speaker: str
    text: str
    emotion: Emotion | None = None
    voice: str | None = None


class DialogueRequest(BaseModel):
    """Multi-turn dialogue request."""

    turns: list[DialogueTurn]
    speaker_map: dict[str, str | dict[str, Any]] = Field(default_factory=dict)
    pause_between_turns: float = Field(default=0.5, ge=0.0)
    output_format: OutputFormat = OutputFormat.WAV
    sample_rate: int = 24000
