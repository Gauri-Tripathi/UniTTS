"""Base provider contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from unitts.core.capabilities import CapabilityMatrix
from unitts.core.schemas import DialogueRequest, TTSRequest, TTSResponse


class BaseProvider(ABC):
    """Abstract base class for all TTS providers."""

    name: str
    capabilities: CapabilityMatrix
    controls_model: ClassVar[type[BaseModel] | None] = None
    manages_device: ClassVar[bool] = False

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        self.device = device
        self.options = kwargs

    def setup(self) -> dict[str, Any]:
        """Prepare a managed environment when the provider supplies one."""
        return self.status()

    def status(self) -> dict[str, Any]:
        """Inspect configuration without loading a model or making network calls."""
        return {"provider": self.name, "managed": False, "configured_device": self.device}

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSResponse:
        """Generate speech audio for a request."""

    def synthesize_dialogue(self, request: DialogueRequest) -> TTSResponse:
        """Generate a dialogue by concatenating individual turns."""

        from unitts.utils.audio import merge_wav_responses

        responses: list[TTSResponse] = []
        warnings: list[str] = []
        for turn in request.turns:
            speaker_config = request.speaker_map.get(turn.speaker, turn.voice or turn.speaker)
            if isinstance(speaker_config, dict):
                config = dict(speaker_config)
                voice = config.pop("voice", turn.voice or turn.speaker)
                turn_request = TTSRequest(text=turn.text, voice=voice, emotion=turn.emotion, **config)
            else:
                turn_request = TTSRequest(text=turn.text, voice=speaker_config, emotion=turn.emotion)
            response = self.synthesize(turn_request)
            warnings.extend(response.warnings)
            responses.append(response)
        return merge_wav_responses(responses, pause_seconds=request.pause_between_turns, warnings=warnings)

    def list_voices(self) -> list[dict[str, Any]]:
        """Return voices known by the provider."""

        return []

    def cleanup(self) -> None:
        """Release provider resources."""
        return None
