"""Main UniTTS engine."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from unitts.core.capabilities import ProviderType
from unitts.core.config import provider_options
from unitts.core.device import resolve_device
from unitts.core.provider import BaseProvider
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import DialogueRequest, TTSRequest, TTSResponse

logger = logging.getLogger(__name__)


class UniTTS:
    """Unified text-to-speech engine.

    Args:
        provider: Provider name registered in ``ProviderRegistry``.
        device: Device preference: ``auto``, ``cpu``, ``cuda``, or ``mps``.
        **kwargs: Provider-specific options such as ``api_key``.
    """

    def __init__(self, provider: str = "piper", device: str = "auto", **kwargs: Any) -> None:
        self.device_preference = device
        self.provider_name = ""
        self.provider: BaseProvider | None = None
        self.set_provider(provider, **kwargs)

    def set_provider(self, provider: str, **kwargs: Any) -> None:
        """Switch providers, cleaning up any loaded model first."""

        if self.provider is not None:
            self.provider.cleanup()
        provider_cls = ProviderRegistry.get(provider)
        options = provider_options(provider, kwargs)
        min_vram_gb = options.pop("min_vram_gb", None)
        preference = self.device_preference
        if preference == "auto" and provider_cls.capabilities.provider_type in {ProviderType.API, ProviderType.CPU}:
            preference = "cpu"
        # Managed providers must inspect hardware in their own interpreter.
        device = preference if getattr(provider_cls, "manages_device", False) else resolve_device(preference, min_vram_gb=min_vram_gb)
        logger.info("Loading provider %s on %s", provider, device)
        self.provider = provider_cls(device=device, **options)
        self.provider_name = provider

    def synthesize(self, text: str | TTSRequest, **kwargs: Any) -> TTSResponse:
        """Synthesize speech from text or a prebuilt request."""

        if self.provider is None:
            raise RuntimeError("No provider loaded")
        request = text if isinstance(text, TTSRequest) else TTSRequest(text=text, **kwargs)
        started = perf_counter()
        response = self.provider.synthesize(request)
        elapsed = perf_counter() - started
        response.metadata["synthesis_seconds"] = elapsed
        if response.duration_seconds > 0:
            response.metadata["real_time_factor"] = elapsed / response.duration_seconds
        return response

    def synthesize_to_file(self, text: str | TTSRequest, output: str | Path, **kwargs: Any) -> TTSResponse:
        """Synthesize speech and write audio bytes to a file."""

        target = Path(output).expanduser()
        if target.is_dir():
            raise ValueError(f"Output must be a filename, not a directory: {target}")
        if not target.parent.is_dir():
            raise ValueError(f"Output directory does not exist: {target.parent}")
        response = self.synthesize(text, **kwargs)
        target.write_bytes(response.audio)
        return response

    def synthesize_dialogue(self, request: DialogueRequest) -> TTSResponse:
        """Synthesize a multi-turn dialogue request."""

        if self.provider is None:
            raise RuntimeError("No provider loaded")
        return self.provider.synthesize_dialogue(request)

    def list_voices(self) -> list[dict[str, Any]]:
        """Return voices from the active provider."""

        if self.provider is None:
            raise RuntimeError("No provider loaded")
        return self.provider.list_voices()

    def cleanup(self) -> None:
        """Release active provider resources."""

        if self.provider is not None:
            self.provider.cleanup()
            self.provider = None

    def __enter__(self) -> "UniTTS":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()
