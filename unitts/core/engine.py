"""Main UniTTS engine."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any

from unitts.core.capabilities import Capabilities, ProviderType
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
        """Validate the replacement before releasing the current provider."""

        provider_cls = ProviderRegistry.get(provider)
        options = provider_options(provider, kwargs)
        min_vram_gb = options.pop("min_vram_gb", None)
        preference = self.device_preference
        if preference == "auto" and provider_cls.capabilities.provider_type in {ProviderType.API, ProviderType.CPU}:
            preference = "cpu"
        # Managed providers must inspect hardware in their own interpreter.
        device = preference if provider_cls.manages_device else resolve_device(preference, min_vram_gb=min_vram_gb)
        logger.info("Configuring provider %s on %s", provider, device)
        replacement = provider_cls(device=device, **options)
        if self.provider is not None:
            try:
                self.provider.cleanup()
            except BaseException:
                replacement.cleanup()
                raise
        self.provider = replacement
        self.provider_name = provider

    def _active_provider(self) -> BaseProvider:
        if self.provider is None:
            raise RuntimeError("No provider loaded")
        return self.provider

    @property
    def capabilities(self) -> Capabilities:
        """Capabilities of the configured provider, safe for callers to inspect."""
        return self._active_provider().capabilities.model_copy(deep=True)

    def setup(self) -> dict[str, Any]:
        """Prepare the active provider through the same entrypoint as synthesis."""
        return self._active_provider().setup()

    def status(self) -> dict[str, Any]:
        """Inspect the active provider without downloads or model loading."""
        return self._active_provider().status()

    def synthesize(self, text: str | TTSRequest, **kwargs: Any) -> TTSResponse:
        """Synthesize speech from text or a prebuilt request."""

        provider = self._active_provider()
        request = text if isinstance(text, TTSRequest) else TTSRequest(text=text, **kwargs)
        started = perf_counter()
        response = provider.synthesize(request)
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

        return self._active_provider().synthesize_dialogue(request)

    def list_voices(self) -> list[dict[str, Any]]:
        """Return voices from the active provider."""

        return self._active_provider().list_voices()

    def cleanup(self) -> None:
        """Release active provider resources."""

        if self.provider is not None:
            self.provider.cleanup()
            self.provider = None

    def __enter__(self) -> "UniTTS":
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()
