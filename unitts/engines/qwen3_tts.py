"""Qwen service metadata; execution delegates to the common SDK adapter."""

from unitts.contracts import EngineManifest, EngineStatus, LicenseInfo
from unitts.contracts.models import HardwareRequirements
from unitts.engine_sdk.provider import ProviderEngine
from unitts.providers.open_weights import Qwen3TTSProvider


class Qwen3TTSEngine(ProviderEngine):
    provider_class = Qwen3TTSProvider

    @classmethod
    def manifest(cls) -> EngineManifest:
        return EngineManifest(
            engine_id="qwen3-tts", family="Qwen3-TTS", engine_version="0.1.0",
            protocol_versions=["1.0"], status=EngineStatus.EXPERIMENTAL,
            entrypoint="unitts.engines.qwen3_tts:Qwen3TTSEngine",
            upstream_code="https://github.com/QwenLM/Qwen3-TTS",
            checkpoints=["Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"],
            capabilities=Qwen3TTSProvider.capabilities.model_copy(deep=True),
            hardware=HardwareRequirements(accelerator="cuda"),
            license=LicenseInfo(code="Apache-2.0 (verify upstream version)", weights="checkpoint-specific; verify", commercial_use="verify"),
            limitations=["Requires an isolated qwen-tts/Torch environment", "Streaming adapter is not yet validated"],
        )
