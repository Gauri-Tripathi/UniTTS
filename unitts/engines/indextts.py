"""IndexTTS service metadata; execution delegates to the common SDK adapter."""

from unitts.contracts import EngineManifest, EngineStatus, LicenseInfo
from unitts.contracts.models import HardwareRequirements
from unitts.engine_sdk.provider import ProviderEngine
from unitts.managed.indextts import MODEL_REPO, SOURCE_REVISION, UPSTREAM_URL
from unitts.providers.indextts import IndexTTSProvider


class IndexTTSEngine(ProviderEngine):
    provider_class = IndexTTSProvider

    @classmethod
    def manifest(cls) -> EngineManifest:
        return EngineManifest(
            engine_id="indextts-2.5", family="IndexTTS-2.5", engine_version="0.1.0",
            protocol_versions=["1.0"], status=EngineStatus.EXPERIMENTAL,
            entrypoint="unitts.engines.indextts:IndexTTSEngine",
            upstream_code=f"{UPSTREAM_URL.removesuffix('.git')}/tree/{SOURCE_REVISION}",
            checkpoints=[MODEL_REPO],
            capabilities=IndexTTSProvider.capabilities.model_copy(deep=True),
            hardware=HardwareRequirements(accelerator="cuda", cuda_runtime="12.8 (upstream locked Torch build)"),
            license=LicenseInfo(code="LicenseRef-Bilibili-IndexTTS", weights="bilibili Model Use License Agreement", commercial_use="restricted",
                                notes="Upstream terms apply to separately downloaded code and weights; see upstream LICENSE."),
            limitations=["Experimental; real-GPU validation required", "WAV only; one local reference", "Streaming not implemented", "No guaranteed minimum VRAM profile"],
        )
