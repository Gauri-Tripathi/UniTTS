"""IndexTTS-2.5 through a managed Python 3.11 process, including from notebooks."""

from __future__ import annotations

import base64
import logging
import threading
import weakref
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from unitts.core.capabilities import CapabilityMatrix, ProviderType
from unitts.core.provider import BaseProvider
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import TTSRequest, TTSResponse
from unitts.managed.indextts import WORKER_SCRIPT, IndexTTSEnvironment
from unitts.managed.process import WorkerProcess

logger = logging.getLogger(__name__)
Intensity = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class IndexTTSSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    cache_dir: Path | None = None
    model_dir: Path | None = None
    auto_install: bool = True
    setup_timeout: float = Field(default=1800, gt=0, le=86400)
    load_timeout: float = Field(default=600, gt=0, le=86400)
    timeout: float = Field(default=300, gt=0, le=86400)
    dtype: str = "auto"


class IndexTTSControls(BaseModel):
    """Optional upstream controls; common emotion/speed fields work without these."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    emo_audio_prompt: Path | None = None
    emo_alpha: float = Field(default=0.6, ge=0, le=1)
    emo_vector: list[Intensity] | None = Field(default=None, min_length=8, max_length=8)
    use_emo_text: bool = False
    emo_text: str | None = None
    use_random: bool = False
    max_mel_tokens: int = Field(default=1500, ge=50, le=3000)
    max_text_tokens_per_segment: int = Field(default=120, ge=20, le=200)
    interval_silence: int = Field(default=200, ge=0, le=2000)


@ProviderRegistry.register("index-tts")
@ProviderRegistry.register("indextts")
@ProviderRegistry.register("indextts-2.5")
class IndexTTSProvider(BaseProvider):
    name = "indextts-2.5"
    controls_model = IndexTTSControls
    manages_device = True  # Host Torch may be absent or incompatible (e.g. Colab).
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU, languages=["en", "zh", "ja", "es", "ar"],
        voices=False, voice_cloning=True, emotion=True, speed=True, generation_controls=True,
        streaming=False, output_formats=["wav"], requires_gpu=True, max_text_length=10000,
        notes="Managed IndexTTS-2.5 worker. Requires one local reference recording; CPU is opt-in.",
        execution_strategy="autoregressive", supports_seed=True,
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        # Honor the shared configuration's existing cache option.
        model_cache = kwargs.pop("model_cache", None)
        if "cache_dir" not in kwargs and model_cache is not None:
            kwargs["cache_dir"] = model_cache
        settings = IndexTTSSettings.model_validate(kwargs)
        kwargs = settings.model_dump()
        super().__init__(device=device, **kwargs)
        if device not in {"auto", "cuda", "cuda:0", "cpu"}:
            raise ValueError("IndexTTS device must be auto, cuda, cuda:0, or cpu")
        if kwargs.get("dtype", "auto") not in {"auto", "bfloat16", "float32"}:
            raise ValueError("IndexTTS-2.5 dtype must be auto, bfloat16, or float32")
        self.environment = IndexTTSEnvironment(kwargs.get("cache_dir"), kwargs.get("model_dir"))
        self._worker: WorkerProcess | None = None
        self._lock = threading.RLock()
        self._runtime_info: dict[str, Any] = {}
        self._finalizer: weakref.finalize | None = None

    def setup(self) -> dict[str, Any]:
        return self.environment.ensure(
            device=self.device, timeout=float(self.options.get("setup_timeout", 1800)),
            install=self.options.get("auto_install", True),
        )

    def status(self) -> dict[str, Any]:
        return {
            **self.environment.status(), "provider": self.name, "managed": True,
            "configured_device": self.device,
            "worker_running": self._worker is not None and self._worker.process.poll() is None,
        }

    def _start(self) -> None:
        start = perf_counter()
        self.setup()
        setup_seconds = perf_counter() - start
        logger.warning("Loading IndexTTS in its managed worker; log: %s", self.environment.install_dir / "worker.log")
        worker = WorkerProcess(
            [str(self.environment.python), "-I", str(WORKER_SCRIPT), "--source", str(self.environment.source)],
            cwd=self.environment.source, log_path=self.environment.install_dir / "worker.log",
            label="IndexTTS",
        )
        try:
            start = perf_counter()
            self._runtime_info = worker.call({
                "operation": "load", "device": self.device,
                "dtype": self.options.get("dtype", "auto"), "model_dir": str(self.environment.model_dir),
            }, timeout=float(self.options.get("load_timeout", 600)))
            self._runtime_info.update(setup_seconds=setup_seconds, model_load_seconds=perf_counter() - start)
        except BaseException:
            worker.close()
            raise
        self._worker = worker
        self._finalizer = weakref.finalize(self, worker.close)

    def _prepare(self, request: TTSRequest) -> tuple[dict[str, Any], list[str]]:
        if not request.text or not request.text.strip():
            raise ValueError("IndexTTS requires nonempty plain text")
        if len(request.text) > 10000:
            raise ValueError("IndexTTS text exceeds 10000 characters; split it into smaller requests")
        if request.output_format != "wav":
            raise ValueError("IndexTTS supports WAV output only")
        if request.stream:
            raise ValueError("IndexTTS streaming is not implemented; use synthesize()")
        if len(request.reference_audio) != 1:
            raise ValueError("IndexTTS requires exactly one reference_audio file (a clean 3–15s recording)")
        reference = request.reference_audio[0].expanduser().resolve()
        if not reference.is_file():
            raise ValueError(f"Reference audio file not found: {reference}")
        language = (request.language or "en").lower().split("-")[0]
        if language not in self.capabilities.languages:
            raise ValueError(f"Unsupported IndexTTS-2.5 language '{language}'; choose en, zh, ja, es, or ar")
        speed = request.speed if request.speed is not None else 1.0
        if not 0.5 <= speed <= 2.0:
            raise ValueError("IndexTTS speed must be between 0.5 and 2.0")
        if request.voice_description:
            raise ValueError("IndexTTS clones reference_audio; voice_description is not supported. For emotion descriptions use controls={'emo_text': '...'}.")
        if request.max_length_seconds is not None:
            raise ValueError("IndexTTS does not enforce exact output duration; use controls.max_mel_tokens and the provider timeout")
        controls = IndexTTSControls.model_validate(request.controls)
        sources = sum([controls.emo_audio_prompt is not None, controls.emo_vector is not None,
                       controls.use_emo_text or controls.emo_text is not None, request.emotion is not None])
        if sources > 1:
            raise ValueError("Choose one emotion source: emotion, emo_audio_prompt, emo_vector, or emotion text")
        options = controls.model_dump(exclude_none=True)
        if controls.emo_audio_prompt:
            emotion_path = controls.emo_audio_prompt.expanduser().resolve()
            if not emotion_path.is_file():
                raise ValueError(f"Emotion reference file not found: {emotion_path}")
            options["emo_audio_prompt"] = str(emotion_path)
        if controls.emo_text is not None:
            options["use_emo_text"] = True
        if request.emotion is not None:
            positions = {"happy": 0, "angry": 1, "sad": 2, "fearful": 3, "disgusted": 4, "surprised": 6, "calm": 7, "neutral": 7}
            if request.emotion not in positions:
                raise ValueError(f"Unsupported IndexTTS emotion '{request.emotion}'; supported: {', '.join(positions)}")
            vector = [0.0] * 8
            vector[positions[request.emotion]] = request.emotion_intensity if request.emotion_intensity is not None else 0.5
            options["emo_vector"] = vector
        options.update(text=request.text, spk_audio_prompt=str(reference), lang=language.upper(), duration_factor=1.0 / speed)
        # Preserve upstream sampling defaults unless the caller explicitly changes them.
        for field in ("temperature", "top_k", "top_p", "repetition_penalty"):
            value = getattr(request, field)
            if field in request.model_fields_set and value is not None:
                options[field] = value
        unsupported = self.capabilities.unsupported_request_fields(request)
        warnings = [f"IndexTTS ignores unsupported field: {field}" for field in unsupported]
        for field in ("voice", "reference_text", "speaker_id", "gender", "age", "cfg_scale", "pause_between_sentences", "word_gap"):
            if getattr(request, field) is not None:
                warnings.append(f"IndexTTS ignores unsupported field: {field}")
        return options, warnings

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        options, warnings = self._prepare(request)  # Validate before any install/download/load.
        with self._lock:
            cold_start = self._worker is None
            if self._worker is None:
                self._start()
            assert self._worker is not None
            try:
                result = self._worker.call({
                    "operation": "generate", "options": options, "seed": request.seed,
                    "sample_rate": request.sample_rate,
                }, timeout=float(self.options.get("timeout", 300)))
            except BaseException:
                self.cleanup()
                raise
            return TTSResponse(
                audio=base64.b64decode(result["audio_base64"], validate=True),
                sample_rate=result["sample_rate"], duration_seconds=result["duration_seconds"],
                warnings=warnings + result.get("warnings", []),
                metadata={
                    "provider": self.name, "model": "IndexTeam/IndexTTS-2.5", **self._runtime_info,
                    "cold_start": cold_start, "diagnostics": result.get("diagnostics", {}),
                },
            )

    def cleanup(self) -> None:
        with self._lock:
            if self._worker is not None:
                self._worker.close()
            self._worker = None
            if self._finalizer is not None:
                self._finalizer.detach()
                self._finalizer = None
