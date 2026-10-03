"""Open-weight local model providers.

All heavy model packages are imported lazily inside provider methods.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, ConfigDict, Field

from unitts.core.capabilities import CapabilityMatrix, ProviderType
from unitts.core.device import cleanup_torch
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import TTSRequest, TTSResponse
from unitts.providers.base_http import HttpProvider


@ProviderRegistry.register("kokoro")
class KokoroProvider(HttpProvider):
    """Kokoro local open-weight TTS provider."""

    name = "kokoro"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.CPU,
        languages=["en", "ja", "zh", "es", "fr", "hi", "it", "pt"],
        speed=True,
        output_formats=["wav"],
        notes="Uses the installed kokoro package API.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._pipeline: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        try:
            import soundfile as sf
            from kokoro import KPipeline
        except Exception as exc:
            raise ImportError("Install kokoro plus unitts[audio] to use the Kokoro provider") from exc
        lang_code = self.options.get("lang_code") or (request.language or "en")[:1].lower()
        if self._pipeline is None:
            self._pipeline = KPipeline(lang_code=lang_code)
        generator = self._pipeline(request.text or request.ssml or request.phonemes or "", voice=request.voice or self.options.get("voice", "af_heart"), speed=request.speed or 1.0)
        _graphemes, _phonemes, audio = next(generator)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        try:
            sf.write(output, audio, request.sample_rate or 24000)
            data = output.read_bytes()
        finally:
            output.unlink(missing_ok=True)
        return TTSResponse(audio=data, sample_rate=request.sample_rate or 24000, metadata={"provider": self.name}, warnings=warnings)

    def cleanup(self) -> None:
        self._pipeline = None
        cleanup_torch()


@ProviderRegistry.register("f5")
@ProviderRegistry.register("f5-tts")
class F5TTSProvider(HttpProvider):
    """F5-TTS open-weight voice cloning provider."""

    name = "f5-tts"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["en", "zh", "many"],
        voice_cloning=True,
        speed=True,
        output_formats=["wav"],
        generation_controls=True,
        notes="Requires the F5-TTS package; reference_text is needed for best cloning.",
        requires_gpu=True,
        min_gpu_vram_gb=8.0,
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._infer: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if not request.reference_audio:
            warnings.append("F5-TTS voice cloning works best with reference_audio")
        try:
            from f5_tts.api import F5TTS
        except Exception as exc:
            raise ImportError(
                "F5-TTS is not installed. Install it in this environment with "
                "'python -m pip install f5-tts'. See docs/gpu-providers.md."
            ) from exc
        if self._infer is None:
            self._infer = F5TTS(model=self.options.get("model", "F5TTS_v1_Base"), device=self.device)
        wav, sample_rate, _spect = self._infer.infer(
            ref_file=str(request.reference_audio[0]) if request.reference_audio else None,
            ref_text=request.reference_text or "",
            gen_text=request.text or request.ssml or request.phonemes or "",
            seed=request.seed,
        )
        return _write_float_wav(wav, sample_rate, {"provider": self.name}, warnings)

    def cleanup(self) -> None:
        self._infer = None
        cleanup_torch()


@ProviderRegistry.register("dia")
class DiaProvider(HttpProvider):
    """Dia open-weight dialogue and non-verbal speech provider."""

    name = "dia"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["en"],
        multi_speaker=True,
        dialogue=True,
        output_formats=["wav"],
        generation_controls=True,
        non_verbal=True,
        requires_gpu=True,
        min_gpu_vram_gb=10.0,
        notes="Supports [S1]/[S2] dialogue markup and non-verbal tokens such as [laugh].",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        try:
            from dia.model import Dia
        except Exception as exc:
            raise ImportError(
                "Dia is not installed. Install it from its source repository in a separate "
                "environment; see docs/gpu-providers.md."
            ) from exc
        if self._model is None:
            self._model = Dia.from_pretrained(self.options.get("model", "nari-labs/Dia-1.6B"), device=self.device)
        audio = self._model.generate(
            request.text or request.ssml or request.phonemes or "",
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 0.95,
            cfg_scale=request.cfg_scale or 3.0,
        )
        return _write_float_wav(audio, request.sample_rate or 44100, {"provider": self.name}, warnings)

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


@ProviderRegistry.register("xtts")
@ProviderRegistry.register("xtts-v2")
class XTTSV2Provider(HttpProvider):
    """Coqui XTTS v2 provider."""

    name = "xtts-v2"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"],
        voice_cloning=True,
        speed=True,
        output_formats=["wav"],
        requires_gpu=True,
        min_gpu_vram_gb=4.0,
        notes="Uses Coqui TTS model tts_models/multilingual/multi-dataset/xtts_v2.",
    )

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if not request.reference_audio:
            warnings.append("XTTS v2 voice cloning expects reference_audio; using provider default speaker if configured")
        try:
            from TTS.api import TTS
        except ModuleNotFoundError as exc:
            if exc.name == "torchaudio":
                raise ImportError(
                    "XTTS v2 requires torchaudio matching the installed torch build. "
                    "See docs/gpu-providers.md for the CUDA 12.4 environment setup."
                ) from exc
            raise ImportError(
                "Coqui TTS is not installed. Install it with "
                "'python -m pip install coqui-tts'. See docs/gpu-providers.md."
            ) from exc
        except ImportError as exc:
            if "DTensor" in str(exc) and "torch.distributed.tensor" in str(exc):
                raise ImportError(
                    "XTTS v2 has incompatible torch and transformers versions. Reinstall "
                    "the matching CUDA 12.4 torch==2.6.0 and torchaudio==2.6.0 pair from "
                    "docs/gpu-providers.md."
                ) from exc
            if "isin_mps_friendly" in str(exc) and "transformers.pytorch_utils" in str(exc):
                raise ImportError(
                    "XTTS v2 is incompatible with Transformers 5.1+. Install "
                    "transformers==4.57.6; see docs/gpu-providers.md."
                ) from exc
            raise ImportError(
                "Coqui TTS could not be imported. Check its Torch and Transformers "
                "dependencies in docs/gpu-providers.md."
            ) from exc
        except Exception as exc:
            raise ImportError(
                "Coqui TTS could not be imported. Check its Torch and Transformers "
                "dependencies in docs/gpu-providers.md."
            ) from exc
        if self._model is None:
            model_name = self.options.get("model", "tts_models/multilingual/multi-dataset/xtts_v2")
            self._model = TTS(model_name=model_name).to(self.device)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            output = Path(handle.name)
        try:
            kwargs: dict[str, Any] = {
                "text": request.text or request.ssml or request.phonemes or "",
                "file_path": str(output),
                "language": request.language or self.options.get("language", "en"),
            }
            if request.reference_audio:
                kwargs["speaker_wav"] = str(request.reference_audio[0])
            elif request.voice:
                kwargs["speaker"] = request.voice
            self._model.tts_to_file(**kwargs)
            data = output.read_bytes()
        finally:
            output.unlink(missing_ok=True)
        return TTSResponse(audio=data, sample_rate=request.sample_rate or 24000, metadata={"provider": self.name, "voice": request.voice}, warnings=warnings)

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


@ProviderRegistry.register("fish-speech")
@ProviderRegistry.register("fish-speech-local")
class FishSpeechProvider(HttpProvider):
    """Self-hosted Fish Speech provider.

    Fish Speech owns the model process and exposes a local HTTP API. Keeping that
    boundary avoids importing its heavyweight inference stack into UniTTS.
    """

    name = "fish-speech"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["auto"],
        voice_cloning=True,
        output_formats=["wav"],
        generation_controls=True,
        requires_gpu=True,
        notes="Connects to a self-hosted Fish Speech API; S2 requires 24 GB VRAM and emotion markers can be included in text.",
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        endpoint = str(self.options.get("endpoint", "http://127.0.0.1:8080/v1/tts"))
        payload: dict[str, Any] = {
            "text": request.text or request.ssml or request.phonemes or "",
        }
        if request.reference_text:
            payload["reference_text"] = request.reference_text
        if request.reference_audio:
            try:
                payload["reference_audio"] = base64.b64encode(request.reference_audio[0].read_bytes()).decode("ascii")
            except OSError as exc:
                raise ValueError(f"Could not read Fish Speech reference audio: {request.reference_audio[0]}") from exc
        parsed_endpoint = urlparse(endpoint)
        is_loopback = parsed_endpoint.hostname in {"127.0.0.1", "localhost", "::1"}
        try:
            with requests.Session() as session:
                # Cluster proxy variables must not intercept a local Fish Speech server.
                session.trust_env = not is_loopback
                response = session.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = exc.response.text[:500].strip() if exc.response is not None else ""
            suffix = f" Response: {detail}" if detail else ""
            raise RuntimeError(
                f"Fish Speech server returned HTTP {exc.response.status_code if exc.response is not None else 'error'}."
                f" Its model backend is not ready or failed to load.{suffix}"
            ) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                "Could not reach the Fish Speech server. Start the self-hosted API and "
                "configure endpoint=.../v1/tts; see docs/gpu-providers.md."
            ) from exc
        audio = _fish_speech_audio(response)
        return TTSResponse(
            audio=audio,
            sample_rate=request.sample_rate or 24000,
            metadata={"provider": self.name, "endpoint": endpoint},
            warnings=warnings,
        )


class QwenControls(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    voice_description: str | None = None


@ProviderRegistry.register("qwen3-tts")
@ProviderRegistry.register("qwen-tts")
class Qwen3TTSProvider(HttpProvider):
    """Qwen3-TTS provider.

    Uses the ``qwen_tts`` package (``pip install -U qwen-tts``) which exposes
    ``Qwen3TTSModel`` with three inference methods:
      * ``generate_voice_clone``  – reference-audio cloning
      * ``generate_voice_design`` – description-driven voice
      * ``generate_custom_voice`` – named speaker voice

    All three return ``(List[np.ndarray], sample_rate)``.
    """

    name = "qwen3-tts"
    controls_model = QwenControls
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.GPU,
        languages=["zh", "en", "de", "fr", "es", "it", "pt", "ja", "ko", "ar"],
        voice_cloning=True,
        streaming=False,
        voice_description=True,
        output_formats=["wav"],
        generation_controls=True,
        requires_gpu=True,
        min_gpu_vram_gb=8.0,
        notes="Supported features depend on the selected Qwen3-TTS model variant.",
        execution_strategy="autoregressive", supports_seed=True,
    )

    # Correct defaults for the qwen-tts PyPI package.
    _DEFAULT_MODULE = "qwen_tts"
    _DEFAULT_CLASS = "Qwen3TTSModel"
    _DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
    _DEFAULT_SPEAKER = "ryan"

    # The qwen_tts API requires FULL language names, not ISO codes.
    _LANG_MAP: dict[str, str] = {
        "en": "english", "zh": "chinese", "de": "german", "fr": "french",
        "es": "spanish", "it": "italian", "pt": "portuguese", "ja": "japanese",
        "ko": "korean", "ar": "arabic", "ru": "russian",
    }

    def __init__(self, *, device: str = "auto", **kwargs: Any) -> None:
        super().__init__(device=device, **kwargs)
        self._model: Any | None = None

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        if self._model is None:
            self._model = self._load_model()
        result = self._call_inference(request)
        # The qwen_tts API returns (List[np.ndarray], sample_rate).
        audio_arrays, sample_rate = _unpack_qwen_result(result, request.sample_rate or 24000)
        import numpy as np

        audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
        return _write_float_wav(audio, int(sample_rate), {"provider": self.name, "voice": request.voice}, warnings)

    def _load_model(self) -> Any:
        import importlib

        module_name = self.options.get("module", self._DEFAULT_MODULE)
        class_name = self.options.get("class_name", self._DEFAULT_CLASS)
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ImportError(
                f"Cannot import '{module_name}'. Install via: pip install -U qwen-tts"
            ) from exc
        model_cls = getattr(module, class_name, None)
        if model_cls is None:
            raise ImportError(
                f"Module '{module_name}' has no attribute '{class_name}'. "
                f"Available: {[a for a in dir(module) if not a.startswith('_')]}"
            )
        model_id = self.options.get("model", self._DEFAULT_MODEL)
        if hasattr(model_cls, "from_pretrained"):
            model = model_cls.from_pretrained(model_id)
        else:
            try:
                model = model_cls(model=model_id, device=self.device)
            except TypeError:
                model = model_cls(model=model_id)
        if self.device != "cpu" and hasattr(model, "to"):
            model = model.to(self.device)
        return model

    def _resolve_language(self, lang: str | None) -> str:
        """Map ISO language codes to the full names the qwen_tts API expects."""
        code = (lang or "en").lower().split("-")[0]  # e.g. "en-US" → "en"
        return self._LANG_MAP.get(code, code)  # pass through if already full

    def _call_inference(self, request: TTSRequest) -> Any:
        overrides = QwenControls.model_validate(request.controls).model_dump(exclude_unset=True, exclude_none=True)
        for field, value in overrides.items():
            if field in request.model_fields_set and getattr(request, field) != value:
                raise ValueError(f"Conflicting Qwen control: {field}")
        if overrides:
            request = TTSRequest.model_validate(request.model_dump(exclude_unset=True) | overrides)
        if request.seed is not None:
            import torch
            torch.manual_seed(request.seed)
        generation = {
            field: getattr(request, field) for field in ("temperature", "top_k", "top_p", "repetition_penalty")
            if field in request.model_fields_set and getattr(request, field) is not None
        }
        text = request.text or request.ssml or request.phonemes or ""
        language = self._resolve_language(request.language)
        model_id = str(self.options.get("model", self._DEFAULT_MODEL))
        is_custom_voice_model = "customvoice" in model_id.lower()

        # --- voice cloning path ---
        if request.reference_audio:
            if is_custom_voice_model:
                raise ValueError(
                    f"Qwen model '{model_id}' is a CustomVoice model and does not support "
                    "voice cloning. Configure a Qwen3-TTS model variant that supports "
                    "generate_voice_clone."
                )
            ref_audio = str(request.reference_audio[0])
            return self._model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=request.reference_text,
                **generation,
            )

        # --- voice design path (description-driven) ---
        instruct = request.voice_description or _voice_description_from_request(request)
        if instruct:
            if is_custom_voice_model:
                raise ValueError(
                    f"Qwen model '{model_id}' is a CustomVoice model and does not support "
                    "description-driven voice generation. Configure a Qwen3-TTS VoiceDesign "
                    "model variant instead."
                )
            try:
                return self._model.generate_voice_design(
                    text=text,
                    instruct=instruct,
                    language=language,
                    **generation,
                )
            except ValueError:
                # CustomVoice model variants don't support voice design;
                # fall through to custom_voice with a default speaker.
                pass

        # --- named speaker path ---
        speaker = request.voice or self.options.get("voice", self._DEFAULT_SPEAKER)
        return self._model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            **generation,
        )

    def cleanup(self) -> None:
        self._model = None
        cleanup_torch()


def _write_float_wav(audio: Any, sample_rate: int, metadata: dict[str, Any], warnings: list[str]) -> TTSResponse:
    try:
        import soundfile as sf
    except Exception as exc:
        raise ImportError("Install unitts[audio] for local open-weight audio output") from exc
    import tempfile

    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    elif hasattr(audio, "cpu") and hasattr(audio, "numpy"):
        audio = audio.cpu().numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        output = Path(handle.name)
    try:
        sf.write(output, audio, sample_rate)
        data = output.read_bytes()
    finally:
        output.unlink(missing_ok=True)
    return TTSResponse(audio=data, sample_rate=sample_rate, metadata=metadata, warnings=warnings)


def _fish_speech_audio(response: requests.Response) -> bytes:
    """Accept raw audio and the base64 JSON envelope used by Fish Speech servers."""
    content_type = response.headers.get("content-type", "").lower()
    if "json" not in content_type:
        return response.content
    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("Fish Speech returned invalid JSON instead of audio") from exc
    candidates: list[Any] = []
    if isinstance(body, dict):
        candidates.extend([body.get("audio"), body.get("audio_base64")])
        data = body.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("audio"), data.get("audio_base64")])
    for value in candidates:
        if isinstance(value, str):
            try:
                return base64.b64decode(value, validate=True)
            except ValueError:
                continue
    raise RuntimeError("Fish Speech JSON response did not contain base64 audio")


def _voice_description_from_request(request: TTSRequest) -> str | None:
    parts = [
        str(value.value if hasattr(value, "value") else value)
        for value in (request.gender, request.age, request.emotion, request.speaking_style)
        if value
    ]
    return ", ".join(parts) if parts else None


def _extract_audio_result(result: Any) -> Any:
    """Normalise provider return values to a usable form."""
    if isinstance(result, dict):
        for key in (
            "audio",
            "audio_array",
            "waveform",
            "wav",
            "speech",
            "tts_speech",
            "samples",
        ):
            if key in result:
                audio = result[key]
                sample_rate = result.get("sample_rate") or result.get("sampling_rate")
                return (audio, sample_rate) if sample_rate else audio
    return result


def _unpack_qwen_result(result: Any, default_sr: int = 24000) -> tuple:
    """Unpack qwen_tts returns of ``(List[np.ndarray], sample_rate)``."""
    if isinstance(result, tuple) and len(result) == 2:
        arrays, sr = result
        if isinstance(arrays, list):
            return arrays, int(sr)
        # Single array + sr
        return [arrays], int(sr)
    # Fallback: treat as raw audio
    if isinstance(result, list):
        return result, default_sr
    return [result], default_sr


def _accepted_kwargs(method: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    signature = inspect.signature(method)
    parameters = signature.parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {
        key: value
        for key, value in kwargs.items()
        if value is not None and key in parameters
    }
