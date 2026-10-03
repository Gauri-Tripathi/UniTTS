"""API-backed TTS providers."""

from __future__ import annotations

import html
import time
from typing import Any

import requests

from unitts.core.capabilities import CapabilityMatrix, ProviderType
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import OutputFormat, TTSRequest, TTSResponse
from unitts.providers.base_http import HttpProvider


@ProviderRegistry.register("openai")
class OpenAIProvider(HttpProvider):
    """OpenAI speech provider using the official SDK when installed."""

    name = "openai"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.API,
        languages=["auto"],
        streaming=False,
        speed=True,
        output_formats=["mp3", "opus", "wav", "flac"],
        notes="Emotion and cloning are intentionally unsupported and produce warnings.",
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        api_key = self.options.get("api_key")
        if not api_key:
            raise ValueError("OpenAI provider requires api_key or UNITTS_OPENAI_API_KEY")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise ImportError("Install unitts[api] to use the OpenAI provider") from exc

        client = OpenAI(api_key=api_key)
        model = self.options.get("model", "tts-1")
        voice = request.voice or self.options.get("voice", "alloy")
        result = client.audio.speech.create(
            model=model,
            voice=voice,
            input=request.ssml or request.text or request.phonemes or "",
            response_format=request.output_format.value,
            speed=request.speed or 1.0,
        )
        audio = result.read() if hasattr(result, "read") else bytes(result.content)
        return TTSResponse(audio=audio, output_format=request.output_format, metadata={"provider": self.name, "model": model, "voice": voice}, warnings=warnings)

    def list_voices(self) -> list[dict[str, Any]]:
        return [{"id": voice} for voice in ["alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"]]


@ProviderRegistry.register("elevenlabs")
class ElevenLabsProvider(HttpProvider):
    """ElevenLabs text-to-speech provider."""

    name = "elevenlabs"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.API,
        languages=["auto"],
        voice_cloning=True,
        streaming=False,
        emotion=True,
        speed=True,
        output_formats=["mp3"],
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        api_key = self.options.get("api_key")
        if not api_key:
            raise ValueError("ElevenLabs provider requires api_key or UNITTS_ELEVENLABS_API_KEY")
        voice_id = request.voice or self.options.get("voice", "21m00Tcm4TlvDq8ikWAM")
        model = self.options.get("model", "eleven_multilingual_v2")
        stability = 0.5
        similarity_boost = 0.75
        if request.emotion_intensity is not None:
            stability = max(0.0, min(1.0, 1.0 - request.emotion_intensity * 0.6))
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        payload = {
            "text": request.ssml or request.text or request.phonemes or "",
            "model_id": model,
            "voice_settings": {"stability": stability, "similarity_boost": similarity_boost},
        }
        audio = self._post_audio(url, headers={"xi-api-key": api_key, "Accept": "audio/mpeg"}, json=payload)
        return TTSResponse(audio=audio, output_format=OutputFormat.MP3, metadata={"provider": self.name, "model": model, "voice": voice_id}, warnings=warnings)

    def list_voices(self) -> list[dict[str, Any]]:
        api_key = self.options.get("api_key")
        if not api_key:
            raise ValueError("ElevenLabs provider requires api_key or UNITTS_ELEVENLABS_API_KEY")
        data = self._get_json("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": api_key})
        return [{"id": item.get("voice_id"), "name": item.get("name")} for item in data.get("voices", [])]


@ProviderRegistry.register("fish")
@ProviderRegistry.register("fish-audio")
class FishAudioProvider(HttpProvider):
    """Fish Audio REST provider."""

    name = "fish"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.API,
        languages=["auto"],
        voice_cloning=True,
        streaming=False,
        output_formats=["mp3", "wav"],
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        api_key = self.options.get("api_key")
        if not api_key:
            raise ValueError("Fish Audio provider requires api_key or UNITTS_FISH_API_KEY")
        model = self.options.get("model", "speech-1.6")
        payload: dict[str, Any] = {
            "text": request.text or request.ssml or request.phonemes or "",
            "format": request.output_format.value,
            "mp3_bitrate": 128,
            "normalize": request.normalize,
        }
        if request.voice:
            payload["reference_id"] = request.voice
        audio = self._post_audio(
            "https://api.fish.audio/v1/tts",
            headers={"Authorization": f"Bearer {api_key}", "model": model},
            json=payload,
        )
        return TTSResponse(audio=audio, output_format=request.output_format, metadata={"provider": self.name, "model": model}, warnings=warnings)


@ProviderRegistry.register("smallest")
@ProviderRegistry.register("smallest-ai")
class SmallestAIProvider(HttpProvider):
    """Smallest AI low-latency TTS provider."""

    name = "smallest"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.API,
        languages=["auto"],
        streaming=False,
        speed=True,
        output_formats=["wav", "mp3"],
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        api_key = self.options.get("api_key")
        if not api_key:
            raise ValueError("Smallest AI provider requires api_key or UNITTS_SMALLEST_API_KEY")
        endpoint = self.options.get("endpoint", "https://waves-api.smallest.ai/api/v1/lightning/get_speech")
        payload = {
            "text": request.text or request.ssml or request.phonemes or "",
            "voice_id": request.voice or self.options.get("voice", "emily"),
            "sample_rate": request.sample_rate or 24000,
            "speed": request.speed or 1.0,
            "output_format": request.output_format.value,
        }
        audio = self._post_audio(endpoint, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
        return TTSResponse(audio=audio, sample_rate=payload["sample_rate"], output_format=request.output_format, metadata={"provider": self.name}, warnings=warnings)


@ProviderRegistry.register("azure")
class AzureProvider(HttpProvider):
    """Azure Cognitive Services TTS provider."""

    name = "azure"
    capabilities = CapabilityMatrix(
        provider_type=ProviderType.API,
        languages=["auto"],
        ssml=True,
        emotion=True,
        speaking_style=True,
        speed=True,
        pitch=True,
        volume=True,
        output_formats=["wav", "mp3", "ogg"],
    )

    def synthesize(self, request: TTSRequest) -> TTSResponse:
        warnings = self._warnings_for(request)
        api_key = self.options.get("api_key")
        region = self.options.get("region")
        if not api_key or not region:
            raise ValueError("Azure provider requires api_key and region")
        token = self._azure_token(api_key, region)
        ssml = request.ssml or self._build_ssml(request)
        fmt = _azure_format(request.output_format)
        url = f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        audio = self._post_audio(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": fmt,
                "User-Agent": "unitts",
            },
            data=ssml,
        )
        return TTSResponse(audio=audio, output_format=request.output_format, metadata={"provider": self.name, "azure_format": fmt}, warnings=warnings)

    def _azure_token(self, api_key: str, region: str) -> str:
        response = requests.post(
            f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken",
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.text

    def _build_ssml(self, request: TTSRequest) -> str:
        voice = request.voice or self.options.get("voice", "en-US-JennyNeural")
        lang = request.language or self.options.get("language", "en-US")
        text = html.escape(request.text or request.phonemes or "")
        prosody: list[str] = []
        if request.speed:
            prosody.append(f'rate="{request.speed:.2f}"')
        if request.pitch:
            prosody.append(f'pitch="{request.pitch:+.1f}st"')
        if request.volume:
            prosody.append(f'volume="{request.volume * 100:.0f}%"')
        inner = text
        if prosody:
            inner = f"<prosody {' '.join(prosody)}>{inner}</prosody>"
        if request.speaking_style or request.emotion:
            style = request.speaking_style.value if request.speaking_style else request.emotion.value
            degree = request.emotion_intensity or request.expressiveness or 1.0
            inner = f'<mstts:express-as style="{style}" styledegree="{degree:.2f}">{inner}</mstts:express-as>'
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{lang}">'
            f'<voice name="{voice}">{inner}</voice></speak>'
        )


def _azure_format(output_format: OutputFormat) -> str:
    mapping = {
        OutputFormat.WAV: "riff-24khz-16bit-mono-pcm",
        OutputFormat.MP3: "audio-24khz-160kbitrate-mono-mp3",
        OutputFormat.OGG: "ogg-24khz-16bit-mono-opus",
        OutputFormat.OPUS: "ogg-24khz-16bit-mono-opus",
        OutputFormat.FLAC: "riff-24khz-16bit-mono-pcm",
    }
    return mapping[output_format]


def rate_limit_sleep(retry_after: str | None) -> None:
    """Sleep for a provider rate-limit response header."""

    if retry_after and retry_after.isdigit():
        time.sleep(min(int(retry_after), 30))
