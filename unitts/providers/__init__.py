"""Provider package.

Importing this module registers lightweight provider classes only. Heavy model
libraries are imported lazily inside provider methods.
"""

from unitts.providers.api import AzureProvider, ElevenLabsProvider, FishAudioProvider, OpenAIProvider, SmallestAIProvider
from unitts.providers.local import EspeakProvider, PiperProvider
from unitts.providers.indextts import IndexTTSProvider
from unitts.providers.open_weights import DiaProvider, F5TTSProvider, FishSpeechProvider, KokoroProvider, Qwen3TTSProvider, XTTSV2Provider

__all__ = [
    "AzureProvider",
    "DiaProvider",
    "ElevenLabsProvider",
    "EspeakProvider",
    "F5TTSProvider",
    "FishAudioProvider",
    "FishSpeechProvider",
    "KokoroProvider",
    "IndexTTSProvider",
    "OpenAIProvider",
    "PiperProvider",
    "Qwen3TTSProvider",
    "SmallestAIProvider",
    "XTTSV2Provider",
]
