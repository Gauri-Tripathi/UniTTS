"""Explicit opt-in: downloads and real GPU inference are never part of offline CI."""

import io
import os
import wave

import pytest

from unitts import UniTTS


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("UNITTS_TEST_INDEXTTS_GPU") != "1", reason="Real IndexTTS GPU test is opt-in")
def test_real_indextts_synthesis():
    reference = os.environ.get("UNITTS_TEST_REFERENCE")
    assert reference, "Set UNITTS_TEST_REFERENCE to a short speech recording"
    with UniTTS(provider="indextts", device="cuda") as tts:
        result = tts.synthesize("Hello. This is an Index TTS test.", reference_audio=reference)
        assert result.metadata["device"] == "cuda:0"
        with wave.open(io.BytesIO(result.audio)) as wav:
            assert wav.getnframes() > 0
            assert wav.getframerate() == result.sample_rate
            assert wav.getnframes() / wav.getframerate() == pytest.approx(result.duration_seconds)
