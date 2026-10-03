# GPU Provider Installation and Usage

UniTTS loads GPU providers lazily. Installing `.[gpu]` supplies shared packages
only; it does not install F5-TTS, Dia, Coqui TTS, or Qwen3-TTS.

Before starting a provider that runs in the current Python environment, check
CUDA availability there. On a cluster, run inside your GPU allocation.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Keep Provider Environments Separate

Do not install every GPU stack into the Qwen environment. F5-TTS, Dia, and XTTS
have independent Torch, Transformers, Python, and system-library
requirements. Create one environment per provider, then install this checkout in
that environment:

```bash
conda create -n unitts-f5 python=3.11
conda activate unitts-f5
# Run the following command from your UniTTS checkout.
python -m pip install -e ".[gpu]"
```

Install a PyTorch build approved for the allocated GPU and driver before installing
the provider package. Do not use the commands below to upgrade PyTorch blindly.

## Qwen3-TTS

```bash
python -m pip install -U qwen-tts
```

The default UniTTS model is `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`. It accepts
named speakers such as `ryan`; its language argument is normalized from `en` to
`english` by UniTTS. This model variant does not support reference-audio cloning
or description-driven voice design.

```python
from unitts import TTSRequest, UniTTS

with UniTTS(provider="qwen3-tts", device="cuda") as tts:
    response = tts.synthesize(TTSRequest(
        text="Hello from Qwen3-TTS.",
        language="en",
        voice="ryan",
    ))

open("qwen.wav", "wb").write(response.audio)
```

## F5-TTS

In a separate environment, install FFmpeg using your operating system's package manager,
then install the official package:

```bash
python -m pip install f5-tts
```

F5-TTS performs best when `reference_text` is the accurate transcript of the
reference audio.

```python
from unitts import TTSRequest, UniTTS

with UniTTS(provider="f5-tts", device="cuda") as tts:
    response = tts.synthesize(TTSRequest(
        text="This is a cloned voice.",
        reference_audio="reference.wav",
        reference_text="Accurate transcript of reference.wav.",
    ))

open("f5.wav", "wb").write(response.audio)
```

## XTTS v2

Use the maintained Coqui package, not the unmaintained `TTS` PyPI package:

```bash
# Example for a separate environment using CUDA 12.4 wheels:
python -m pip install --upgrade --force-reinstall \
  torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install coqui-tts transformers==4.57.6
```

`coqui-tts` requires both `torch` and `torchaudio`; it does not install either
one. Their versions and CUDA build must match. The CUDA 12.4 pair above is an
example, not a hardware check. Choose a compatible pair for your GPU and driver.
The current UniTTS XTTS adapter rejects Transformers 5.1+; this example pins
Transformers to `4.57.6`.

```python
from unitts import TTSRequest, UniTTS

with UniTTS(provider="xtts-v2", device="cuda") as tts:
    response = tts.synthesize(TTSRequest(
        text="Hello from XTTS.",
        language="en",
        reference_audio="reference.wav",
    ))

open("xtts.wav", "wb").write(response.audio)
```

## Dia

Dia is installed from the official source repository. Keep it in its own
environment because it can require a different Transformers/Torch combination.

```bash
python -m pip install git+https://github.com/nari-labs/dia.git
```

Dia expects dialogue tags. Keep generations moderately sized and alternate
`[S1]` and `[S2]`.

```python
from unitts import TTSRequest, UniTTS

with UniTTS(provider="dia", device="cuda") as tts:
    response = tts.synthesize(TTSRequest(
        text="[S1] Hello there. [S2] Hello back. [S1] Nice to meet you.",
        language="en",
    ))

open("dia.wav", "wb").write(response.audio)
```

## Fish Speech

`fish-speech` is a client for the official self-hosted Fish Speech API, not the
Fish Audio cloud provider (`fish` / `fish-audio`). Keep the Fish Speech checkout
and its model environment separate from UniTTS. The smaller
`openaudio-s1-mini` checkpoint is intended for local deployment; the S2 model
requires at least 24 GB of VRAM.

```bash
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech
# Follow the upstream installation instructions for this environment.
hf download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
python -m tools.api_server \
  --listen 127.0.0.1:8080 \
  --llama-checkpoint-path checkpoints/openaudio-s1-mini \
  --decoder-checkpoint-path checkpoints/openaudio-s1-mini/codec.pth \
  --decoder-config-name modded_dac_vq
```

With the server running, use UniTTS from its own environment. The default
endpoint is `http://127.0.0.1:8080/v1/tts`.

```python
from unitts import TTSRequest, UniTTS

with UniTTS(provider="fish-speech") as tts:
    response = tts.synthesize(TTSRequest(
        text="(excited) Hello from Fish Speech.",
        reference_audio="reference.wav",
        reference_text="Accurate transcript of reference.wav.",
    ))

open("fish-speech.wav", "wb").write(response.audio)
```

Pass `endpoint="http://host:port/v1/tts"` to `UniTTS` when the Fish Speech
server is not local.

## IndexTTS-2.5

IndexTTS uses a managed environment rather than your current Python's GPU stack.
Install `.[indextts]`, then use `UniTTS(provider="indextts")` for setup, status,
and synthesis. See [the IndexTTS guide](indextts.md) for the full workflow and
the explicit real-GPU test. Ordinary unit tests do not download model weights.

## Upstream References

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [F5-TTS installation](https://github.com/SWivid/F5-TTS)
- [Dia installation](https://github.com/nari-labs/dia)
- [Coqui TTS installation](https://coqui-tts.readthedocs.io/en/latest/installation.html)
- [Fish Speech self-hosting](https://docs.fish.audio/developer-guide/self-hosting/running-inference)
