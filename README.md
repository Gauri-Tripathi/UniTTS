# UniTTS

UniTTS provides one Python interface for local and hosted text-to-speech systems.
It includes a Python SDK, command-line interface, local web UI, and an experimental
service runtime for isolated speech engines.

> **Project status:** pre-release. Piper is the recommended local CPU path. Other
> providers require their own credentials, packages, model weights, and hardware.

## Highlights

- Common request and response models across providers
- Lazy loading for optional CPU, GPU, and API dependencies
- Provider capability discovery from the CLI and web UI
- Reusable model sessions for repeated local inference
- Isolated engine manifests and lifecycle contracts for service deployments

## Quick start

UniTTS requires Python 3.11 or newer. From the repository root in WSL or Linux:

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install -e ".[cpu]"

export UNITTS_PIPER_MODEL_CACHE="$PWD/.unitts-cache/piper"
unitts synthesize \
  "Hello from UniTTS" \
  --provider piper \
  --voice en_US-lessac-medium \
  --output hello.wav
```

The first Piper request downloads the selected voice. Later requests reuse the
cached model and run locally on CPU.

## Python API

```python
from unitts import UniTTS

with UniTTS(
    provider="piper",
    device="cpu",
    model_cache=".unitts-cache/piper",
) as tts:
    result = tts.synthesize_to_file(
        "Hello from UniTTS",
        "hello.wav",
        voice="en_US-lessac-medium",
    )

print(result.metadata)
```

## IndexTTS-2.5

Install the lightweight managed-engine tools into your current environment:

```bash
python -m pip install -e ".[indextts]"
```

Then use the same Python API, including in a Colab Python 3.13 notebook:

```python
from unitts import UniTTS

with UniTTS(provider="indextts") as tts:
    result = tts.synthesize_to_file(
        "Hello from IndexTTS.",
        "hello.wav",
        reference_audio="reference.wav",
        language="en",
    )
```

First use prepares a separate Python 3.11 environment, installs the pinned upstream
dependencies, downloads model files, and starts an isolated worker. Your notebook's
Torch and Transformers installations are untouched. Later calls on the same `tts`
object reuse its loaded model. `cleanup()` or the context manager stops the worker.
Git and a working NVIDIA GPU/driver are required for the default path. Downloads
require network access and several GB of free disk space.

Optionally prepare the environment ahead of first synthesis:

```bash
unitts setup indextts
unitts doctor indextts
unitts synthesize "Hello" --provider indextts --reference-audio reference.wav --output hello.wav
```

The backend is experimental; real-GPU compatibility is not certified. Model code
and weights retain their upstream license. See [the IndexTTS guide](docs/indextts.md)
for Colab setup, emotion controls, cached checkpoints, logs, and limitations.

## Commands

```bash
# Inspect registered providers and capabilities.
unitts providers

# List voices exposed by a provider.
unitts voices --provider piper

# Start the local UI at http://localhost:8765.
unitts-web
```

The web UI retains the selected model between requests, reducing repeated inference
startup cost. The `unitts-api` command starts the experimental engine runtime; its
`dummy` engine generates test tones rather than speech.

## Development

```bash
python -m pip install -e ".[dev,service]"
python -m pytest -q
```

GPU engines should use separate environments because their PyTorch, CUDA, and model
package requirements can conflict. Normal tests do not download model weights.

## Project layout

- `unitts/core` and `unitts/providers`: synchronous SDK, CLI, and web UI
- `unitts/contracts` and `unitts/engine_sdk`: engine interfaces and wire models
- `unitts/runtime` and `unitts/services`: scheduling, workers, streaming, and API
- `engines`: isolated engine manifests and packaging metadata

## License

MIT
