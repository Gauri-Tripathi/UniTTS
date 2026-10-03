# IndexTTS-2.5 through UniTTS

The SDK manages a separate Python 3.11 process for IndexTTS. It does not depend on
your notebook's Torch, Transformers, or CUDA Python packages. The implementation
is experimental: offline tests cover the process and adapter, while actual audio,
driver compatibility, VRAM requirements and the Docker image need GPU validation.

## Install and generate

From the UniTTS checkout:

```bash
python -m pip install -e ".[indextts]"
```

In Colab, with a GPU runtime and the repository at `/content/UniTTS`:

```python
%pip install -e "/content/UniTTS[indextts]"
```

Use normal notebook cells. No environment activation, upstream imports, custom
kernel, or separate user-written script is necessary:

```python
from unitts import UniTTS
from IPython.display import Audio, display

tts = UniTTS(provider="indextts")
result = tts.synthesize(
    "Hello! This is my first IndexTTS sentence.",
    reference_audio="/content/reference.wav",
    language="en",
)
print(result.metadata)  # actual worker device and precision
display(Audio(result.audio))
```

Use a clean, audible recording of one speaker, preferably 3–15 seconds. References
over 30 seconds are rejected. A reference transcript is not required. To save:

```python
result = tts.synthesize_to_file(
    "This sentence is saved to a WAV file.",
    "/content/output.wav",
    reference_audio="/content/reference.wav",
)
```

Keep `tts` for subsequent generations. When finished, call `tts.cleanup()` to
terminate its process and reclaim the GPU allocation. A context manager provides
the same cleanup automatically. Each `UniTTS` instance owns a separate worker;
avoid constructing several instances on a small GPU.

## What setup does

First synthesis logs its stages and automatically:

1. Fetches upstream source at `d9e41aac89fd00b3d71497fddb287b7f24613712`.
2. Uses `uv` to provision Python 3.11 and `uv sync --frozen --no-dev` against that
   revision's dependency lock, including Torch 2.8 and Transformers 4.52.1.
3. Checks imports and CUDA in that interpreter. `auto` requires CUDA; there is no
   silent fallback to CPU. Explicit `device="cpu"` is available but slow.
4. Downloads `IndexTeam/IndexTTS-2.5` at
   `c39ce5ba981572cb187443877ff559dfb246ce63` plus upstream auxiliary resources.
5. Starts one persistent worker. It chooses BF16 when supported, otherwise FP32.
   Custom CUDA kernels, DeepSpeed and compilation are disabled by default.

Auxiliary resources follow the pinned upstream downloader; their own remote
revisions are not fully pinned. The source and primary checkpoint are pinned.
This setup requires Git and network access. On Linux, install FFmpeg, libsndfile
and libgomp through your OS if missing. The managed environment uses upstream's
CUDA 12.8 Torch wheels, so the NVIDIA driver must support that runtime.

`unitts setup indextts` performs preparation ahead of time. In a notebook:

```python
from unitts import UniTTS

with UniTTS(provider="indextts") as tts:
    print(tts.setup())   # Explicit preparation; synthesis also prepares on first use.
    print(tts.status())  # Paths and worker state; no downloads or model loading.
```

To prevent automatic installation, use `UniTTS(provider="indextts",
auto_install=False)`. It reports the setup command if the environment is absent.

## Emotion, speed and generation controls

Common controls:

```python
result = tts.synthesize(
    "The package has arrived.",
    reference_audio="reference.wav",
    emotion="happy",
    emotion_intensity=0.6,
    speed=1.2,
    seed=11,
)
```

Supported common emotion labels: happy, angry, sad, fearful, disgusted, surprised,
calm and neutral. Speed is 0.5–2.0; larger values mean faster speech. Unsupported
emotions, language codes, streaming and non-WAV output fail before setup.

Pass advanced controls through `controls`, with exactly one emotion source:

```python
# Separate emotional performance while retaining the reference speaker.
result = tts.synthesize(
    "I thought you would be here today.", reference_audio="reference.wav",
    controls={"emo_audio_prompt": "sad.wav", "emo_alpha": 0.6},
)

# Text emotion guidance. Loads QwenEmotion lazily in the same worker.
result = tts.synthesize(
    "The meeting is tomorrow.", reference_audio="reference.wav",
    controls={"emo_text": "I am excited and delighted!", "emo_alpha": 0.6},
)

# Emotion vector order: happy, angry, sad, afraid, disgusted,
# melancholic, surprised, calm.
result = tts.synthesize(
    "The meeting is tomorrow.", reference_audio="reference.wav",
    controls={"emo_vector": [0, 0, 0.6, 0, 0, 0, 0, 0]},
)
```

Additional validated controls: `use_emo_text`, `use_random`, `max_mel_tokens`
(50–3000, default 1500), `max_text_tokens_per_segment` (20–200, default 120),
and `interval_silence` (milliseconds, default 200). Explicit common sampling
fields `temperature`, `top_k`, `top_p` and `repetition_penalty` are forwarded;
otherwise upstream defaults are preserved. Seeds do not guarantee identical
results on different hardware. A token-limit warning means output may be cut off.

Supported target languages: en, zh, ja, es, ar. Upstream pronunciation annotations
can be included in the plain text. This provider clones a reference voice; it has
no preset voice catalogue or arbitrary voice-design mode. `voice_description` is
rejected; separate emotion descriptions use `controls.emo_text`.

## Paths, diagnostics and failures

The default root is `~/.cache/unitts`. Override with `cache_dir=...` or
`UNITTS_CACHE_DIR`. Paths are resolved once and remain independent of notebook
directory changes. Downloads use a process lock and are reused on later runs.

Existing complete checkpoints can be reused without downloading the primary model:

```python
tts = UniTTS(provider="indextts", model_dir="/content/checkpoints")
```

This uses the provided folder as-is; auxiliary downloads may still occur. It must
contain the correct IndexTTS-2.5 checkpoint files. An arbitrary model ID or older
IndexTTS checkpoint is not accepted by this adapter.

`unitts doctor indextts` reports interpreter, checkpoint, revision and log paths
without downloads. `setup.log` contains install/preflight output. `worker.log`
contains model loading and inference diagnostics; it may contain submitted text.

The SDK exposes the same diagnostics as `tts.status()` and capabilities as
`tts.capabilities`. Setup, status, synthesis, and cleanup all go through `UniTTS`;
applications do not need to instantiate IndexTTS providers or environment managers.

Default limits: setup commands/downloads 1800s each, model loading 600s, generation
300s. Override `setup_timeout`, `load_timeout`, and `timeout` in the constructor.
A timeout or worker error terminates the process; the next call starts a fresh
one. A long generation can therefore be interrupted without poisoning the
notebook's CUDA context. First load can fetch auxiliary files and take longer.

The parent kernel does not gain an `IndexTTS2` object: only the common `tts` object
and response live there. Do not import `indextts` in the notebook or install its
dependencies into your Qwen environment.

## Real GPU smoke test

Offline tests cover installation commands, failure recovery, persistent IPC,
request validation, control routing and WAV handling without downloading models.
For a real synthesis test, explicitly opt in and supply a reference:

```bash
UNITTS_TEST_INDEXTTS_GPU=1 UNITTS_TEST_REFERENCE=/absolute/reference.wav \
  python -m pytest tests/integration/test_indextts_gpu.py -q
```

This test downloads dependencies and model files if absent. A passing mock test
does not establish model quality or a guaranteed minimum GPU-memory requirement.

Upstream code and weights use the bilibili Model Use License Agreement; see
[the pinned upstream license](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/LICENSE).
UniTTS does not relicense or redistribute those weights.
