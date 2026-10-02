"""Standalone IndexTTS worker. Does not import UniTTS or its host dependencies."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import random
import sys
import tempfile
import warnings
from pathlib import Path
from time import perf_counter


def preflight(device: str) -> dict:
    import torch
    import transformers
    from indextts.infer_v2_5 import IndexTTS2  # noqa: F401

    if sys.version_info[:2] != (3, 11) or transformers.__version__ != "4.52.1" or not torch.__version__.startswith("2.8."):
        raise RuntimeError("Incompatible managed environment; rerun unitts setup indextts after removing its installed.json marker")
    if device not in {"auto", "cuda", "cuda:0", "cpu", "imports"}:
        raise ValueError("IndexTTS device must be auto, cuda, cuda:0, or cpu")
    if device == "imports":
        return {"imports": "ok"}
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("IndexTTS needs a CUDA-enabled runtime. Enable a GPU in Colab or install a compatible NVIDIA driver. CPU requires explicit device='cpu'.")
    resolved = "cpu" if device == "cpu" else "cuda:0"
    return {"device": resolved, "gpu": torch.cuda.get_device_name(0) if resolved != "cpu" else None}


def load_model(settings: dict):
    import torch
    from indextts.infer_v2_5 import IndexTTS2

    info = preflight(settings["device"])
    dtype = settings.get("dtype", "auto")
    bf16 = info["device"] != "cpu" and torch.cuda.is_bf16_supported()
    if dtype == "bfloat16" and not bf16:
        raise ValueError("This GPU does not support BF16. Use dtype='auto' or 'float32'.")
    if dtype not in {"auto", "bfloat16", "float32"}:
        raise ValueError("IndexTTS-2.5 dtype must be auto, bfloat16, or float32")
    model_dir = Path(settings["model_dir"])
    if not (model_dir / "config.yaml").is_file():
        raise ValueError(f"Missing config.yaml in {model_dir}")
    # Download auxiliary resources during load rather than partway through inference.
    from indextts.utils.model_download import ensure_models_available
    ensure_models_available(str(model_dir))
    model = IndexTTS2(
        cfg_path=str(model_dir / "config.yaml"), model_dir=str(model_dir), device=info["device"],
        use_bf16=bf16 and dtype != "float32", use_cuda_kernel=False, use_deepspeed=False,
        use_qwen_emo=False,
    )
    info["dtype"] = "bfloat16" if bf16 and dtype != "float32" else "float32"
    return model, info


def generate(model, payload: dict) -> dict:
    import numpy as np
    import soundfile as sf
    import torch

    options = dict(payload["options"])
    references = {}
    seed = payload.get("seed")
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
    for field in ("spk_audio_prompt", "emo_audio_prompt"):
        if not options.get(field):
            continue
        samples, rate = sf.read(options[field])
        duration = len(samples) / rate
        if samples.size == 0 or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-5:
            raise ValueError(f"{field} must contain finite, audible speech")
        if duration > 30:
            raise ValueError(f"{field} is {duration:.1f}s long. Use a clean 3–15s excerpt (maximum 30s).")
        references[field] = {
            "duration_seconds": duration, "sample_rate": rate,
            "channels": 1 if samples.ndim == 1 else samples.shape[1],
            "peak": float(np.max(np.abs(samples))),
            "near_full_scale_fraction": float(np.mean(np.abs(samples) >= 0.999)),
        }
    if options.get("use_emo_text") and model.qwen_emo is None:
        # Optional emotion model is only loaded when requested, in this same worker.
        from indextts.infer_v2_5 import QwenEmotion
        model.qwen_emo = QwenEmotion(str(Path(model.model_dir) / model.cfg.qwen_emo_path))
    with tempfile.TemporaryDirectory(prefix="unitts-index-") as temp:
        output = Path(temp) / "speech.wav"
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.infer(output_path=str(output), **options)
        inference_seconds = perf_counter() - start
        audio, rate = sf.read(output)
        if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
            raise RuntimeError("IndexTTS returned empty, silent, or non-finite audio")
        # Respect the common request's sample rate without falsifying the WAV header.
        native_rate = rate
        native_duration = len(audio) / rate
        target_rate = payload.get("sample_rate") or rate
        start = perf_counter()
        if target_rate != rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=target_rate, axis=0)
            rate = target_rate
            sf.write(output, audio, rate, subtype="PCM_16")
        resample_seconds = perf_counter() - start if target_rate != native_rate else 0.0
        data = output.read_bytes()
        return {
            "audio_base64": base64.b64encode(data).decode("ascii"),
            "sample_rate": rate, "duration_seconds": len(audio) / rate,
            "warnings": [str(item.message) for item in caught],
            "diagnostics": {
                "inference_seconds": inference_seconds,
                "resample_seconds": resample_seconds,
                "native_sample_rate": native_rate, "output_sample_rate": rate,
                "native_duration_seconds": native_duration,
                "audio_sha256": hashlib.sha256(data).hexdigest(),
                "text_sha256": hashlib.sha256(options["text"].encode("utf-8")).hexdigest(),
                "input_characters": len(options["text"]),
                "language": options.get("lang"), "duration_factor": options.get("duration_factor", 1.0),
                "references": references,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--check")
    parser.add_argument("--download")
    parser.add_argument("--revision")
    args = parser.parse_args()
    sys.path.insert(0, args.source)
    if args.check:
        print(json.dumps(preflight(args.check)))
        return
    if args.download:
        from huggingface_hub import snapshot_download
        from indextts.utils.model_download import ensure_models_available
        snapshot_download("IndexTeam/IndexTTS-2.5", revision=args.revision, local_dir=args.download)
        ensure_models_available(args.download)
        return
    # -I intentionally excludes the host package: load the stdlib-only sibling.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from protocol import serve

    model = None

    def dispatch(payload):
        nonlocal model
        if payload["operation"] == "load":
            model, result = load_model(payload)
            return result
        if payload["operation"] == "generate" and model is not None:
            return generate(model, payload)
        raise ValueError("Worker needs a load operation before generation")

    # Exit after failures so a damaged CUDA context is never reused.
    asyncio.run(serve(dispatch, stop_on_error=True))


if __name__ == "__main__":
    main()
