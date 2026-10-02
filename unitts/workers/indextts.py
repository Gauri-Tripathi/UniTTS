"""Standalone IndexTTS worker. Does not import UniTTS or its host dependencies."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import tempfile
import traceback
import warnings
from pathlib import Path


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
    if options.get("use_emo_text") and model.qwen_emo is None:
        # Optional emotion model is only loaded when requested, in this same worker.
        from indextts.infer_v2_5 import QwenEmotion
        model.qwen_emo = QwenEmotion(str(Path(model.model_dir) / model.cfg.qwen_emo_path))
    with tempfile.TemporaryDirectory(prefix="unitts-index-") as temp:
        output = Path(temp) / "speech.wav"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.infer(output_path=str(output), **options)
        audio, rate = sf.read(output)
        if not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
            raise RuntimeError("IndexTTS returned empty, silent, or non-finite audio")
        # Respect the common request's sample rate without falsifying the WAV header.
        target_rate = payload.get("sample_rate") or rate
        if target_rate != rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=target_rate, axis=0)
            rate = target_rate
            sf.write(output, audio, rate, subtype="PCM_16")
        return {
            "audio_base64": base64.b64encode(output.read_bytes()).decode("ascii"),
            "sample_rate": rate, "duration_seconds": len(audio) / rate,
            "warnings": [str(item.message) for item in caught],
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
    # Keep protocol output separate even from native libraries writing to fd 1.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    model = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if payload["operation"] == "load":
                model, result = load_model(payload)
            elif payload["operation"] == "generate" and model is not None:
                result = generate(model, payload)
            else:
                raise ValueError("Worker needs a load operation before generation")
            protocol.write(json.dumps({"ok": True, "result": result}) + "\n")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            protocol.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
            # Exit after failures so a damaged CUDA context is never reused.
            break


if __name__ == "__main__":
    main()
