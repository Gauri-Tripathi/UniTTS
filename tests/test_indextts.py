"""Offline tests of the real adapter boundaries; no weights or GPU downloads."""

import base64
import hashlib
import io
import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf
from click.testing import CliRunner

from unitts import TTSRequest, UniTTS
from unitts.cli import main
from unitts.managed.indextts import SOURCE_REVISION, IndexTTSEnvironment, child_environment
from unitts.managed.process import WorkerProcess
from unitts.providers.indextts import IndexTTSProvider
from unitts.workers.indextts import generate


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.wav"
    sf.write(path, np.sin(np.arange(2400) * 0.1) * 0.1, 24000)
    return path


def test_provider_does_not_import_host_torch(monkeypatch, tmp_path):
    monkeypatch.setattr("unitts.core.engine.resolve_device", Mock(side_effect=AssertionError("host CUDA probe")))
    with UniTTS(provider="indextts", cache_dir=tmp_path) as tts:
        assert tts.provider.device == "auto"
        assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_config_strings_are_parsed_and_shared_cache_is_honored(tmp_path):
    provider = IndexTTSProvider(model_cache=tmp_path, auto_install="false", timeout="30")
    assert provider.options["auto_install"] is False
    assert provider.options["timeout"] == 30
    assert provider.environment.root == tmp_path.resolve()


@pytest.mark.parametrize("changes,match", [
    ({"text": "   "}, "nonempty"),
    ({"reference_audio": []}, "exactly one"),
    ({"language": "fr"}, "Unsupported.*language"),
    ({"speed": 3}, "speed"),
    ({"output_format": "mp3"}, "WAV"),
    ({"stream": True}, "streaming"),
    ({"voice_description": "deep voice"}, "voice_description"),
    ({"controls": {"typo": 1}}, "Extra inputs"),
    ({"controls": {"emo_vector": [0.5]}}, "at least 8"),
    ({"controls": {"emo_vector": [-1] * 8}}, "greater than or equal"),
    ({"controls": {"emo_text": "happy"}, "emotion": "sad"}, "one emotion"),
    ({"max_length_seconds": 2}, "exact output duration"),
])
def test_invalid_request_does_not_install_or_load(reference, monkeypatch, changes, match):
    provider = IndexTTSProvider()
    start = Mock(side_effect=AssertionError("must validate first"))
    monkeypatch.setattr(provider, "_start", start)
    request = TTSRequest(**({"text": "hello", "reference_audio": reference} | changes))
    with pytest.raises(ValueError, match=match):
        provider.synthesize(request)
    start.assert_not_called()


def test_control_mapping_and_sampling_defaults(reference):
    provider = IndexTTSProvider()
    options, warnings = provider._prepare(TTSRequest(
        text="hello", reference_audio=reference, language="en-US", speed=2,
        emotion="happy", emotion_intensity=0.7,
    ))
    assert options["lang"] == "EN"
    assert options["duration_factor"] == 0.5
    assert options["emo_vector"] == [0.7, 0, 0, 0, 0, 0, 0, 0]
    assert options["spk_audio_prompt"] == str(reference.resolve())
    assert "temperature" not in options
    assert "repetition_penalty" not in options
    assert not warnings
    options, _ = provider._prepare(TTSRequest(text="hello", reference_audio=reference, temperature=0.4, top_p=0.8))
    assert options["temperature"] == 0.4
    assert options["top_p"] == 0.8


def test_text_emotion_is_enabled_explicitly(reference):
    options, _ = IndexTTSProvider()._prepare(TTSRequest(text="hello", reference_audio=reference, controls={"emo_text": "delighted"}))
    assert options["use_emo_text"] is True


def test_provider_reuses_process_and_preserves_real_metadata(reference, monkeypatch):
    result = {"audio_base64": base64.b64encode(b"wav").decode(), "sample_rate": 22050, "duration_seconds": 0.5}
    worker = Mock()
    worker.call.side_effect = [{"device": "cuda:0", "dtype": "float32"}, result, result]
    factory = Mock(return_value=worker)
    monkeypatch.setattr("unitts.providers.indextts.WorkerProcess", factory)
    monkeypatch.setattr(IndexTTSEnvironment, "ensure", Mock(return_value={}))
    provider = IndexTTSProvider()
    request = TTSRequest(text="hello", reference_audio=reference)
    one = provider.synthesize(request)
    two = provider.synthesize(request)
    assert one.sample_rate == two.sample_rate == 22050
    assert one.duration_seconds == 0.5
    assert one.metadata["device"] == "cuda:0"
    assert factory.call_count == 1
    provider.cleanup()
    assert provider._worker is None
    worker.close.assert_called_once()


def test_worker_failure_clears_provider_for_retry(reference):
    provider = IndexTTSProvider()
    worker = Mock()
    worker.call.side_effect = RuntimeError("device-side assert")
    provider._worker = worker
    with pytest.raises(RuntimeError, match="device-side"):
        provider.synthesize(TTSRequest(text="hello", reference_audio=reference))
    assert provider._worker is None
    worker.close.assert_called_once()


def test_output_path_rejected_before_synthesis(tmp_path, monkeypatch):
    with UniTTS(provider="indextts") as tts:
        synthesize = Mock(side_effect=AssertionError("expensive generation"))
        monkeypatch.setattr(tts, "synthesize", synthesize)
        with pytest.raises(ValueError, match="filename"):
            tts.synthesize_to_file("hello", tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            tts.synthesize_to_file("hello", tmp_path / "missing" / "out.wav")
        synthesize.assert_not_called()


def test_setup_pins_source_and_uses_upstream_lock(tmp_path, monkeypatch):
    environment = IndexTTSEnvironment(tmp_path)
    commands = []
    monkeypatch.setattr("unitts.managed.indextts.shutil.which", lambda name: "/tools/" + name)
    monkeypatch.setattr(environment, "_run", lambda command, **kwargs: commands.append(command))
    environment.install_dir.mkdir(parents=True)
    environment._install(30)
    assert any(SOURCE_REVISION in command and "fetch" in command for command in commands)
    sync = next(command for command in commands if "sync" in command)
    assert "--frozen" in sync and "3.11" in sync
    assert "--all-extras" not in sync
    assert json.loads((environment.install_dir / "installed.json").read_text())["source_revision"] == SOURCE_REVISION


def test_failed_setup_does_not_mark_complete(tmp_path, monkeypatch):
    environment = IndexTTSEnvironment(tmp_path)
    monkeypatch.setattr("unitts.managed.indextts.shutil.which", lambda name: name)
    monkeypatch.setattr(environment, "_run", Mock(side_effect=RuntimeError("network unavailable")))
    with pytest.raises(RuntimeError):
        environment._install(30)
    assert not (environment.install_dir / "installed.json").exists()


def test_external_checkpoint_is_not_silently_downloaded(tmp_path):
    environment = IndexTTSEnvironment(tmp_path / "cache", tmp_path / "missing")
    with pytest.raises(ValueError, match="config.yaml"):
        environment.ensure()


def test_auto_install_false_is_actionable(tmp_path):
    with pytest.raises(RuntimeError, match="unitts setup indextts"):
        IndexTTSEnvironment(tmp_path).ensure(install=False)


def test_completed_setup_reuses_environment_and_weights(tmp_path, monkeypatch):
    environment = IndexTTSEnvironment(tmp_path)
    environment.python.parent.mkdir(parents=True)
    environment.python.touch()
    (environment.install_dir / "installed.json").write_text(json.dumps({"source_revision": SOURCE_REVISION}))
    environment.model_dir.mkdir(parents=True)
    (environment.model_dir / "config.yaml").touch()
    (environment.model_dir / ".unitts-complete").touch()
    run = Mock()
    monkeypatch.setattr(environment, "_run", run)
    result = environment.ensure()
    assert result["installed"] is True
    assert run.call_count == 1  # Preflight only: no Git, uv sync, or model downloads.
    assert "--check" in run.call_args.args[0]


def test_subprocess_environment_ignores_host_virtualenv(monkeypatch):
    for key in ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME"):
        monkeypatch.setenv(key, "wrong-environment")
    env = child_environment()
    assert all(key not in env for key in ("VIRTUAL_ENV", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME"))


def test_doctor_never_installs(tmp_path, monkeypatch):
    install = Mock(side_effect=AssertionError("no downloads"))
    monkeypatch.setattr(IndexTTSEnvironment, "ensure", install)
    result = CliRunner().invoke(main, ["doctor", "indextts", "--cache-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["installed"] is False
    install.assert_not_called()


def test_cli_reference_and_emotion_forwarded(reference, tmp_path, monkeypatch):
    from unitts.core.schemas import TTSResponse

    def synthesize(self, request):
        assert request.reference_audio == [reference]
        assert request.emotion == "happy"
        return TTSResponse(audio=b"wav")
    monkeypatch.setattr(IndexTTSProvider, "synthesize", synthesize)
    result = CliRunner().invoke(main, ["synthesize", "hello", "--provider", "indextts", "--reference-audio", str(reference), "--emotion", "happy", "-o", str(tmp_path / "out.wav")])
    assert result.exit_code == 0, result.output


def test_real_process_reuse_and_timeout_cleanup(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(
        "import json, sys, time, os\n"
        "for line in sys.stdin:\n"
        " p = json.loads(line)\n"
        " if p.get('wait'): time.sleep(10)\n"
        " print(json.dumps({'protocol_version': p['protocol_version'], 'message_id': p['message_id'], 'ok': True, 'result': {'pid': os.getpid()}}), flush=True)\n",
        encoding="utf-8",
    )
    worker = WorkerProcess([sys.executable, "-u", str(script)], cwd=tmp_path, log_path=tmp_path / "worker.log")
    try:
        assert worker.call({}, timeout=10) == worker.call({}, timeout=10)
        with pytest.raises(RuntimeError, match="timed out"):
            worker.call({"wait": True}, timeout=0.1)
        assert worker.process.poll() is not None
    finally:
        worker.close()


@pytest.mark.parametrize("body,match", [
    ("import sys; sys.exit(3)", "exited"),
    ("import sys; sys.stdin.readline(); print('not-json', flush=True)", "Engine"),
    ("import json, sys; p=json.loads(sys.stdin.readline()); print(json.dumps({'protocol_version': p['protocol_version'], 'message_id': p['message_id'], 'ok': False, 'error': {'message': 'CUDA failure'}}), flush=True)", "CUDA failure"),
])
def test_real_process_failure_is_bounded(tmp_path, body, match):
    worker = WorkerProcess([sys.executable, "-u", "-c", body], cwd=tmp_path, log_path=tmp_path / "worker.log")
    try:
        with pytest.raises(RuntimeError, match=match):
            worker.call({}, timeout=10)
        assert worker.process.poll() is not None
    finally:
        worker.close()


def test_actual_worker_protocol_survives_python_and_native_stdout(tmp_path, reference):
    # Replace only model loading; execute the real worker transport and audio path.
    script = tmp_path / "driver.py"
    source = Path(__file__).resolve().parents[1]
    script.write_text(
        "import sys, os\n"
        f"sys.path.insert(0, {str(source)!r})\n"
        "from types import SimpleNamespace\n"
        "from unitts.workers import indextts as worker\n"
        "def load(settings):\n"
        " import numpy as np, soundfile as sf\n"
        " print('upstream Python log', flush=True)\n"
        " os.write(1, b'upstream native log\\n')\n"
        " def infer(output_path, **options):\n"
        "  print('inference log', flush=True)\n"
        "  sf.write(output_path, np.ones(2205)*0.1, 22050)\n"
        " return SimpleNamespace(infer=infer), {'device': 'test'}\n"
        "worker.load_model = load\n"
        "sys.modules['torch'] = SimpleNamespace()\n"
        "worker.main()\n",
        encoding="utf-8",
    )
    worker = WorkerProcess([sys.executable, "-u", str(script), "--source", str(source)], cwd=tmp_path, log_path=tmp_path / "worker.log")
    try:
        assert worker.call({"operation": "load"}, timeout=20)["device"] == "test"
        result = worker.call({"operation": "generate", "options": {"text": "hello", "spk_audio_prompt": str(reference)}, "sample_rate": 22050}, timeout=20)
        assert result["duration_seconds"] == 0.1
        assert base64.b64decode(result["audio_base64"]).startswith(b"RIFF")
    finally:
        worker.close()
    log = (tmp_path / "worker.log").read_text()
    assert "upstream Python log" in log and "upstream native log" in log


def test_worker_audio_validation_and_wav_metadata(reference, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    def infer(output_path, **options):
        sf.write(output_path, np.ones(2205) * 0.1, 22050)
    result = generate(SimpleNamespace(infer=infer), {
        "options": {"spk_audio_prompt": str(reference), "text": "hello"}, "sample_rate": 22050,
    })
    with wave.open(io.BytesIO(base64.b64decode(result["audio_base64"]))) as wav:
        assert wav.getframerate() == result["sample_rate"] == 22050
        assert wav.getnframes() == 2205
    assert result["duration_seconds"] == 0.1
    diagnostics = result["diagnostics"]
    assert diagnostics["native_sample_rate"] == diagnostics["output_sample_rate"] == 22050
    assert diagnostics["resample_seconds"] == 0
    assert diagnostics["references"]["spk_audio_prompt"]["duration_seconds"] == 0.1
    assert diagnostics["audio_sha256"] == hashlib.sha256(base64.b64decode(result["audio_base64"])).hexdigest()
    assert diagnostics["input_characters"] == len("hello")


def test_worker_rejects_long_reference_before_inference(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    path = tmp_path / "long.wav"
    sf.write(path, np.ones(31 * 8000) * 0.1, 8000)
    infer = Mock()
    with pytest.raises(ValueError, match="maximum 30s"):
        generate(SimpleNamespace(infer=infer), {"options": {"spk_audio_prompt": str(path)}})
    infer.assert_not_called()


def test_manifest_entrypoint_and_packaged_catalogue_agree():
    import yaml

    from unitts.engines.indextts import IndexTTSEngine
    from unitts.services.api import build_registry

    disk = yaml.safe_load(Path("engines/indextts-2.5/manifest.yaml").read_text())
    manifest = IndexTTSEngine.manifest()
    for field in ("engine_id", "entrypoint", "status", "upstream_code", "checkpoints", "limitations"):
        assert disk[field] == manifest.model_dump(mode="json")[field]
    assert isinstance(build_registry().create("indextts-2.5"), IndexTTSEngine)


@pytest.mark.asyncio
async def test_service_adapter_maps_shared_request(reference, monkeypatch):
    from unitts.contracts import InferenceContext, LoadContext, SynthesisRequest, VoiceReference
    from unitts.core.schemas import TTSResponse
    from unitts.engines.indextts import IndexTTSEngine

    engine = IndexTTSEngine()
    await engine.load(LoadContext(replica_id="test", device="auto"))
    result = TTSResponse(audio=b"wav", sample_rate=22050, duration_seconds=1)
    synthesize = Mock(return_value=result)
    monkeypatch.setattr(IndexTTSProvider, "synthesize", synthesize)
    request = SynthesisRequest(
        text="hello", reference=VoiceReference(uri=str(reference)), seed=11,
        controls={"emo_vector": [0.5, 0, 0, 0, 0, 0, 0, 0]},
    )
    response = await engine.generate(request, InferenceContext(request_id=request.request_id))
    actual = synthesize.call_args.args[0]
    assert actual.reference_audio == [reference]
    assert actual.seed == 11 and actual.controls == request.controls
    assert response.engine_id == "indextts-2.5" and response.data == b"wav"
    await engine.unload()
