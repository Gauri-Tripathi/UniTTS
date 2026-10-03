"""Regression tests for the common SDK/service execution and worker contracts."""

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from unitts import Capabilities, ProviderCapabilities, TTSRequest, TTSResponse, UniTTS
from unitts.contracts import EngineCapabilities, InferenceContext, LoadContext, SynthesisRequest, VoiceReference
from unitts.core.capabilities import CapabilityMatrix
from unitts.core.registry import ProviderRegistry
from unitts.engines.indextts import IndexTTSEngine
from unitts.engines.qwen3_tts import Qwen3TTSEngine
from unitts.managed.indextts import IndexTTSEnvironment
from unitts.managed.process import WorkerProcess
from unitts.providers.indextts import IndexTTSProvider
from unitts.providers.open_weights import Qwen3TTSProvider
from unitts.services.api import build_registry
from unitts.workers.protocol import PROTOCOL_VERSION, read_response, request_message


def test_public_index_management_uses_unitts_without_loading_models(tmp_path, monkeypatch):
    prepare = Mock(return_value={"installed": True})
    monkeypatch.setattr(IndexTTSEnvironment, "ensure", prepare)
    monkeypatch.setattr("unitts.core.engine.resolve_device", Mock(side_effect=AssertionError("host GPU probe")))
    with UniTTS(provider="indextts", cache_dir=tmp_path) as tts:
        state = tts.status()
        assert state["managed"] and not state["worker_running"]
        assert not state["installed"]
        assert tts.capabilities.voice_cloning
        snapshot = tts.capabilities
        snapshot.voice_cloning = False
        assert tts.capabilities.voice_cloning
        prepare.assert_not_called()
        assert tts.setup()["installed"]
        prepare.assert_called_once_with(device="auto", timeout=1800.0, install=True)


def test_cli_setup_and_doctor_delegate_to_public_sdk(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from unitts.cli import main

    setup = Mock(return_value={"installed": True})
    status = Mock(return_value={"installed": False})
    monkeypatch.setattr(UniTTS, "setup", setup)
    monkeypatch.setattr(UniTTS, "status", status)
    runner = CliRunner()
    assert runner.invoke(main, ["setup", "indextts", "--cache-dir", str(tmp_path)]).exit_code == 0
    assert runner.invoke(main, ["doctor", "indextts", "--cache-dir", str(tmp_path)]).exit_code == 0
    setup.assert_called_once()
    status.assert_called_once()


def test_failed_provider_switch_keeps_the_current_provider(tmp_path, monkeypatch):
    monkeypatch.setattr("unitts.core.engine.provider_options", lambda provider, options: options)
    with UniTTS(provider="indextts", cache_dir=tmp_path) as tts:
        original = tts.provider
        cleanup = Mock()
        monkeypatch.setattr(original, "cleanup", cleanup)
        with pytest.raises(ValueError, match="Unknown provider"):
            tts.set_provider("nonexistent-provider")
        with pytest.raises(ValueError, match="dtype"):
            tts.set_provider("indextts", dtype="invalid")
        assert tts.provider is original
        cleanup.assert_not_called()
        tts.set_provider("indextts", cache_dir=tmp_path)
        cleanup.assert_called_once()
        assert tts.provider is not original


@pytest.mark.parametrize("field", ["text", "ssml", "phonemes"])
def test_shared_request_rejects_whitespace_only_input(field):
    with pytest.raises(ValueError, match="required"):
        TTSRequest(**{field: " \t\n"})


def test_capability_names_are_the_same_schema():
    assert Capabilities is CapabilityMatrix is ProviderCapabilities is EngineCapabilities
    caps = ProviderCapabilities(
        provider_type="gpu", supports_streaming=True, supported_languages=["en"],
        supported_formats=["wav"], supports_reference_audio=True,
    )
    assert caps.streaming and caps.supports_streaming
    assert caps.voice_cloning and caps.supports_voice_cloning
    assert caps.languages == ["en"]
    assert "supports_streaming" not in caps.model_dump()
    caps.supports_streaming = False
    assert caps.streaming is False
    with pytest.raises(ValueError, match="Conflicting"):
        Capabilities(streaming=False, supports_streaming=True)


@pytest.mark.parametrize("engine,provider", [
    (IndexTTSEngine, IndexTTSProvider), (Qwen3TTSEngine, Qwen3TTSProvider),
])
def test_service_catalogue_uses_provider_capabilities(engine, provider):
    manifest = engine.manifest()
    assert manifest.capabilities.model_dump() == provider.capabilities.model_dump()
    registry = build_registry()
    assert registry.get(manifest.engine_id).capabilities == provider.capabilities
    assert registry.get(manifest.engine_id).capabilities is not provider.capabilities


def test_wire_request_conversion_preserves_transcript_and_sampling_controls():
    request = SynthesisRequest(
        text="Hello", language="en", voice="ryan", speed=1.2, seed=7,
        reference=VoiceReference(uri="voice.wav", transcript="Original words"),
        controls={"temperature": 0.4, "top_p": 0.8},
    ).to_tts_request()
    assert request.reference_audio == [Path("voice.wav")]
    assert request.reference_text == "Original words"
    assert request.voice == "ryan" and request.language == "en"
    assert request.speed == 1.2 and request.seed == 7
    assert request.controls == {"temperature": 0.4, "top_p": 0.8}
    assert "temperature" not in request.model_fields_set
    # Wire defaults must not accidentally override a provider's sampling defaults.
    assert "seed" not in SynthesisRequest(text="Hello").to_tts_request().model_fields_set


def test_service_envelope_inherits_sdk_validation_and_fields():
    assert issubclass(SynthesisRequest, TTSRequest)
    request = SynthesisRequest(text="hello", emotion="happy", temperature=0.4, sample_rate=22050)
    converted = request.to_tts_request()
    assert converted.emotion == "happy" and converted.temperature == 0.4
    assert converted.sample_rate == 22050
    for request_cls in (TTSRequest, SynthesisRequest):
        with pytest.raises(ValueError):
            request_cls(text="hello", speed=0.05)
    assert SynthesisRequest(text="hello", streaming=True).to_tts_request().stream
    assert SynthesisRequest(text="hello").model_copy(update={"streaming": True}).stream
    with pytest.raises(ValueError, match="not both"):
        SynthesisRequest(text="hello", reference=VoiceReference(uri="one.wav"), reference_audio="two.wav")


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_cls,provider_cls", [
    (IndexTTSEngine, IndexTTSProvider), (Qwen3TTSEngine, Qwen3TTSProvider),
])
async def test_sdk_and_service_execute_identical_provider_requests(monkeypatch, engine_cls, provider_cls):
    requests = []

    def synthesize(self, request):
        requests.append(request.model_dump(mode="json"))
        return TTSResponse(audio=b"wav", sample_rate=22050, duration_seconds=1, warnings=["test warning"])

    monkeypatch.setattr(provider_cls, "synthesize", synthesize)
    monkeypatch.setattr("unitts.core.engine.provider_options", lambda provider, options: options)
    request = SynthesisRequest(
        text="Hello", language="en", reference=VoiceReference(uri="voice.wav", transcript="Reference words"),
        seed=11, controls={},
    )
    with UniTTS(provider=provider_cls.name, device="cpu") as sdk:
        direct = sdk.synthesize(request.to_tts_request())
    engine = engine_cls()
    await engine.load(LoadContext(replica_id="test", device="cpu"))
    response = await engine.generate(request, InferenceContext(request_id=request.request_id))
    await engine.unload()
    assert requests[0] == requests[1]
    assert response.data == direct.audio
    assert response.metadata["warnings"] == direct.warnings


def test_qwen_namespaced_controls_reach_the_actual_model(monkeypatch):
    seed = Mock()
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=seed))
    provider = Qwen3TTSProvider(device="cpu")
    generate = Mock(return_value=([object()], 24000))
    provider._model = SimpleNamespace(generate_custom_voice=generate)
    provider._call_inference(TTSRequest(text="hello", seed=11, controls={"temperature": 0.4, "top_p": 0.8}))
    seed.assert_called_once_with(11)
    assert generate.call_args.kwargs["temperature"] == 0.4
    assert generate.call_args.kwargs["top_p"] == 0.8


@pytest.mark.parametrize("changes,match", [
    ({"protocol_version": "unknown"}, "protocol version"),
    ({"message_id": "stale-response"}, "does not match"),
])
def test_protocol_rejects_version_mismatch_and_stale_response(changes, match):
    request = request_message({"operation": "health"})
    response = {"protocol_version": PROTOCOL_VERSION, "message_id": request["message_id"], "ok": True, "result": {}}
    with pytest.raises(ValueError, match=match):
        read_response(json.dumps(response | changes), request["message_id"])


def test_generic_engine_process_uses_the_same_client_as_indextts(tmp_path, monkeypatch):
    monkeypatch.setenv("UNITTS_ENGINE_ID", "dummy")
    root = Path(__file__).resolve().parents[1]
    worker = WorkerProcess(
        [sys.executable, "-u", "-m", "unitts.services.engine_worker"],
        cwd=root, log_path=tmp_path / "generic-worker.log",
    )
    try:
        assert worker.call({"operation": "load", "context": {"replica_id": "test", "device": "cpu"}}, timeout=15)["loaded"]
        response = worker.call({"operation": "generate", "request": {"text": "hello", "engine_id": "dummy"}}, timeout=15)
        assert base64.b64decode(response["audio_base64"]).startswith(b"RIFF")
        assert worker.call({"operation": "unload"}, timeout=15)["unloaded"]
    finally:
        worker.close()
    assert worker.process.poll() is not None


def test_provider_discovery_never_advertises_sdk_streaming():
    assert not any(provider.capabilities.streaming for provider in ProviderRegistry.all().values())
