from pathlib import Path

import pytest
import yaml

from unitts.contracts import (
    EngineManifest,
    ProtocolVersionError,
    SynthesisRequest,
    UnsupportedCapabilityError,
)
from unitts.engines.dummy import DummyEngine
from unitts.runtime.registry import EngineRegistry


def test_synthesis_request_rejects_blank_text_and_unknown_fields() -> None:
    with pytest.raises(ValueError):
        SynthesisRequest(text="   ")
    with pytest.raises(ValueError):
        SynthesisRequest(text="hello", temperature=1.0)


def test_manifest_validation_and_protocol_rejection() -> None:
    manifest = DummyEngine.manifest()
    with pytest.raises(ProtocolVersionError):
        manifest.assert_protocol("99.0")
    with pytest.raises(ValueError):
        EngineManifest.model_validate({"engine_id": "Bad ID"})


def test_catalogue_contains_remaining_model_families_and_dummy() -> None:
    registry = EngineRegistry()
    registry.discover([Path("engines")])
    public = registry.list()
    model_families = [item for item in public if item.engine_id != "dummy"]
    assert any(item.engine_id == "dummy" for item in public)
    assert {item.family for item in model_families} == {
        "Qwen3-TTS", "F5-TTS", "Fish Audio S2 Pro",
        "Fun-CosyVoice3", "IndexTTS-2.5", "XTTS-v2",
    }


def test_all_manifests_validate() -> None:
    for path in Path("engines").glob("*/manifest.yaml"):
        EngineManifest.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def test_dummy_rejects_unvalidated_engine_controls() -> None:
    with pytest.raises(UnsupportedCapabilityError):
        DummyEngine().validate_controls(SynthesisRequest(text="hello", controls={"steps": 3}))
