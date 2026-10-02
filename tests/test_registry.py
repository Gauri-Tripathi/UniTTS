"""Provider registry tests."""

from unitts.core.registry import ProviderRegistry


def test_builtin_providers_register() -> None:
    providers = ProviderRegistry.all()
    assert "piper" in providers
    assert "openai" in providers
    assert "dia" in providers
    assert "xtts" in providers
    assert "qwen3-tts" in providers
