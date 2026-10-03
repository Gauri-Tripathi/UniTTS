"""Click command-line interface for UniTTS."""

from __future__ import annotations

import json
import logging
import shutil
from functools import wraps
from pathlib import Path

import click
import requests

from unitts import UniTTS
from unitts.core.config import init_config
from unitts.core.registry import ProviderRegistry
from unitts.core.schemas import OutputFormat, TTSRequest
from unitts.utils.downloader import hf_download


def user_errors(function):
    """Show actionable failures; --verbose retains the debugging traceback."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ImportError, ValueError, RuntimeError, OSError, requests.RequestException) as exc:
            if click.get_current_context().find_root().params.get("verbose"):
                raise
            raise click.ClickException(str(exc)) from exc
    return wrapped


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
def main(verbose: bool = False) -> None:
    """Universal Text-to-Speech command line tool."""

    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


@main.command()
@click.argument("provider", type=click.Choice(["indextts", "indextts-2.5", "index-tts"]))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--model-dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default="auto")
@user_errors
def setup(provider: str, cache_dir: Path | None, model_dir: Path | None, device: str) -> None:
    """Prepare a managed engine without changing this interpreter's packages."""
    with UniTTS(provider=provider, device=device, cache_dir=cache_dir, model_dir=model_dir) as engine:
        click.echo(json.dumps(engine.setup(), indent=2))


@main.command()
@click.argument("provider", type=click.Choice(["indextts", "indextts-2.5", "index-tts"]))
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--model-dir", type=click.Path(file_okay=False, path_type=Path))
@user_errors
def doctor(provider: str, cache_dir: Path | None, model_dir: Path | None) -> None:
    """Show managed paths and installation state without downloading models."""
    with UniTTS(provider=provider, cache_dir=cache_dir, model_dir=model_dir) as engine:
        click.echo(json.dumps(engine.status(), indent=2))


@main.command()
@click.argument("text", required=False)
@click.option("--file", "-f", "input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Read text from a file.")
@click.option("--provider", "-p", default="piper", show_default=True, help="Provider name.")
@click.option("--voice", help="Voice id or model voice name.")
@click.option("--language", help="Language code.")
@click.option("--reference-audio", type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Reference recording for voice cloning.")
@click.option("--emotion", help="Emotion supported by the selected provider.")
@click.option("--speed", type=float, help="Speech speed multiplier.")
@click.option("--pitch", type=float, help="Pitch in semitones where supported.")
@click.option("--volume", type=float, help="Volume multiplier where supported.")
@click.option("--format", "output_format", type=click.Choice([item.value for item in OutputFormat]), help="Audio format; inferred from --output or provider default.")
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), help="Output file (default: output.<format>).")
@user_errors
def synthesize(
    text: str | None,
    input_file: Path | None,
    provider: str,
    voice: str | None,
    language: str | None,
    reference_audio: Path | None,
    emotion: str | None,
    speed: float | None,
    pitch: float | None,
    volume: float | None,
    output_format: str | None,
    output: Path | None,
) -> None:
    """Synthesize text to an audio file."""

    if input_file:
        body = input_file.read_text(encoding="utf-8")
    elif text:
        body = text
    else:
        body = click.get_text_stream("stdin").read()
    if not body.strip():
        raise click.UsageError("Provide text, --file, or stdin input")

    formats = ProviderRegistry.get(provider).capabilities.output_formats
    suffix = output.suffix.lstrip(".").lower() if output else ""
    output_format = output_format or suffix or ("wav" if "wav" in formats else formats[0])
    if output_format not in formats:
        raise click.UsageError(f"Provider '{provider}' supports: {', '.join(formats)}. Choose --format and an output filename with that extension.")
    if suffix and suffix != output_format:
        raise click.UsageError(f"Output extension '.{suffix}' does not match --format {output_format}.")
    output = output or Path(f"output.{output_format}")
    request = TTSRequest(
        text=body.strip(),
        voice=voice,
        language=language,
        reference_audio=reference_audio,
        emotion=emotion,
        speed=speed,
        pitch=pitch,
        volume=volume,
        output_format=OutputFormat(output_format),
    )
    with UniTTS(provider=provider) as engine:
        response = engine.synthesize_to_file(request, output)
    for warning in response.warnings:
        click.echo(f"warning: {warning}", err=True)
    click.echo(str(output))


@main.command()
@click.option("--provider", "-p", required=True, help="Provider name.")
@user_errors
def voices(provider: str) -> None:
    """List voices for a provider."""

    with UniTTS(provider=provider) as engine:
        click.echo(json.dumps(engine.list_voices(), indent=2))


@main.command("providers")
def providers_command() -> None:
    """List registered providers and capabilities."""

    headers = ("provider", "type", "cloning", "emotion", "streaming", "formats")
    rows: list[tuple[str, ...]] = []
    for name, provider_cls in sorted(ProviderRegistry.all().items()):
        cap = provider_cls.capabilities
        rows.append((
            name,
            cap.provider_type.value,
            "yes" if cap.voice_cloning else "no",
            "yes" if cap.emotion else "no",
            "yes" if cap.streaming else "no",
            ", ".join(cap.output_formats),
        ))

    widths = [max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render(row: tuple[str, ...]) -> str:
        cells = [f" {value:<{width}} " for value, width in zip(row, widths, strict=True)]
        return "|" + "|".join(cells) + "|"

    click.echo(separator)
    click.echo(render(headers))
    click.echo(separator)
    for row in rows:
        click.echo(render(row))
    click.echo(separator)


@main.group()
def config() -> None:
    """Manage UniTTS configuration."""


@config.command("init")
def config_init() -> None:
    """Create ~/.unitts/config.yaml."""

    click.echo(str(init_config()))


@main.group()
def models() -> None:
    """Manage cached model files."""


@models.command("list")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path.home() / ".unitts" / "models", show_default=True)
def models_list(cache_dir: Path) -> None:
    """List files in the UniTTS model cache."""

    if not cache_dir.exists():
        click.echo("No model cache found.")
        return
    for path in sorted(item for item in cache_dir.rglob("*") if item.is_file()):
        click.echo(str(path.relative_to(cache_dir)))


@models.command("download")
@click.argument("repo_id")
@click.argument("filename")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path.home() / ".unitts" / "models", show_default=True)
def models_download(repo_id: str, filename: str, cache_dir: Path) -> None:
    """Download a model file from Hugging Face Hub."""

    click.echo(str(hf_download(repo_id, filename, cache_dir=cache_dir)))


@models.command("clear")
@click.option("--cache-dir", type=click.Path(file_okay=False, path_type=Path), default=Path.home() / ".unitts" / "models", show_default=True)
def models_clear(cache_dir: Path) -> None:
    """Clear the UniTTS model cache."""

    if not cache_dir.exists():
        click.echo("No model cache found.")
        return
    click.confirm(f"Delete cached model files under {cache_dir}?", abort=True)
    shutil.rmtree(cache_dir)
    click.echo("Model cache cleared.")


if __name__ == "__main__":
    main()
