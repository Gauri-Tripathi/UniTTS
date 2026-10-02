"""Packaged entrypoint for the managed IndexTTS-2.5 adapter."""

from unitts.engines.indextts import IndexTTSEngine

STATUS = "experimental"
__all__ = ["IndexTTSEngine", "STATUS"]
