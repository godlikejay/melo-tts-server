"""
Runtime compatibility helpers for Melo TTS tooling.

Currently ensures Python 3.9's stdlib `importlib.metadata` exposes
`packages_distributions`, which is required by newer huggingface-hub releases.
"""
from __future__ import annotations

try:  # pragma: no cover - depends on interpreter version
    from importlib import metadata as _stdlib_metadata
except ImportError:  # pragma: no cover
    _stdlib_metadata = None

try:  # pragma: no cover - optional backport
    import importlib_metadata as _backport_metadata  # type: ignore
except ImportError:  # pragma: no cover
    _backport_metadata = None  # type: ignore

_PATCHED = False


def ensure_importlib_metadata_patch() -> None:
    """Make sure stdlib metadata exposes packages_distributions on Python 3.9."""
    global _PATCHED
    if _PATCHED:
        return
    if (
        _stdlib_metadata
        and _backport_metadata
        and not hasattr(_stdlib_metadata, "packages_distributions")
    ):
        _stdlib_metadata.packages_distributions = (
            _backport_metadata.packages_distributions  # type: ignore[attr-defined]
        )
    _PATCHED = True


# Patch immediately on import so callers can simply `import compat`.
ensure_importlib_metadata_patch()


__all__ = ["ensure_importlib_metadata_patch"]
