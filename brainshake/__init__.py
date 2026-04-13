"""Development shim for the src-layout package.

This repository uses a ``src/`` layout. When running from the repo root without
installing the package (e.g. without ``pip install -e .``), Python will not find
``src/brainshake`` on its import path.

This shim makes ``import brainshake`` and ``python -m brainshake ...`` work
directly from the repository root by extending the package search path to include
``src/brainshake``.

When the project is installed, the regular package in ``src/brainshake`` is used
as usual.
"""

from __future__ import annotations

from pathlib import Path
import pkgutil

# Extend this package's module search path with the src-layout package directory.
__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]

_src_pkg = Path(__file__).resolve().parent.parent / "src" / "brainshake"
if _src_pkg.is_dir():
    try:
        __path__.append(str(_src_pkg))  # type: ignore[attr-defined]
    except AttributeError:
        __path__ = list(__path__) + [str(_src_pkg)]
