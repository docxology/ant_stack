"""Console wrapper for the Complexity & Energetics analysis workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    module_path = ROOT / "papers" / "complexity_energetics" / "src" / "runner.py"
    spec = importlib.util.spec_from_file_location("antstack_ce_runner", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load complexity energetics runner from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Run the canonical complexity energetics command."""
    _load_runner().main()


def run_manifest(*args, **kwargs):
    """Run a complexity energetics manifest through the canonical runner."""
    return _load_runner().run_manifest(*args, **kwargs)

__all__ = ["main", "run_manifest"]


if __name__ == "__main__":
    main()
