"""Console wrapper for the shared Ant Stack paper build pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_build_main():
    module_path = ROOT / "scripts" / "common_pipeline" / "build_core.py"
    spec = importlib.util.spec_from_file_location("antstack_build_core", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load build pipeline from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    """Run the canonical paper build command."""
    _load_build_main()()


__all__ = ["main"]


if __name__ == "__main__":
    main()
