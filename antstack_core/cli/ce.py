"""Console wrapper for the Complexity & Energetics analysis workflow."""

from __future__ import annotations

from antstack_core.orchestration import complexity_energetics


def main() -> None:
    """Run the canonical complexity energetics command."""
    complexity_energetics.main()


def run_manifest(*args, **kwargs):
    """Run a complexity energetics manifest through the canonical runner."""
    return complexity_energetics.run_manifest(*args, **kwargs)

__all__ = ["main", "run_manifest"]


if __name__ == "__main__":
    main()
