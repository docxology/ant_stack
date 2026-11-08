"""Console wrapper for the shared Ant Stack paper build pipeline."""

from __future__ import annotations

import sys

from antstack_core.publishing.paper_builder import main as build_main


def main() -> int:
    """Build papers in the selected project and preserve the failure exit status."""
    return build_main()


__all__ = ["main"]


if __name__ == "__main__":
    sys.exit(main())
