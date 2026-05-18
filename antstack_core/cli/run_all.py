"""Console wrapper for canonical Ant Stack run-all orchestration."""

from __future__ import annotations

from antstack_core.orchestration import main, run_all

__all__ = ["main", "run_all"]


if __name__ == "__main__":
    raise SystemExit(main())
