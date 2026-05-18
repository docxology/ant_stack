#!/usr/bin/env python3
"""Thin script wrapper for package-owned Ant Stack run-all orchestration."""

from __future__ import annotations

from antstack_core.orchestration import main


if __name__ == "__main__":
    raise SystemExit(main())
