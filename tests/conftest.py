"""Shared pytest fixtures for deterministic test-suite hygiene."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
    """Close figures after each test so visualization tests do not leak state."""
    yield
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.close("all")
