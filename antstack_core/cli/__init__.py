"""Command line entry points for Ant Stack workflows."""

from importlib import import_module

__all__ = ["build", "ce", "run_all"]


def __getattr__(name: str):
    """Load command modules on demand without preloading ``python -m`` targets."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
