"""CLI subprocess tests for the three console-script entrypoints.

Real subprocess execution against the project venv interpreters — no mocks.
--help / --validate-only paths are used so runs stay fast and side-effect free.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *args: str, timeout: int = 120) -> subprocess.CompletedProcess:
    python = Path(sys.executable)
    # Prefer the project venv console-script environment when present.
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    interpreter = venv_python if venv_python.is_file() else python
    return subprocess.run(
        [str(interpreter), "-m", "antstack_core.cli." + script, *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=timeout,
    )


class TestAntstackBuild:
    def test_help_exits_zero_and_documents_flags(self) -> None:
        result = _run("build", "--help")
        assert result.returncode == 0, result.stderr
        for flag in ("--paper", "--validate-only", "--no-tests", "--project-root"):
            assert flag in result.stdout

    def test_module_help_matches_console_script_contract(self) -> None:
        """python -m antstack_core.cli.build --help is the console-script target."""
        result = _run("build", "--help")
        assert "Modular Scientific Publication Build System" in result.stdout


class TestAntstackCe:
    def test_help_exits_zero_and_documents_flags(self) -> None:
        result = _run("ce", "--help")
        assert result.returncode == 0, result.stderr
        for flag in ("--out", "--validate-only", "--verbose"):
            assert flag in result.stdout

    def test_validate_only_rejects_missing_manifest(self, tmp_path: Path) -> None:
        result = _run("ce", str(tmp_path / "ghost.yaml"), "--validate-only")
        assert result.returncode != 0

    def test_validate_only_accepts_example_manifest(self) -> None:
        manifest = PROJECT_ROOT / "papers" / "complexity_energetics" / "manifest.example.yaml"
        if not manifest.is_file():
            import pytest
            pytest.skip("example manifest not present")
        result = _run(
            "ce", str(manifest), "--validate-only",
            "--out", str(PROJECT_ROOT / "papers" / "complexity_energetics" / "out"),
        )
        assert result.returncode == 0, result.stderr


class TestRunAllAntstack:
    def test_help_exits_zero_and_documents_flags(self) -> None:
        result = _run("run_all", "--help")
        assert result.returncode == 0, result.stderr
        for flag in ("--config", "--out", "--run-id", "--validate-only", "--tasks"):
            assert flag in result.stdout

    def test_validate_only_accepts_example_config(self) -> None:
        config = PROJECT_ROOT / "configs" / "run_all_antstack.example.yaml"
        if not config.is_file():
            import pytest
            pytest.skip("example config not present")
        result = _run("run_all", "--config", str(config), "--validate-only")
        assert result.returncode == 0, result.stderr
