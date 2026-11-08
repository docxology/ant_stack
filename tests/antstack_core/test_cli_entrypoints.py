"""Consumer-visible command failures and validation side effects."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import yaml


def _validate_paper(project: Path, paper: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "antstack_core.cli.build", "--project-root", str(project),
         "--paper", paper, "--validate-only"],
        cwd=project, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30,
    )


def test_missing_paper_fails_process_without_writing_outputs(tmp_path: Path) -> None:
    result = _validate_paper(tmp_path, "absent")
    assert result.returncode == 1
    assert list(tmp_path.iterdir()) == []


def test_missing_configured_content_is_not_a_successful_validation(tmp_path: Path) -> None:
    paper = tmp_path / "papers" / "example"
    paper.mkdir(parents=True)
    config = {"paper": {"title": "Example"}, "content": {"files": ["Methods.md"]},
              "build": {}, "latex": {}}
    (paper / "paper_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = _validate_paper(tmp_path, "example")
    assert result.returncode == 1
    assert not (tmp_path / "example.pdf").exists()
    assert not (tmp_path / "build_report.md").exists()


def test_valid_single_paper_validation_does_not_render(tmp_path: Path) -> None:
    paper = tmp_path / "papers" / "example"
    paper.mkdir(parents=True)
    config = {"paper": {"title": "Example"}, "content": {"files": ["Methods.md"]},
              "build": {}, "latex": {}}
    (paper / "paper_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (paper / "Methods.md").write_text("# Methods\n\nAn analytic example.\n", encoding="utf-8")
    result = _validate_paper(tmp_path, "example")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "example.pdf").exists()
    assert not (tmp_path / "build_report.md").exists()
