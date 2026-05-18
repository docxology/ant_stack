"""Tests for generated artifact provenance helpers."""

from __future__ import annotations

import json

from antstack_core.publishing import (
    ProvenanceRecord,
    build_run_provenance,
    collect_dependency_versions,
    write_provenance,
)


def test_build_run_provenance_has_required_contract_fields(tmp_path) -> None:
    """Provenance records should describe command, inputs, outputs, and environment."""
    record = build_run_provenance(
        command=["antstack-ce", "manifest.yaml", "--out", "out"],
        input_paths=["manifest.yaml"],
        output_paths=[tmp_path / "results.csv"],
        parameters={"seed": 123},
        cwd=tmp_path,
        dependency_names=("pytest", "definitely-not-installed-antstack-package"),
        created_at_utc="2026-05-18T00:00:00+00:00",
    )

    payload = record.to_dict()
    assert payload["project"] == "ant-stack"
    assert payload["command"] == ["antstack-ce", "manifest.yaml", "--out", "out"]
    assert payload["input_paths"] == ["manifest.yaml"]
    assert payload["output_paths"] == [str(tmp_path / "results.csv")]
    assert payload["parameters"] == {"seed": 123}
    assert payload["created_at_utc"] == "2026-05-18T00:00:00+00:00"
    assert payload["package_version"]
    assert payload["python_version"]
    assert "pytest" in payload["dependencies"]


def test_write_provenance_round_trip(tmp_path) -> None:
    """Provenance JSON should round-trip as parseable structured data."""
    record = ProvenanceRecord(
        project="ant-stack",
        package_version="1.0.0",
        created_at_utc="2026-05-18T00:00:00+00:00",
        command=["cmd"],
        input_paths=["input.yaml"],
        output_paths=["output.csv"],
    )

    path = write_provenance(tmp_path / "nested" / "provenance.json", record)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert loaded["project"] == "ant-stack"
    assert loaded["input_paths"] == ["input.yaml"]
    assert loaded["output_paths"] == ["output.csv"]


def test_collect_dependency_versions_omits_missing_packages() -> None:
    """Dependency collection should be best-effort and omit missing packages."""
    versions = collect_dependency_versions(("pytest", "definitely-not-installed-antstack-package"))

    assert "pytest" in versions
    assert "definitely-not-installed-antstack-package" not in versions
