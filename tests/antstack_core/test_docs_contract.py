"""Repository documentation signposting contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = ROOT / "tools" / "ensure_folder_docs.py"
SPEC = importlib.util.spec_from_file_location("ensure_folder_docs_for_tests", TOOLS_PATH)
assert SPEC is not None and SPEC.loader is not None
ensure_folder_docs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ensure_folder_docs)


def test_every_intentional_directory_has_local_docs() -> None:
    """All intentional directories should have both README.md and AGENTS.md."""
    assert ensure_folder_docs.missing_docs() == []


def test_missing_docs_flags_declared_but_absent_directories() -> None:
    """A declared-but-absent directory is itself reported as missing."""
    documented = [ROOT / rel for rel in ensure_folder_docs.DIRECTORIES]
    absent = [d for d in documented if not d.is_dir()]
    if absent:
        reported = ensure_folder_docs.missing_docs()
        for directory in absent:
            assert directory in reported


def test_missing_docs_reports_missing_readme_and_agents_in_a_tmp_tree(
    tmp_path: Path,
) -> None:
    """A real directory tree missing doc files is reported file-by-file."""
    spec2 = importlib.util.spec_from_file_location("ensure_folder_docs_tmp", TOOLS_PATH)
    loader = spec2.loader
    assert loader is not None
    fresh = importlib.util.module_from_spec(spec2)
    loader.exec_module(fresh)
    fresh.ROOT = tmp_path
    meta = {"role": "Test role.", "commands": "pytest"}
    fresh.DIRECTORIES = {"some/real/dir": meta}

    assert fresh.render_readme("some/real/dir", meta)
    assert fresh.render_agents("some/real/dir", meta)

    # A declared-but-absent directory is reported as the directory itself.
    absent_dir = tmp_path / "some" / "absent"
    fresh.DIRECTORIES["some/absent"] = meta
    assert absent_dir in fresh.missing_docs()

    # An existing directory missing the doc files is reported file-by-file.
    (tmp_path / "some" / "real" / "dir").mkdir(parents=True)
    fresh.DIRECTORIES = {"some/real/dir": meta}
    missing = fresh.missing_docs()
    assert (tmp_path / "some" / "real" / "dir" / "README.md") in missing
    assert (tmp_path / "some" / "real" / "dir" / "AGENTS.md") in missing

    # write_missing() must actually create the files and clear the report.
    written = fresh.write_missing()
    assert len(written) == 2
    assert (tmp_path / "some" / "real" / "dir" / "README.md").is_file()
    assert (tmp_path / "some" / "real" / "dir" / "AGENTS.md").is_file()
    assert fresh.missing_docs() == []


def test_doc_coverage_map_references_existing_directories() -> None:
    """The folder coverage map should not include stale paths."""
    stale = [rel for rel in ensure_folder_docs.DIRECTORIES if not (ROOT / rel).is_dir()]
    assert stale == []


