"""Behavioral tests for antstack_core.figures.assets asset management.

Uses real files under tmp_path: directory scaffolding, registration with
freshness-aware copying, manifest generation, validation, and cleanup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from antstack_core.figures.assets import (
    AssetManager,
    copy_figure_files,
    organize_figure_assets,
)


def _make_image(directory: Path, name: str = "fig1.png", payload: bytes = b"\x89PNG\r\n\x1a\ndata") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / name
    image.write_bytes(payload)
    return image


class TestAssetManagerSetup:
    def test_creates_standard_structure_with_gitkeep(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        assert (tmp_path / "assets").is_dir()
        assert (tmp_path / "assets" / "figures").is_dir()
        assert (tmp_path / "assets" / "mermaid").is_dir()
        assert (tmp_path / "assets" / "tmp_images").is_dir()
        # .gitkeep is only added to empty directories; assets/ itself contains
        # subdirectories at setup time, so only the leaf dirs get one.
        for leaf in ("figures", "mermaid", "tmp_images"):
            assert (tmp_path / "assets" / leaf / ".gitkeep").is_file()
        assert not (tmp_path / "assets" / ".gitkeep").exists()

    def test_gitkeep_not_duplicated(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        manager.setup_directory_structure()
        gitkeeps = list((tmp_path / "assets").rglob(".gitkeep"))
        assert len(gitkeeps) == 3


class TestRegisterAsset:
    def test_register_figure_copies_into_figures_dir(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        rel = manager.register_asset(source, "figure")
        assert rel == "assets/figures/fig1.png"
        assert (tmp_path / rel).is_file()
        assert (tmp_path / rel).read_bytes() == source.read_bytes()

    def test_register_mermaid_goes_to_mermaid_dir(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming", "diagram.svg")
        rel = manager.register_asset(source, "mermaid")
        assert rel == "assets/mermaid/diagram.svg"

    def test_register_tmp_goes_to_tmp_dir(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming", "scratch.png")
        rel = manager.register_asset(source, "tmp")
        assert rel == "assets/tmp_images/scratch.png"

    def test_unknown_type_defaults_to_assets_dir(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming", "table.csv", b"a,b,c\n1,2,3\n")
        rel = manager.register_asset(source, "data")
        assert rel == "assets/table.csv"

    def test_metadata_recorded_in_manifest(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        rel = manager.register_asset(source, "figure", metadata={"caption": "Ant workers"})
        info = manager.asset_manifest[rel]
        assert info["metadata"] == {"caption": "Ant workers"}
        assert info["type"] == "figure"
        assert info["size_bytes"] == source.stat().st_size

    def test_reregister_with_older_source_keeps_existing_copy(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        rel = manager.register_asset(source, "figure")
        first = (tmp_path / rel).read_bytes()
        # Overwrite the source with different content but force an older mtime.
        import os
        source.write_bytes(b"changed")
        older = source.stat().st_mtime - 10_000
        os.utime(source, (older, older))
        manager.register_asset(source, "figure")
        assert (tmp_path / rel).read_bytes() == first

    def test_reregister_with_newer_source_refreshes_copy(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        rel = manager.register_asset(source, "figure")
        source.write_bytes(b"\x89PNG\r\n\x1a\nnew content")
        import os
        newer = source.stat().st_mtime + 10_000
        os.utime(source, (newer, newer))
        manager.register_asset(source, "figure")
        assert (tmp_path / rel).read_bytes() == source.read_bytes()


class TestOrganizeFigureAssets:
    def test_scans_and_categorizes_images(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        src = tmp_path / "src"
        _make_image(src, "photo.png")
        _make_image(src, "render_mermaid.svg")
        _make_image(src, "chart_plot.jpg")
        organized = manager.organize_figure_assets(src)
        assert len(organized) == 3
        types = {info["type"] for info in manager.asset_manifest.values()}
        assert types == {"figure", "mermaid", "plot"}

    def test_skips_files_already_under_assets(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        _make_image(tmp_path / "assets" / "figures", "existing.png")
        assert manager.organize_figure_assets() == []


class TestCopyFigureFiles:
    def test_exact_paths(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        _make_image(tmp_path / "src", "a.png")
        _make_image(tmp_path / "src", "b.png")
        copied = manager.copy_figure_files(["src/a.png", "src/b.png"])
        assert set(copied) == {str(tmp_path / "src" / "a.png"), str(tmp_path / "src" / "b.png")}
        assert (tmp_path / "assets" / "figures" / "a.png").is_file()

    def test_glob_pattern(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        _make_image(tmp_path / "src", "a.png")
        _make_image(tmp_path / "src", "b.png")
        _make_image(tmp_path / "src", "c.txt", b"text")
        copied = manager.copy_figure_files(["src/*.png"])
        assert set(copied) == {str(tmp_path / "src" / "a.png"), str(tmp_path / "src" / "b.png")}

    def test_missing_file_is_silently_skipped(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        assert manager.copy_figure_files(["src/ghost.png"]) == {}


class TestManifestAndValidation:
    def test_manifest_writes_json(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        manager.register_asset(source, "figure")
        out = tmp_path / "manifest.json"
        manifest = manager.generate_asset_manifest(out)
        assert manifest["asset_count"] == 1
        assert manifest["total_size_bytes"] == source.stat().st_size
        on_disk = json.loads(out.read_text())
        assert on_disk["asset_count"] == manifest["asset_count"]

    def test_validate_reports_missing_asset(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        rel = manager.register_asset(source, "figure")
        (tmp_path / rel).unlink()
        report = manager.validate_assets()
        assert report["valid"] is False
        assert any("Missing asset" in issue for issue in report["issues"])

    def test_validate_reports_size_mismatch(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        rel = manager.register_asset(source, "figure")
        (tmp_path / rel).write_bytes(b"\x89PNG\r\n\x1a\ntampered")
        report = manager.validate_assets()
        assert report["valid"] is True  # only a warning
        assert any("Size mismatch" in w for w in report["warnings"])

    def test_validate_reports_zero_size(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming", payload=b"")
        rel = manager.register_asset(source, "figure")
        report = manager.validate_assets()
        assert any("Zero-size" in w for w in report["warnings"])
        assert report["validated_assets"] == 1


class TestCleanup:
    def test_cleanup_removes_tmp_and_orphans(self, tmp_path: Path) -> None:
        manager = AssetManager(tmp_path)
        manager.setup_directory_structure()
        source = _make_image(tmp_path / "incoming")
        managed_rel = manager.register_asset(source, "figure")
        # tmp file + orphan figure (not registered)
        _make_image(tmp_path / "assets" / "tmp_images", "scratch.png")
        _make_image(tmp_path / "assets" / "figures", "orphan.png")
        removed = manager.cleanup_temporary_assets()
        # 2 added files + the pre-existing leaf .gitkeep files (3) count as removed too
        assert removed == 5
        assert not (tmp_path / "assets" / "tmp_images" / "scratch.png").exists()
        assert not (tmp_path / "assets" / "figures" / "orphan.png").exists()
        assert (tmp_path / managed_rel).is_file()


class TestModuleLevelFunctions:
    def test_organize_with_markdown_copies_referenced_images(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _make_image(src, "hero.png")
        markdown = "# Title\n\n![hero](hero.png)\n"
        result = organize_figure_assets(
            markdown_content=markdown, source_dir=src, output_dir=tmp_path / "out"
        )
        assert result is True
        assert (tmp_path / "out" / "hero.png").is_file()

    def test_organize_markdown_requires_both_dirs(self, tmp_path: Path) -> None:
        result = organize_figure_assets(markdown_content="![x](a.png)", source_dir=tmp_path)
        assert result is False

    def test_organize_markdown_missing_source_dir(self, tmp_path: Path) -> None:
        result = organize_figure_assets(
            markdown_content="![x](a.png)",
            source_dir=tmp_path / "nope",
            output_dir=tmp_path / "out",
        )
        assert result is False

    def test_organize_requires_base_dir_without_kwargs(self) -> None:
        assert organize_figure_assets() is False

    def test_copy_figure_files_file_list_mode(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        _make_image(src, "a.png")
        result = copy_figure_files(
            file_list=["a.png"], source_dir=src, dest_dir=tmp_path / "out"
        )
        assert result is True
        assert (tmp_path / "out" / "a.png").is_file()

    def test_copy_figure_files_file_list_requires_dirs(self, tmp_path: Path) -> None:
        assert copy_figure_files(file_list=["a.png"], source_dir=tmp_path) is False

    def test_copy_figure_files_legacy_mode(self, tmp_path: Path) -> None:
        _make_image(tmp_path, "a.png")
        copied = copy_figure_files(
            source_files=[str(tmp_path / "a.png")], dest_dir=tmp_path / "assets_root"
        )
        assert copied and (tmp_path / "assets_root" / "assets" / "figures" / "a.png").is_file()

    def test_copy_figure_files_legacy_requires_args(self) -> None:
        assert copy_figure_files() == {}
