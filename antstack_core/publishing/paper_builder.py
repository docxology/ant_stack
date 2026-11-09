"""Configured paper builds through the installed document-scoped PDF renderer."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import yaml

from antstack_core.figures import validate_cross_references
from antstack_core.publishing.pdf_generator import PDFGenerationConfig, PDFGenerator
from antstack_core.publishing.provenance import atomic_text
from antstack_core.publishing.reference_manager import ReferenceManager


class ModularPaperBuilder:
    """Validate configured ordered sources and publish only a successful current render."""

    def __init__(self, project_root: Path | None = None):
        self.project_root = Path(project_root or Path.cwd()).absolute()
        self.papers_dir = self.project_root / "papers"
        self.build_results = {}
        self.validation_results = {}

    def discover_papers(self) -> list[str]:
        if not self.papers_dir.is_dir():
            return []
        return sorted(path.name for path in self.papers_dir.iterdir()
                      if path.is_dir() and (path / "paper_config.yaml").is_file())

    def load_paper_config(self, paper_name: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", paper_name):
            raise ValueError("Paper name must be one ordinary directory name")
        path = self.papers_dir / paper_name / "paper_config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Paper config must be a mapping")
        for section in ("paper", "content", "build", "latex"):
            if not isinstance(config.get(section), dict):
                raise ValueError(f"Missing or invalid config section: {section}")
        files = config["content"].get("files")
        if not isinstance(files, list) or not files or any(not isinstance(name, str) or not name for name in files):
            raise ValueError("content.files must be a nonempty list of paths")
        if len(set(files)) != len(files):
            raise ValueError("content.files contains duplicate sources")
        paper = path.parent.resolve()
        for name in files:
            candidate = Path(name)
            if candidate.is_absolute() or not (paper / candidate).resolve().is_relative_to(paper):
                raise ValueError(f"Content source escapes paper directory: {name}")
        for name in ("has_computational_analysis", "has_generated_content", "mermaid_preprocessing", "cross_reference_validation"):
            if name in config["build"] and not isinstance(config["build"][name], bool):
                raise ValueError(f"build.{name} must be a boolean")
        filename = config["paper"].get("output_filename", self._default_pdf_name(paper_name))
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".pdf"):
            raise ValueError("paper.output_filename must be one PDF filename")
        return config

    @staticmethod
    def _default_pdf_name(paper_name: str) -> str:
        return {"ant_stack": "1_ant_stack.pdf", "complexity_energetics": "2_complexity_energetics.pdf",
                "documentation": "1_ant_stack_documentation.pdf"}.get(paper_name, paper_name + ".pdf")

    def _get_expected_pdf_path(self, paper_name: str) -> Path:
        config = self.load_paper_config(paper_name)
        return self.project_root / config["paper"].get("output_filename", self._default_pdf_name(paper_name))

    def _validate_output_destinations(self, paper_names: Sequence[str]) -> None:
        destinations: dict[Path, str] = {}
        for paper_name in sorted(set(paper_names)):
            destination = self._get_expected_pdf_path(paper_name).resolve()
            if destination in destinations:
                raise ValueError(
                    f"Duplicate PDF output destination for papers '{destinations[destination]}' "
                    f"and '{paper_name}': {destination}")
            destinations[destination] = paper_name

    def validate_paper_environment(self, paper_name: str) -> dict[str, Any]:
        result = {"paper_name": paper_name, "valid": True, "issues": [], "warnings": []}
        try:
            config = self.load_paper_config(paper_name)
            paper = self.papers_dir / paper_name
            content = []
            for name in config["content"]["files"]:
                path = paper / name
                if not path.is_file():
                    result["issues"].append(f"Missing content file: {name}")
                else:
                    content.append(path.read_text(encoding="utf-8"))
            if content:
                references = validate_cross_references("\n\n".join(content), str(paper))
                result["issues"].extend(references["issues"])
                result["warnings"].extend(references["warnings"])
            bibliography = config["latex"].get("bibliography")
            if bibliography and not (paper / bibliography).is_file():
                result["issues"].append(f"Missing bibliography: {bibliography}")
            if bibliography and (paper / bibliography).is_file() and len(content) == len(config["content"]["files"]):
                citations = ReferenceManager().validate_citations(
                    [paper / name for name in config["content"]["files"]], paper / bibliography)
                result["issues"].extend(citations["errors"])
        except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
            result["issues"].append(str(error))
        result["valid"] = not result["issues"]
        self.validation_results[paper_name] = result
        return result

    def run_analysis_pipeline(self, paper_name: str, config: dict[str, Any]) -> bool:
        if not config["build"].get("has_computational_analysis", False):
            return True
        analysis = config.get("analysis", {})
        if analysis.get("runner_module") != "antstack_core.orchestration.complexity_energetics":
            raise ValueError("Unsupported analysis runner; configure the installed complexity energetics module")
        from antstack_core.orchestration.complexity_energetics import run_manifest, project_result
        paper = self.papers_dir / paper_name
        manifest = (paper / analysis.get("manifest_file", "manifest.example.yaml")).resolve()
        if analysis.get("output_dir", "out") != "out":
            raise ValueError("Configured paper projection output_dir must be out")
        if not manifest.is_relative_to(paper.resolve()):
            raise ValueError("Analysis manifest must stay within the paper directory")
        cache = self.project_root / "cache" / "paper-build"
        cache.mkdir(parents=True, exist_ok=True)
        fresh = Path(tempfile.mkdtemp(prefix=paper_name + "-", dir=cache)) / "out"
        generated = run_manifest(str(manifest), str(fresh))
        project_result(generated, paper)
        return True

    def build_paper(self, paper_name: str, validate_only: bool = False) -> tuple[bool, dict[str, Any]]:
        try:
            self._validate_output_destinations([*self.discover_papers(), paper_name])
        except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as error:
            result = {"success": False, "error": str(error)}
            self.build_results[paper_name] = result
            return False, result
        return self._build_paper(paper_name, validate_only)

    def _build_paper(self, paper_name: str, validate_only: bool) -> tuple[bool, dict[str, Any]]:
        try:
            config = self.load_paper_config(paper_name)
            if not validate_only:
                self.run_analysis_pipeline(paper_name, config)
            validation = self.validate_paper_environment(paper_name)
            if not validation["valid"]:
                result = {"success": False, "error": "Validation failed", "validation": validation}
            elif validate_only:
                result = {"success": True, "validation": validation, "mode": "validation-only"}
            else:
                paper = self.papers_dir / paper_name
                latex = config["latex"]
                arguments = ["--toc"]
                for name in ("geometry", "mainfont", "mathfont", "monofont"):
                    if latex.get(name):
                        arguments.extend(["--variable", f"{name}:{latex[name]}"])
                if latex.get("document_class"):
                    arguments.extend(["--variable", f"documentclass:{latex['document_class']}"])
                if latex.get("bibliography"):
                    arguments.extend(["--bibliography", str((paper / latex["bibliography"]).resolve()), "--citeproc"])
                renderer = PDFGenerator(PDFGenerationConfig(
                    pandoc_args=arguments,
                    enable_mermaid=config["build"].get("mermaid_preprocessing", True),
                    enable_crossrefs=config["build"].get("cross_reference_validation", True),
                ))
                metadata = {key: config["paper"][key] for key in ("title", "author", "date", "subtitle") if key in config["paper"]}
                rendered = renderer.generate_pdf([paper / name for name in config["content"]["files"]],
                                                  self._get_expected_pdf_path(paper_name), metadata)
                result = {"success": rendered.success, "validation": validation,
                          "output_path": str(rendered.output_path) if rendered.output_path else None,
                          "error": rendered.error_message, "warnings": rendered.warnings,
                          "source_binding": rendered.metadata}
        except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as error:
            result = {"success": False, "error": str(error)}
        self.build_results[paper_name] = result
        return result["success"], result

    def build_all_papers(self, validate_only: bool = False) -> dict[str, Any]:
        papers = self.discover_papers()
        results = {"success": bool(papers), "papers_built": 0, "papers_failed": 0,
                   "build_results": {}, "timestamp": datetime.now(timezone.utc).isoformat(),
                   "mode": "validation-only" if validate_only else "render"}
        if not papers:
            results["error"] = "No papers found"
        try:
            self._validate_output_destinations(papers)
        except (OSError, ValueError, TypeError, RuntimeError, yaml.YAMLError) as error:
            results["error"] = str(error)
            results["papers_failed"] = len(papers)
            results["build_results"] = {paper: {"success": False, "error": str(error)} for paper in papers}
            self.build_results.update(results["build_results"])
        else:
            for paper in papers:
                success, result = self._build_paper(paper, validate_only)
                results["build_results"][paper] = result
                results["papers_built" if success else "papers_failed"] += 1
        results["success"] = bool(papers) and results["papers_failed"] == 0
        self._generate_build_report(results)
        return results

    def _generate_build_report(self, results: dict[str, Any]) -> None:
        path = self.project_root / "build_report.md"
        lines = ["# Build Report", "", f"Generated: {results['timestamp']}", f"Mode: {results['mode']}",
                 f"Status: {'SUCCESS' if results['success'] else 'FAILED'}", ""]
        for name, result in results["build_results"].items():
            lines.extend([f"## {name}", "", f"Status: {'SUCCESS' if result['success'] else 'FAILED'}"])
            if result.get("error"):
                lines.append(f"Error: {result['error']}")
            for issue in result.get("validation", {}).get("issues", []):
                lines.append(f"Error: {issue}")
            for warning in result.get("warnings", []):
                lines.append(f"Warning: {warning}")
            lines.append("")
        if results.get("error"):
            lines.extend([results["error"], ""])
        atomic_text(path, "\n".join(lines))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the configured paper builder; failures always reach the process status."""
    parser = argparse.ArgumentParser(description="Build configured Ant Stack papers with Pandoc and XeLaTeX.")
    parser.add_argument("--paper", help="Build one configured paper")
    parser.add_argument("--validate-only", action="store_true", help="Validate current source inputs without rendering")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    builder = ModularPaperBuilder(arguments.project_root)
    if arguments.paper:
        success, result = builder.build_paper(arguments.paper, arguments.validate_only)
        print(json.dumps(result, indent=2))
    else:
        result = builder.build_all_papers(arguments.validate_only)
        success = result["success"]
        print(json.dumps(result, indent=2))
    return 0 if success else 1


__all__ = ["ModularPaperBuilder", "main"]

if __name__ == "__main__":
    sys.exit(main())
