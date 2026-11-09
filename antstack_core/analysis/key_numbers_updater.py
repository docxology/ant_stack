"""Canonicalize explicit manuscript metric placeholders without rewriting literals."""

import re
from pathlib import Path
from typing import Dict, List, Optional

from antstack_core.analysis.key_numbers import KeyNumbersLoader, KeyNumbersManager


class PaperKeyNumbersUpdater:
    """Updates paper sections with key numbers placeholders."""

    def __init__(self, project_root: Optional[Path] = None, *, json_path: Optional[Path] = None):
        """Bind inputs to the caller's project or explicit enhanced JSON path."""
        self.project_root = Path.cwd() if project_root is None else Path(project_root)
        source = json_path if json_path is not None else (
            self.project_root / "papers/complexity_energetics/generated_content/key_numbers.json"
        )
        self.key_manager = KeyNumbersManager(KeyNumbersLoader(source))

    def _update_semantic_placeholders(self, file_path: Path) -> bool:
        """Canonicalize authored metric placeholders; never infer meaning from numbers.

        For example scaling.body.exponent maps to scaling_exponents.body_energy.
        Literal measurements, uncertainty ranges, coefficients and mathematical
        notation are left unchanged. Authors must identify the metric explicitly.
        """
        content = file_path.read_text(encoding="utf-8")

        def canonicalize(match):
            parts = match.group(1).split(":", 1)
            path = self.key_manager.resolve_key_path(parts[0].strip())
            spec = ":" + parts[1].strip() if len(parts) == 2 else ""
            if self.key_manager.loader.get_value(path) is None:
                raise KeyError(f"Missing key-number metric: {path}")
            self.key_manager.loader.get_formatted_value(path, parts[1].strip() if len(parts) == 2 else ".3f")
            return "{{key_numbers." + path + spec + "}}"

        updated = re.sub(r"\{\{key_numbers\.([^}]+)\}\}", canonicalize, content)
        if updated != content:
            file_path.write_text(updated, encoding="utf-8")
        return True

    def update_abstract(self, file_path: Path) -> bool:
        return self._update_semantic_placeholders(file_path)

    def update_energetics(self, file_path: Path) -> bool:
        return self._update_semantic_placeholders(file_path)

    def update_results(self, file_path: Path) -> bool:
        return self._update_semantic_placeholders(file_path)

    def update_background(self, file_path: Path) -> bool:
        return self._update_semantic_placeholders(file_path)

    def update_complexity(self, file_path: Path) -> bool:
        return self._update_semantic_placeholders(file_path)

    def update_paper_section(self, section_name: str, file_path: Path) -> bool:
        """Update a specific paper section with key numbers placeholders."""
        updater_map = {
            'abstract': self.update_abstract,
            'energetics': self.update_energetics,
            'results': self.update_results,
            'background': self.update_background,
            'complexity': self.update_complexity
        }

        if section_name.lower() in updater_map:
            return updater_map[section_name.lower()](file_path)
        else:
            print(f"No updater available for section: {section_name}")
            return False

    def update_all_paper_sections(self, paper_name: str = "complexity_energetics") -> Dict[str, bool]:
        """Update all paper sections for a given paper."""
        paper_dir = self.project_root / "papers" / paper_name
        results = {}

        section_files = {
            'abstract': paper_dir / "Abstract.md",
            'background': paper_dir / "Background.md",
            'complexity': paper_dir / "Complexity.md",
            'energetics': paper_dir / "Energetics.md",
            'results': paper_dir / "Results.md"
        }

        for section_name, file_path in section_files.items():
            if file_path.exists():
                try:
                    success = self.update_paper_section(section_name, file_path)
                    results[section_name] = success
                    print(f"Updated {section_name}: {'✓' if success else '✗'}")
                except Exception as e:
                    print(f"Error updating {section_name}: {e}")
                    results[section_name] = False
            else:
                print(f"Section file not found: {file_path}")
                results[section_name] = False

        return results

    def validate_key_numbers_placeholders(self, file_path: Path) -> Dict[str, List[str]]:
        """Validate exact paths and format specs against the producer schema."""
        content = file_path.read_text(encoding="utf-8")
        results = {'valid': [], 'invalid': [], 'missing': []}
        for placeholder in re.findall(r"\{\{key_numbers\.([^}]+)\}\}", content):
            parts = placeholder.split(":", 1)
            try:
                path = self.key_manager.resolve_key_path(parts[0].strip())
                if self.key_manager.loader.get_value(path) is None:
                    results['invalid'].append(placeholder)
                    continue
                self.key_manager.loader.get_formatted_value(path, parts[1].strip() if len(parts) == 2 else ".3f")
                results['valid'].append(placeholder)
            except KeyError:
                results['missing'].append(placeholder)
            except (ValueError, TypeError):
                results['invalid'].append(placeholder)
        return results


def run(args) -> int:
    """Execute against caller-owned paths, returning a failure-honest exit status."""
    try:
        updater = PaperKeyNumbersUpdater()
        if args.validate and not args.section:
            print("Specify --section for validation mode")
            return 1

        if args.section:
            paper_dir = updater.project_root / "papers" / args.paper
            section_file = paper_dir / f"{args.section.capitalize()}.md"
            if not section_file.is_file():
                print(f"Section file not found: {section_file}")
                return 1
            if args.validate:
                results = updater.validate_key_numbers_placeholders(section_file)
                print(f"Validation results for {args.section}:")
                for category in ("valid", "invalid", "missing"):
                    print(f"  {category.capitalize()}: {len(results[category])}")
                    if category != "valid" and results[category]:
                        print(f"  {category.capitalize()} placeholders:")
                        for placeholder in results[category]:
                            print(f"    - {placeholder}")
                return int(bool(results["invalid"] or results["missing"]))
            success = updater.update_paper_section(args.section, section_file)
            print(f"Updated {args.section}: {'✓' if success else '✗'}")
            return 0 if success else 1

        results = updater.update_all_paper_sections(args.paper)
        total_updated = sum(1 for success in results.values() if success)
        print(f"\nUpdated {total_updated}/{len(results)} sections successfully")
        return 0 if results and all(results.values()) else 1
    except (OSError, KeyError, ValueError, TypeError) as error:
        print(f"Key-number operation failed: {error}")
        return 1
