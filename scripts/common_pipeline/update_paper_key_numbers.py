#!/usr/bin/env python3
"""
Update Paper Sections with Key Numbers Integration

Automatically replaces hardcoded numbers in paper sections with key_numbers.json
placeholders for dynamic content generation.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.key_numbers import get_key_numbers_manager


class PaperKeyNumbersUpdater:
    """Updates paper sections with key numbers placeholders."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize updater with project root."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = project_root
        self.key_manager = get_key_numbers_manager()

    def update_abstract(self, file_path: Path) -> bool:
        """Update Abstract.md with key numbers placeholders."""
        content = file_path.read_text(encoding='utf-8')

        # Replace scaling exponents
        content = re.sub(
            r'\$E \\propto K\^\{([^}]+)\}\$',
            r'$E \propto K^{{key_numbers.scaling.brain.exponent}}$',
            content, flags=re.MULTILINE | re.VERBOSE
        )
        content = re.sub(
            r'\$E \\propto H_p\^\{([^}]+)\}\$',
            r'$E \propto H_p^{{key_numbers.scaling.mind.exponent}}$',
            content, flags=re.MULTILINE | re.VERBOSE
        )

        # Replace planning horizon limit
        content = re.sub(
            r'H_p \\leq (\d+)',
            r'$H_p \leq$ {{key_numbers.mind_params.H_p_max}}',
            content, flags=re.MULTILINE | re.VERBOSE
        )

        # Replace sparsity value
        content = re.sub(
            r'\\rho \\approx ([0-9.]+)',
            r'$\rho \approx$ {{key_numbers.brain_params.rho}}',
            content, flags=re.MULTILINE | re.VERBOSE
        )

        file_path.write_text(content, encoding='utf-8')
        return True

    def update_energetics(self, file_path: Path) -> bool:
        """Update Energetics.md with key numbers placeholders."""
        content = file_path.read_text(encoding='utf-8')

        # Replace energy coefficients
        replacements = {
            r'e_\{\\(text\{FLOP\})\}': r'$e_{\text{FLOP}}$',
            r'e_\{\\(text\{SRAM\})\}': r'$e_{\text{SRAM}}$',
            r'e_\{\\(text\{DRAM\})\}': r'$e_{\text{DRAM}}$',
            r'E_\{\\(text\{spk\})\}': r'$E_{\text{spk}}$'
        }

        for pattern, replacement in replacements.items():
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        # Replace energy values with placeholders
        content = re.sub(
            r'(\d+(?:\.\d+)?)\s*mJ/decision',
            r'{{key_numbers.per_decision_energy.total_mj}} mJ/decision',
            content, flags=re.MULTILINE
        )

        file_path.write_text(content, encoding='utf-8')
        return True

    def update_results(self, file_path: Path) -> bool:
        """Update Results.md with key numbers placeholders."""
        content = file_path.read_text(encoding='utf-8')

        # Replace scaling relationships
        content = re.sub(
            r'E \\propto J\^\{([^}]+)\}',
            r'$E \propto J^{{key_numbers.scaling.body.exponent}}$',
            content, flags=re.MULTILINE
        )
        content = re.sub(
            r'E \\propto K\^\{([^}]+)\}',
            r'$E \propto K^{{key_numbers.scaling.brain.exponent}}$',
            content, flags=re.MULTILINE
        )
        content = re.sub(
            r'E \\propto H_p\^\{([^}]+)\}',
            r'$E \propto H_p^{{key_numbers.scaling.mind.exponent}}$',
            content, flags=re.MULTILINE
        )

        # Replace R-squared values
        content = re.sub(
            r'R\^2 = ([0-9.]+)',
            r'$R^2 =$ {{key_numbers.scaling.body.r_squared}}',
            content, flags=re.MULTILINE
        )

        # Replace energy ranges
        content = re.sub(
            r'([0-9.]+) \pm ([0-9.]+) mJ',
            r'{{key_numbers.per_decision_energy.body_mj}} mJ',
            content, flags=re.MULTILINE
        )

        file_path.write_text(content, encoding='utf-8')
        return True

    def update_background(self, file_path: Path) -> bool:
        """Update Background.md with key numbers placeholders."""
        content = file_path.read_text(encoding='utf-8')

        # Replace complexity notations
        content = re.sub(
            r'\\mathcal\{O\}\(([^)]+)\)',
            r'$\mathcal{O}(\1)$',
            content, flags=re.MULTILINE
        )

        # Replace scaling parameters
        content = re.sub(
            r'\\rho \\leq ([0-9.]+)',
            r'$\rho \leq$ {{key_numbers.brain_params.rho}}',
            content, flags=re.MULTILINE
        )

        # Replace planning horizon
        content = re.sub(
            r'H_p \\leq (\d+)',
            r'$H_p \leq$ {{key_numbers.mind_params.H_p_max}}',
            content, flags=re.MULTILINE
        )

        file_path.write_text(content, encoding='utf-8')
        return True

    def update_complexity(self, file_path: Path) -> bool:
        """Update Complexity.md with key numbers placeholders."""
        content = file_path.read_text(encoding='utf-8')

        # Replace FLOPs values
        content = re.sub(
            r'(\d+(?:\.\d+)?) FLOPs',
            r'{{key_numbers.computational_load.body_flops}} FLOPs',
            content, flags=re.MULTILINE
        )

        # Replace complexity expressions
        content = re.sub(
            r'\\mathcal\{O\}\(([^)]+)\)',
            r'$\mathcal{O}(\1)$',
            content, flags=re.MULTILINE
        )

        # Replace parameter ranges
        content = re.sub(
            r'(\d+)-(\d+) joints',
            r'{{key_numbers.body_params.J_min}}-{{key_numbers.body_params.J_max}} joints',
            content, flags=re.MULTILINE
        )

        file_path.write_text(content, encoding='utf-8')
        return True

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
        """Validate that key numbers placeholders in a file can be resolved."""
        content = file_path.read_text(encoding='utf-8')

        # Find all key_numbers placeholders
        placeholders = re.findall(r'\{\{key_numbers\.([^}]+)\}\}', content)

        validation_results = {
            'valid': [],
            'invalid': [],
            'missing': []
        }

        for placeholder in placeholders:
            try:
                # Try to resolve the placeholder
                self.key_manager.loader.get_formatted_value(f"per_decision_energy.{placeholder}", ".3f")
                validation_results['valid'].append(placeholder)
            except (KeyError, ValueError):
                validation_results['invalid'].append(placeholder)

        return validation_results


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Update paper sections with key numbers integration")
    parser.add_argument("--paper", default="complexity_energetics",
                       help="Paper name to update (default: complexity_energetics)")
    parser.add_argument("--section", help="Specific section to update")
    parser.add_argument("--validate", action="store_true",
                       help="Validate existing placeholders instead of updating")

    args = parser.parse_args()

    updater = PaperKeyNumbersUpdater()

    if args.validate:
        # Validation mode
        if args.section:
            paper_dir = Path("papers") / args.paper
            section_file = paper_dir / f"{args.section.capitalize()}.md"
            if section_file.exists():
                results = updater.validate_key_numbers_placeholders(section_file)
                print(f"Validation results for {args.section}:")
                print(f"  Valid: {len(results['valid'])}")
                print(f"  Invalid: {len(results['invalid'])}")
                if results['invalid']:
                    print("  Invalid placeholders:")
                    for invalid in results['invalid']:
                        print(f"    - {invalid}")
            else:
                print(f"Section file not found: {section_file}")
        else:
            print("Specify --section for validation mode")
    else:
        # Update mode
        if args.section:
            paper_dir = Path("papers") / args.paper
            section_file = paper_dir / f"{args.section.capitalize()}.md"
            if section_file.exists():
                success = updater.update_paper_section(args.section, section_file)
                print(f"Updated {args.section}: {'✓' if success else '✗'}")
            else:
                print(f"Section file not found: {section_file}")
        else:
            # Update all sections
            results = updater.update_all_paper_sections(args.paper)
            total_updated = sum(1 for success in results.values() if success)
            print(f"\nUpdated {total_updated}/{len(results)} sections successfully")


if __name__ == "__main__":
    main()
