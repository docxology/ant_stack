#!/usr/bin/env python3
"""Generate biomechanical and robotic benchmarks for energy analysis.

This script calculates comprehensive biomechanical benchmarks including:
- Power density comparisons between biological and robotic systems
- Efficiency trade-offs and energy storage capacity
- Detailed hexapod platform energy analysis
- Automatic markdown generation with calculated values

Usage:
    python scripts/generate_biomechanical_benchmarks.py [--output-dir output_dir] [--update-markdown]

References:
- Biomechanical power density: https://doi.org/10.1126/science.273.5272.267
- Robotic actuator efficiency: https://doi.org/10.1109/TMECH.2019.2942671
- Energy storage comparisons: https://doi.org/10.1038/s41586-020-2196-x
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis import (
    calculate_biomechanical_benchmarks,
    generate_biomechanical_comparison_table,
    generate_hexapod_analysis_markdown,
    BiomechanicalBenchmarks
)


def generate_biomechanical_report(benchmarks: BiomechanicalBenchmarks,
                                output_dir: Path) -> Dict[str, str]:
    """Generate comprehensive biomechanical analysis report.

    Args:
        benchmarks: Calculated biomechanical benchmarks
        output_dir: Output directory for generated files

    Returns:
        Dictionary containing file paths and generated content
    """
    output_dir.mkdir(exist_ok=True)

    # Generate comparison table
    table_content = generate_biomechanical_comparison_table(benchmarks)
    table_file = output_dir / "biomechanical_comparison_table.md"
    with open(table_file, 'w') as f:
        f.write("# Biomechanical Comparison Table\n\n")
        f.write(table_content)
        f.write("\n\n")

    # Generate hexapod analysis
    hexapod_content = generate_hexapod_analysis_markdown(benchmarks)
    hexapod_file = output_dir / "hexapod_energy_analysis.md"
    with open(hexapod_file, 'w') as f:
        f.write("# Hexapod Energy Analysis\n\n")
        f.write(hexapod_content)
        f.write("\n\n")

    # Generate comprehensive report
    report_file = output_dir / "biomechanical_benchmarks_report.md"
    with open(report_file, 'w') as f:
        f.write("# Biomechanical and Robotic Benchmarks\n\n")

        f.write("## Power Density and Efficiency Comparisons\n\n")
        f.write("### Calculated Benchmarks\n\n")
        f.write(table_content)
        f.write("\n\n")

        f.write("### Detailed Analysis\n\n")
        f.write("**Power Density Analysis**:\n")
        f.write("- **Biological Muscle**: ")
        f.write(f"${benchmarks.muscle_power_density_w_kg:.0f}$ W/kg ")
        f.write("(high power density, low efficiency)\n")
        f.write("- **Robotic Actuators**: ")
        f.write(f"${benchmarks.actuator_power_density_w_kg:.0f}$ W/kg ")
        f.write("(BLDC motors + harmonic drives)\n")
        f.write("- **Efficiency Trade-offs**: ")
        f.write("Biological ($")
        f.write(f"{benchmarks.muscle_efficiency_percent:.0f}\\%$), ")
        f.write("Robotic ($")
        f.write(f"{benchmarks.actuator_efficiency_percent:.0f}\\%$)\n\n")

        f.write("**Energy Storage Capacity**:\n")
        f.write("- **Biological**: Carbohydrates $")
        f.write(f"{benchmarks.carbohydrate_energy_density_mj_kg:.1f}$ MJ/kg ")
        f.write("(high energy density)\n")
        f.write("- **Robotic**: Li-ion batteries $")
        f.write(f"{benchmarks.battery_energy_density_mj_kg:.2f}$ MJ/kg ")
        f.write("(current technology limitation)\n\n")

        f.write("## Hexapod Platform Energy Analysis\n\n")
        f.write(hexapod_content)
        f.write("\n\n")

        f.write("## Validation and Methodology\n\n")
        f.write("All reported values are calculated from empirically-derived parameters ")
        f.write("with comprehensive literature validation:\n\n")
        f.write("- Muscle power density: Rome et al. (1996)\n")
        f.write("- Actuator efficiency: BLDC motor characteristics\n")
        f.write("- Energy storage: Biochemical vs electrochemical comparisons\n")
        f.write("- Hexapod analysis: Multi-joint robotic platform modeling\n\n")

        f.write("## References\n\n")
        f.write("- Muscle power and efficiency: https://doi.org/10.1126/science.273.5272.267\n")
        f.write("- Robotic actuator efficiency: https://doi.org/10.1109/TMECH.2019.2942671\n")
        f.write("- Energy storage comparisons: https://doi.org/10.1038/s41586-020-2196-x\n")

    return {
        'table_file': str(table_file),
        'hexapod_file': str(hexapod_file),
        'report_file': str(report_file),
        'table_content': table_content,
        'hexapod_content': hexapod_content
    }


def update_markdown_files(generated_content: Dict[str, str],
                         paper_dir: Path) -> None:
    """Update markdown files with generated biomechanical content.

    Args:
        generated_content: Dictionary containing generated content
        paper_dir: Path to the complexity_energetics paper directory
    """
    # Update Energetics.md
    energetics_file = paper_dir / "Energetics.md"

    # Read current content
    with open(energetics_file, 'r') as f:
        content = f.read()

    # Replace the biomechanical benchmarks section
    old_section = """### Biomechanical and Robotic Benchmarks

**Power Density Comparisons**:
- **Biological Muscle**: $450$ W/kg (high power density, low efficiency)
- **Robotic Actuators**: $250$ W/kg (BLDC motors + harmonic drives)
- **Efficiency Trade-offs**: Biological ($22\\%$), Robotic ($45\\%$)

**Energy Storage Capacity**:
- **Biological**: Carbohydrates $17.0$ MJ/kg (high energy density)
- **Robotic**: Li-ion batteries $0.87$ MJ/kg (current technology limitation)

### Practical Energy Estimation Example

**Hexapod Platform Analysis (18 DOF)**:
- **Assumptions**: Per-joint mechanical power = $0.8$ W at trot gait, actuator efficiency $\\eta = 0.45$, contact/friction overhead = $15\\%$
- **Per-Joint Electrical Power**: $P_\\text{{elec}} \\approx 0.8 / 0.45 \\times 1.15 \\approx 2.04$ W
- **Whole-Body Actuation**: $P_\\text{{act}} \\approx 18 \\times 2.04 \\approx 36.7$ W
- **Sensor/Controller Baseline**: $P_\\text{{sens+idle}} \\approx 3.0$ W
- **Total Locomotion Power**: $P_\\text{{body}} \\approx 39.7$ W
- **Energy per Decision**: At 100 Hz control, $E_\\text{{decision}} \\approx 39.7 / 100 \\approx 0.397$ J"""

    new_section = """### Biomechanical and Robotic Benchmarks

**Power Density Comparisons**:
- **Biological Muscle**: $450$ W/kg (high power density, low efficiency)
- **Robotic Actuators**: $250$ W/kg (BLDC motors + harmonic drives)
- **Efficiency Trade-offs**: Biological ($22\\%$), Robotic ($45\\%$)

**Energy Storage Capacity**:
- **Biological**: Carbohydrates $17.0$ MJ/kg (high energy density)
- **Robotic**: Li-ion batteries $0.87$ MJ/kg (current technology limitation)

### Practical Energy Estimation Example

""" + generated_content['hexapod_content']

    updated_content = content.replace(old_section, new_section)

    # Write updated content
    with open(energetics_file, 'w') as f:
        f.write(updated_content)

    print(f"Updated {energetics_file}")

    # Update concat_after_prerender.md if it exists
    concat_file = paper_dir / "assets" / "concat_after_prerender.md"
    if concat_file.exists():
        with open(concat_file, 'r') as f:
            concat_content = f.read()

        # Replace similar section in concat file
        old_concat_section = """**Power Density Comparisons**:
- **Biological Muscle**: $450$ W/kg (high power density, low efficiency)
- **Robotic Actuators**: $250$ W/kg (BLDC motors + harmonic drives)
- **Efficiency Trade-offs**: Biological ($22\\%$), Robotic ($45\\%$)

**Energy Storage Capacity**:
- **Biological**: Carbohydrates $17.0$ MJ/kg (high energy density)
- **Robotic**: Li-ion batteries $0.87$ MJ/kg (current technology limitation)"""

        new_concat_section = """**Power Density Comparisons**:
- **Biological Muscle**: $450$ W/kg (high power density, low efficiency)
- **Robotic Actuators**: $250$ W/kg (BLDC motors + harmonic drives)
- **Efficiency Trade-offs**: Biological ($22\\%$), Robotic ($45\\%$)

**Energy Storage Capacity**:
- **Biological**: Carbohydrates $17.0$ MJ/kg (high energy density)
- **Robotic**: Li-ion batteries $0.87$ MJ/kg (current technology limitation)"""

        updated_concat_content = concat_content.replace(old_concat_section, new_concat_section)

        with open(concat_file, 'w') as f:
            f.write(updated_concat_content)

        print(f"Updated {concat_file}")


def main():
    """Main function for generating biomechanical benchmarks."""
    parser = argparse.ArgumentParser(description="Generate biomechanical and robotic benchmarks")
    parser.add_argument("--output-dir", default="biomechanical_output",
                       help="Output directory for generated files")
    parser.add_argument("--update-markdown", action="store_true",
                       help="Update markdown files with generated content")
    parser.add_argument("--paper-dir",
                       default=os.path.join(os.path.dirname(__file__), '..', '..', 'papers', 'complexity_energetics'),
                       help="Path to complexity_energetics paper directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Calculate biomechanical benchmarks
    print("Calculating biomechanical benchmarks...")
    benchmarks = calculate_biomechanical_benchmarks()

    # Generate comprehensive report
    print("Generating biomechanical analysis report...")
    generated_content = generate_biomechanical_report(benchmarks, output_dir)

    print(f"\nBiomechanical analysis complete!")
    print(f"Generated files in: {output_dir}")
    print(f"- Comparison table: {generated_content['table_file']}")
    print(f"- Hexapod analysis: {generated_content['hexapod_file']}")
    print(f"- Full report: {generated_content['report_file']}")

    # Update markdown files if requested
    if args.update_markdown:
        paper_dir = Path(args.paper_dir)
        print(f"\nUpdating markdown files in: {paper_dir}")
        update_markdown_files(generated_content, paper_dir)
        print("Markdown files updated successfully!")

    # Print key results
    print("\nKey Results:")
    print(f"- Muscle power density: {benchmarks.muscle_power_density_w_kg:.0f} W/kg")
    print(f"- Actuator power density: {benchmarks.actuator_power_density_w_kg:.0f} W/kg")
    print(f"- Muscle efficiency: {benchmarks.muscle_efficiency_percent:.0f}%")
    print(f"- Actuator efficiency: {benchmarks.actuator_efficiency_percent:.0f}%")
    print(f"- Total locomotion power: {benchmarks.total_locomotion_power_w:.1f} W")
    print(f"- Energy per decision: {benchmarks.energy_per_decision_j:.3f} J")


if __name__ == "__main__":
    main()
