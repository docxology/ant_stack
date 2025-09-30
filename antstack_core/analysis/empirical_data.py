"""Empirical data structures for veridical reporting.

This module defines the core data structures used for empirical evidence
generation and case study documentation in scientific research.

Following .cursorrules principles:
- Clear data structure definitions
- Comprehensive type annotations
- Professional documentation with examples
- Immutable data structures using dataclasses
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any, Sequence


@dataclass(frozen=True)
class EmpiricalEvidence:
    """Container for empirical evidence with full provenance.

    This dataclass encapsulates all aspects of empirical evidence including:
    - Raw experimental measurements with uncertainty quantification
    - Statistical analysis results with confidence intervals
    - Validation checks and quality metrics
    - Complete provenance information for reproducibility

    Attributes:
        experiment_id: Unique identifier for the experiment
        timestamp: Unix timestamp when evidence was generated
        data_source: Description of data source (e.g., "computational_analysis")
        raw_measurements: Dict mapping parameter names to lists of measurements
        measurement_uncertainty: Dict mapping parameters to uncertainty values
        statistical_metrics: Dict of computed statistical measures (means, stds, etc.)
        confidence_intervals: Dict of confidence intervals for key metrics
        validation_checks: Dict of boolean validation checks (sufficient data, etc.)
        quality_metrics: Dict of quality scores (completeness, consistency, etc.)
        analysis_parameters: Dict of parameters used in analysis
        software_versions: Dict of software versions for reproducibility
        hardware_specifications: Dict of hardware specs for provenance

    Example:
        >>> evidence = EmpiricalEvidence(
        ...     experiment_id="exp_001",
        ...     timestamp=1640995200.0,
        ...     data_source="simulation",
        ...     raw_measurements={"energy": [1.0, 1.1, 0.9]},
        ...     measurement_uncertainty={"energy": 0.05},
        ...     statistical_metrics={"energy_mean": 1.0},
        ...     confidence_intervals={"energy_95ci": (0.9, 1.1)},
        ...     validation_checks={"sufficient_data": True},
        ...     quality_metrics={"completeness": 1.0},
        ...     analysis_parameters={"method": "bootstrap"},
        ...     software_versions={"python": "3.8+"},
        ...     hardware_specifications={"platform": "x86_64"}
        ... )
    """

    # Data identification
    experiment_id: str
    timestamp: float
    data_source: str

    # Raw measurements
    raw_measurements: Dict[str, List[float]]
    measurement_uncertainty: Dict[str, float]

    # Statistical analysis
    statistical_metrics: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Validation results
    validation_checks: Dict[str, bool]
    quality_metrics: Dict[str, float]

    # Provenance
    analysis_parameters: Dict[str, Any]
    software_versions: Dict[str, str]
    hardware_specifications: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the empirical evidence
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmpiricalEvidence:
        """Create from dictionary representation.

        Args:
            data: Dictionary representation of empirical evidence

        Returns:
            EmpiricalEvidence instance
        """
        return cls(**data)

    def get_parameter_summary(self, parameter: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a specific parameter.

        Args:
            parameter: Name of the parameter to summarize

        Returns:
            Dictionary with parameter summary or None if parameter not found
        """
        if parameter not in self.raw_measurements:
            return None

        return {
            "raw_values": self.raw_measurements[parameter],
            "uncertainty": self.measurement_uncertainty.get(parameter, 0.0),
            "mean": self.statistical_metrics.get(f"{parameter}_mean"),
            "std": self.statistical_metrics.get(f"{parameter}_std"),
            "confidence_interval": self.confidence_intervals.get(f"{parameter}_95ci"),
            "n_measurements": len(self.raw_measurements[parameter])
        }

    def get_quality_score(self) -> float:
        """Calculate overall quality score.

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not self.quality_metrics:
            return 0.0

        return sum(self.quality_metrics.values()) / len(self.quality_metrics)

    def is_valid(self) -> bool:
        """Check if evidence passes all validation checks.

        Returns:
            True if all validation checks pass
        """
        return all(self.validation_checks.values())


@dataclass(frozen=True)
class CaseStudy:
    """Comprehensive case study with empirical validation.

    This dataclass represents a complete case study including experimental
    setup, results, statistical analysis, and validation metrics.

    Attributes:
        case_id: Unique identifier for the case study
        title: Descriptive title
        description: Detailed description of the case study
        experimental_conditions: Dict of experimental conditions/parameters
        parameter_ranges: Dict mapping parameters to (min, max) tuples
        control_conditions: Dict of control/experimental conditions
        primary_results: Dict of main findings and results
        secondary_results: Dict of supporting results and analyses
        statistical_analysis: Dict of statistical test results
        reproducibility_metrics: Dict of reproducibility scores
        validation_results: Dict of validation outcomes
        cross_validation: Dict of cross-validation results
        methodology: Detailed methodology description
        limitations: List of study limitations
        implications: List of research implications

    Example:
        >>> case = CaseStudy(
        ...     case_id="cs_001",
        ...     title="Energy Scaling Analysis",
        ...     description="Analysis of energy scaling in computational systems",
        ...     experimental_conditions={"complexity": 1000},
        ...     parameter_ranges={"energy": (0.8, 1.2)},
        ...     control_conditions={"baseline": True},
        ...     primary_results={"efficiency": 0.95},
        ...     secondary_results={"robustness": 0.9},
        ...     statistical_analysis={"p_value": 0.001},
        ...     reproducibility_metrics={"data_availability": 1.0},
        ...     validation_results={"internal_consistency": True},
        ...     cross_validation={"score": 0.92},
        ...     methodology="Bootstrap analysis with 1000 samples",
        ...     limitations=["Limited parameter range"],
        ...     implications=["Improved energy efficiency"]
        ... )
    """

    case_id: str
    title: str
    description: str

    # Experimental setup
    experimental_conditions: Dict[str, Any]
    parameter_ranges: Dict[str, Tuple[float, float]]
    control_conditions: Dict[str, Any]

    # Results and analysis
    primary_results: Dict[str, Any]
    secondary_results: Dict[str, Any]
    statistical_analysis: Dict[str, Any]

    # Validation and reproducibility
    reproducibility_metrics: Dict[str, float]
    validation_results: Dict[str, bool]
    cross_validation: Dict[str, Any]

    # Documentation
    methodology: str
    limitations: List[str]
    implications: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary representation of the case study
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CaseStudy:
        """Create from dictionary representation.

        Args:
            data: Dictionary representation of case study

        Returns:
            CaseStudy instance
        """
        return cls(**data)

    def get_reproducibility_score(self) -> float:
        """Calculate overall reproducibility score.

        Returns:
            Reproducibility score between 0.0 and 1.0
        """
        if not self.reproducibility_metrics:
            return 0.0

        return sum(self.reproducibility_metrics.values()) / len(self.reproducibility_metrics)

    def is_validated(self) -> bool:
        """Check if case study passes validation.

        Returns:
            True if all validation checks pass
        """
        return all(self.validation_results.values())

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of the case study.

        Returns:
            Dictionary with key summary information
        """
        return {
            "case_id": self.case_id,
            "title": self.title,
            "description": self.description,
            "n_conditions": len(self.experimental_conditions),
            "n_results": len(self.primary_results),
            "n_limitations": len(self.limitations),
            "n_implications": len(self.implications),
            "reproducibility_score": self.get_reproducibility_score(),
            "is_validated": self.is_validated()
        }
