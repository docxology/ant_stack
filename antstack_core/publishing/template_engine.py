"""
Template Engine for Ant Stack Publications

Provides Jinja2-based templating system for consistent document formatting,
supporting dynamic content generation, conditional rendering, and custom macros.

Following .cursorrules specifications for:
- Professional document templates
- Consistent formatting standards
- Dynamic content generation
- LaTeX integration
"""

from __future__ import annotations

import jinja2
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import yaml
import json
import re
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TemplateConfig:
    """Configuration for template engine."""

    template_dir: Path
    cache_templates: bool = True
    auto_reload: bool = True
    custom_filters: Dict[str, Any] = None
    custom_globals: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.custom_filters is None:
            self.custom_filters = {}
        if self.custom_globals is None:
            self.custom_globals = {}


class TemplateEngine:
    """
    Advanced template engine for scientific publications.

    Features:
    - Jinja2 templating with custom filters and macros
    - LaTeX integration and macro support
    - Dynamic content generation
    - Template inheritance and composition
    - Scientific formatting utilities
    """

    def __init__(self, config: Optional[TemplateConfig] = None):
        """Initialize template engine with configuration."""
        if config is None:
            config = TemplateConfig(
                template_dir=Path(__file__).parent / "templates"
            )

        self.config = config
        self._jinja_env = None
        self._setup_jinja_environment()

    def _setup_jinja_environment(self):
        """Configure Jinja2 environment with custom settings."""
        # Create Jinja2 environment
        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.config.template_dir)),
            auto_reload=self.config.auto_reload,
            cache_size=0 if not self.config.cache_templates else 400,
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=['jinja2.ext.do', 'jinja2.ext.loopcontrols']
        )

        # Add custom filters
        self._add_custom_filters()

        # Add custom globals
        self._add_custom_globals()

        # Add user-defined filters and globals
        for name, func in self.config.custom_filters.items():
            self._jinja_env.filters[name] = func

        for name, value in self.config.custom_globals.items():
            self._jinja_env.globals[name] = value

    def _add_custom_filters(self):
        """Add custom Jinja2 filters for scientific formatting."""

        # LaTeX math formatting
        def latex_math(text: str) -> str:
            """Format text for LaTeX math mode."""
            if not text:
                return text
            return f"\\({text}\\)"

        # Unit formatting
        def format_unit(value: Union[float, int], unit: str,
                       precision: int = 3) -> str:
            """Format numerical value with units."""
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.{precision}g}"
            else:
                formatted_value = str(value)
            return f"{formatted_value} {unit}"

        # Scientific notation
        def scientific_notation(value: Union[float, int],
                              precision: int = 2) -> str:
            """Format number in scientific notation."""
            if isinstance(value, (int, float)) and value != 0:
                return f"{value:.{precision}e}"
            return str(value)

        # Uncertainty formatting
        def format_uncertainty(value: Union[float, int],
                             uncertainty: Union[float, int],
                             precision: int = 2) -> str:
            """Format value with uncertainty."""
            if not isinstance(value, (int, float)) or not isinstance(uncertainty, (int, float)):
                return f"{value} ± {uncertainty}"

            # Determine precision based on uncertainty
            uncertainty_str = f"{uncertainty:.{precision}g}"
            if '.' in uncertainty_str:
                decimal_places = len(uncertainty_str.split('.')[-1])
                value_str = f"{value:.{decimal_places}f}"
            else:
                value_str = f"{value:.{precision}g}"

            return f"{value_str} ± {uncertainty}"

        # Cross-reference formatting
        def format_ref(ref_type: str, ref_id: str) -> str:
            """Format cross-reference for LaTeX."""
            return f"\\cref{{{ref_type}:{ref_id}}}"

        # Citation formatting
        def format_citation(key: str, style: str = "numeric") -> str:
            """Format citation for LaTeX."""
            return f"\\cite{{{key}}}"

        # URL formatting with descriptive text
        def format_url(url: str, text: Optional[str] = None) -> str:
            """Format URL with descriptive text."""
            display_text = text or url
            return f"\\href{{{url}}}{{{display_text}}}"

        # Register filters
        filters = {
            'latex_math': latex_math,
            'format_unit': format_unit,
            'scientific_notation': scientific_notation,
            'format_uncertainty': format_uncertainty,
            'format_ref': format_ref,
            'format_citation': format_citation,
            'format_url': format_url,
        }

        for name, func in filters.items():
            self._jinja_env.filters[name] = func

    def _add_custom_globals(self):
        """Add custom global functions and variables."""

        # Current date/time
        import datetime
        self._jinja_env.globals['now'] = datetime.datetime.now
        self._jinja_env.globals['today'] = datetime.date.today

        # Version information
        try:
            import antstack_core
            self._jinja_env.globals['antstack_version'] = antstack_core.__version__
        except ImportError:
            self._jinja_env.globals['antstack_version'] = "unknown"

        # Utility functions
        def include_file(filepath: str, encoding: str = 'utf-8') -> str:
            """Include external file content."""
            try:
                path = Path(filepath)
                if path.is_absolute():
                    return path.read_text(encoding=encoding)
                else:
                    # Relative to template directory
                    template_path = self.config.template_dir / filepath
                    return template_path.read_text(encoding=encoding)
            except Exception as e:
                logger.warning(f"Failed to include file {filepath}: {str(e)}")
                return f"<!-- Failed to include {filepath}: {str(e)} -->"

        def yaml_load(content: str) -> Dict[str, Any]:
            """Load YAML content."""
            try:
                return yaml.safe_load(content)
            except Exception as e:
                logger.warning(f"Failed to parse YAML: {str(e)}")
                return {}

        def json_load(content: str) -> Dict[str, Any]:
            """Load JSON content."""
            try:
                return json.loads(content)
            except Exception as e:
                logger.warning(f"Failed to parse JSON: {str(e)}")
                return {}

        # Register globals
        globals_dict = {
            'include_file': include_file,
            'yaml_load': yaml_load,
            'json_load': json_load,
        }

        for name, func in globals_dict.items():
            self._jinja_env.globals[name] = func

    def render_template(self, template_name: str,
                       context: Dict[str, Any],
                       output_path: Optional[Path] = None) -> str:
        """
        Render template with given context.

        Args:
            template_name: Name of template file (without extension)
            context: Context variables for template
            output_path: Optional path to save rendered output

        Returns:
            Rendered template content
        """
        try:
            template = self._jinja_env.get_template(f"{template_name}.jinja")
            rendered_content = template.render(**context)

            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(rendered_content, encoding='utf-8')
                logger.info(f"Template rendered to: {output_path}")

            return rendered_content

        except jinja2.TemplateNotFound:
            raise FileNotFoundError(f"Template '{template_name}.jinja' not found")
        except jinja2.TemplateError as e:
            raise RuntimeError(f"Template rendering error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Template processing failed: {str(e)}")

    def render_markdown_template(self, template_name: str,
                               context: Dict[str, Any],
                               output_path: Optional[Path] = None) -> str:
        """
        Render markdown template with scientific formatting.

        Args:
            template_name: Name of template file
            context: Context variables for template
            output_path: Optional path to save rendered output

        Returns:
            Rendered markdown content
        """
        # Add scientific formatting helpers to context
        enhanced_context = context.copy()

        # Scientific formatting functions
        enhanced_context['format_scientific'] = lambda x: f"{x:.2e}" if isinstance(x, (int, float)) else str(x)
        enhanced_context['format_percent'] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x)
        enhanced_context['format_table'] = self._format_table

        return self.render_template(template_name, enhanced_context, output_path)

    def _format_table(self, data: List[Dict[str, Any]],
                     columns: Optional[List[str]] = None) -> str:
        """Format data as markdown table."""
        if not data:
            return ""

        # Determine columns
        if columns is None:
            columns = list(data[0].keys())

        # Create header
        header = "| " + " | ".join(columns) + " |"
        separator = "|" + "|".join([":---:" for _ in columns]) + "|"

        # Create rows
        rows = []
        for row in data:
            row_data = []
            for col in columns:
                value = row.get(col, "")
                row_data.append(str(value))
            rows.append("| " + " | ".join(row_data) + " |")

        return "\n".join([header, separator] + rows)

    def create_publication_template(self, paper_type: str,
                                  template_name: str,
                                  base_config: Dict[str, Any]) -> str:
        """
        Create publication template for specific paper type.

        Args:
            paper_type: Type of paper ("ant_stack", "complexity_energetics", etc.)
            template_name: Name for the new template
            base_config: Base configuration for the template

        Returns:
            Path to created template
        """
        template_dir = self.config.template_dir
        template_dir.mkdir(parents=True, exist_ok=True)

        # Template content based on paper type
        if paper_type == "ant_stack":
            template_content = self._create_ant_stack_template(base_config)
        elif paper_type == "complexity_energetics":
            template_content = self._create_complexity_template(base_config)
        else:
            template_content = self._create_generic_template(base_config)

        # Save template
        template_path = template_dir / f"{template_name}.jinja"
        template_path.write_text(template_content, encoding='utf-8')

        logger.info(f"Created template: {template_path}")
        return str(template_path)

    def _create_ant_stack_template(self, config: Dict[str, Any]) -> str:
        """Create template for Ant Stack framework papers."""
        return f"""# {{{{ title }}}}

**{{{{ authors }}}}

**Abstract:** {{{{ abstract }}}}

---

## Table of Contents

{{{{ toc }}}}

---

## Introduction

{{{{ introduction }}}}

## Background

### Ant Colony Optimization Principles
{{{{ background.ant_colony_principles }}}}

### Swarm Intelligence Mechanisms
{{{{ background.swarm_intelligence }}}}

### Collective Decision Making
{{{{ background.collective_decision_making }}}}

## AntBody: Locomotion and Sensing

### Mechanical Design
{{{{ antbody.mechanical_design }}}}

### Sensor Integration
{{{{ antbody.sensor_integration }}}}

### Actuation Systems
{{{{ antbody.actuation_systems }}}}

## AntBrain: Neural Processing

### Neural Architecture
{{{{ antbrain.neural_architecture }}}}

### Learning Mechanisms
{{{{ antbrain.learning_mechanisms }}}}

### Memory Systems
{{{{ antbrain.memory_systems }}}}

## AntMind: Decision Making

### Planning Algorithms
{{{{ antmind.planning_algorithms }}}}

### Goal-directed Behavior
{{{{ antmind.goal_directed_behavior }}}}

### Adaptive Strategies
{{{{ antmind.adaptive_strategies }}}}

## Applications

{{{{ applications|join('\\n\\n') }}}}

## Discussion

{{{{ discussion }}}}

## Conclusion

{{{{ conclusion }}}}

---

## References

{{{{ references }}}}

## Appendices

{{{{ appendices }}}}

---
*Generated with AntStack Core v{{{ antstack_version }}}*
"""

    def _create_complexity_template(self, config: Dict[str, Any]) -> str:
        """Create template for complexity energetics papers."""
        return f"""# {{{{ title }}}}

**{{{{ authors }}}}

**Abstract:** {{{{ abstract }}}}

---

## Introduction

{{{{ introduction }}}}

## Theoretical Framework

### Complexity Analysis
{{{{ theory.complexity_analysis }}}}

### Energy Modeling
{{{{ theory.energy_modeling }}}}

### Scaling Relationships
{{{{ theory.scaling_relationships }}}}

## Methods

### Experimental Setup
{{{{ methods.experimental_setup }}}}

### Analysis Pipeline
{{{{ methods.analysis_pipeline }}}}

### Validation Procedures
{{{{ methods.validation_procedures }}}}

## Results

### Complexity Metrics
{{{{ results.complexity_metrics }}}}

### Energy Consumption
{{{{ results.energy_consumption }}}}

### Scaling Analysis
{{{{ results.scaling_analysis }}}}

## Discussion

{{{{ discussion }}}}

## Conclusion

{{{{ conclusion }}}}

---

## References

{{{{ references }}}}

## Appendices

{{{{ appendices }}}}

---
*Generated with AntStack Core v{{{ antstack_version }}}*
"""

    def _create_generic_template(self, config: Dict[str, Any]) -> str:
        """Create generic template."""
        return f"""# {{{{ title }}}}

**{{{{ authors }}}}

**Abstract:** {{{{ abstract }}}}

---

{{{{ content }}}}

---

## References

{{{{ references }}}}

---
*Generated with AntStack Core v{{{ antstack_version }}}*
"""

    def list_available_templates(self) -> List[str]:
        """List all available templates."""
        if not self.config.template_dir.exists():
            return []

        templates = []
        for file_path in self.config.template_dir.glob("*.jinja"):
            templates.append(file_path.stem)

        return sorted(templates)

    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        Validate template syntax and structure.

        Args:
            template_name: Name of template to validate

        Returns:
            Validation results
        """
        try:
            template = self._jinja_env.get_template(f"{template_name}.jinja")

            # Test basic rendering
            test_context = {
                'title': 'Test Document',
                'authors': 'Test Author',
                'abstract': 'Test abstract',
                'content': 'Test content',
                'references': 'Test references'
            }

            rendered = template.render(**test_context)

            return {
                'valid': True,
                'template_name': template_name,
                'rendered_length': len(rendered),
                'jinja_errors': []
            }

        except jinja2.TemplateSyntaxError as e:
            return {
                'valid': False,
                'template_name': template_name,
                'error': f"Syntax error: {str(e)}",
                'line': e.lineno,
                'jinja_errors': [str(e)]
            }
        except Exception as e:
            return {
                'valid': False,
                'template_name': template_name,
                'error': f"Validation error: {str(e)}",
                'jinja_errors': [str(e)]
            }


# Convenience functions
def create_publication_template(paper_type: str, config: Dict[str, Any],
                              template_dir: Optional[Path] = None) -> TemplateEngine:
    """
    Create template engine with publication-specific configuration.

    Args:
        paper_type: Type of publication
        config: Configuration dictionary
        template_dir: Optional template directory

    Returns:
        Configured TemplateEngine instance
    """
    if template_dir is None:
        template_dir = Path(__file__).parent / "templates"

    template_config = TemplateConfig(
        template_dir=template_dir,
        cache_templates=True,
        auto_reload=False
    )

    engine = TemplateEngine(template_config)

    # Create default template if it doesn't exist
    template_name = f"{paper_type}_default"
    if template_name not in engine.list_available_templates():
        engine.create_publication_template(paper_type, template_name, config)

    return engine


def render_scientific_document(template_name: str, context: Dict[str, Any],
                             output_path: Optional[Path] = None) -> str:
    """
    Render scientific document with optimal settings.

    Args:
        template_name: Template to use
        context: Document context
        output_path: Optional output path

    Returns:
        Rendered document content
    """
    engine = TemplateEngine()
    return engine.render_markdown_template(template_name, context, output_path)
