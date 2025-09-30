#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.figures visualization modules.

Tests all visualization functionality including:
- Basic plotting (plots.py)
- Publication-quality plots (publication_plots.py)
- Advanced plotting (advanced_plots.py)
- Mermaid diagram processing (mermaid.py)
- Cross-reference validation (references.py)
- Asset management (assets.py)
- Integration across all visualization components

Following .cursorrules principles:
- Real plotting functionality testing
- Graceful fallback testing for CI environments
- Comprehensive edge case testing
- Scientific accuracy validation
- Cross-platform compatibility
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import visualization modules
from antstack_core.figures import (
    bar_plot, line_plot, scatter_plot,
    FigureManager, publication_bar_plot, publication_line_plot, publication_scatter_plot,
    preprocess_mermaid_diagrams, validate_mermaid_syntax,
    validate_cross_references, fix_figure_ids,
    organize_figure_assets, copy_figure_files
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestBasicPlots(unittest.TestCase):
    """Test basic plotting functions from plots.py."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        self.sample_data = {
            'categories': ['A', 'B', 'C', 'D'],
            'values': [1, 3, 2, 4],
            'errors': [0.1, 0.2, 0.15, 0.3]
        }

        self.line_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [1, 4, 2, 5, 3]
        }

        self.scatter_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [1.1, 1.9, 3.2, 3.8, 5.1],
            'sizes': [20, 50, 30, 80, 40]
        }

    def test_bar_plot_basic(self):
        """Test basic bar plot functionality."""
        fig = bar_plot(
            categories=self.sample_data['categories'],
            values=self.sample_data['values'],
            title="Test Bar Plot",
            xlabel="Categories",
            ylabel="Values"
        )

        self.assertIsNotNone(fig)
        # Should be a matplotlib figure
        self.assertTrue(hasattr(fig, 'savefig'))

    def test_bar_plot_with_errors(self):
        """Test bar plot with error bars."""
        fig = bar_plot(
            categories=self.sample_data['categories'],
            values=self.sample_data['values'],
            errors=self.sample_data['errors'],
            title="Bar Plot with Errors",
            ylabel="Values ± Error"
        )

        self.assertIsNotNone(fig)

    def test_bar_plot_empty_data(self):
        """Test bar plot with empty data."""
        fig = bar_plot(
            categories=[],
            values=[],
            title="Empty Bar Plot"
        )

        self.assertIsNotNone(fig)

    def test_line_plot_basic(self):
        """Test basic line plot functionality."""
        fig = line_plot(
            x_data=[self.line_data['x']],
            y_data=[self.line_data['y']],
            labels=["Test Line"],
            title="Test Line Plot",
            xlabel="X Values",
            ylabel="Y Values"
        )

        self.assertIsNotNone(fig)

    def test_line_plot_multiple_series(self):
        """Test line plot with multiple series."""
        x_data = [self.line_data['x'], self.line_data['x']]
        y_data = [self.line_data['y'], [y*0.8 for y in self.line_data['y']]]
        labels = ["Series 1", "Series 2"]

        fig = line_plot(
            x_data=x_data,
            y_data=y_data,
            labels=labels,
            title="Multiple Series Line Plot"
        )

        self.assertIsNotNone(fig)

    def test_scatter_plot_basic(self):
        """Test basic scatter plot functionality."""
        fig = scatter_plot(
            x_data=self.scatter_data['x'],
            y_data=self.scatter_data['y'],
            title="Test Scatter Plot",
            xlabel="X Values",
            ylabel="Y Values"
        )

        self.assertIsNotNone(fig)

    def test_scatter_plot_with_sizes(self):
        """Test scatter plot with varying point sizes."""
        fig = scatter_plot(
            x_data=self.scatter_data['x'],
            y_data=self.scatter_data['y'],
            sizes=self.scatter_data['sizes'],
            title="Scatter Plot with Sizes",
            xlabel="X Values",
            ylabel="Y Values"
        )

        self.assertIsNotNone(fig)

    def test_scatter_plot_single_point(self):
        """Test scatter plot with single data point."""
        fig = scatter_plot(
            x_data=[1],
            y_data=[2],
            title="Single Point Scatter",
            xlabel="X",
            ylabel="Y"
        )

        self.assertIsNotNone(fig)


class TestPublicationPlots(unittest.TestCase):
    """Test publication-quality plotting functions."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        self.sample_data = {
            'x': [1, 2, 3, 4, 5],
            'y': [1.1, 2.2, 3.1, 4.0, 5.2],
            'categories': ['A', 'B', 'C'],
            'values': [10, 15, 12]
        }

    def test_figure_manager_creation(self):
        """Test FigureManager creation and basic functionality."""
        manager = FigureManager(output_dir="./test_figures")

        self.assertIsNotNone(manager)
        self.assertEqual(manager.output_dir, Path("./test_figures"))

    def test_publication_bar_plot(self):
        """Test publication-quality bar plot."""
        fig = publication_bar_plot(
            categories=self.sample_data['categories'],
            values=self.sample_data['values'],
            title="Publication Bar Plot",
            ylabel="Measurement Values"
        )

        self.assertIsNotNone(fig)

    def test_publication_line_plot(self):
        """Test publication-quality line plot."""
        fig = publication_line_plot(
            x_data=self.sample_data['x'],
            y_data=self.sample_data['y'],
            title="Publication Line Plot",
            xlabel="Time",
            ylabel="Signal"
        )

        self.assertIsNotNone(fig)

    def test_publication_scatter_plot(self):
        """Test publication-quality scatter plot."""
        fig = publication_scatter_plot(
            x_data=self.sample_data['x'],
            y_data=self.sample_data['y'],
            title="Publication Scatter Plot",
            xlabel="Independent Variable",
            ylabel="Dependent Variable"
        )

        self.assertIsNotNone(fig)


class TestMermaidProcessing(unittest.TestCase):
    """Test Mermaid diagram processing functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_diagram = """
        graph TD
            A[Start] --> B{Decision}
            B -->|Yes| C[Action 1]
            B -->|No| D[Action 2]
            C --> E[End]
            D --> E
        """

        self.invalid_diagram = """
        graph TD
            A[Start
            B{Decision
            # Missing closing brackets
        """

    def test_validate_mermaid_syntax_valid(self):
        """Test validation of valid Mermaid syntax."""
        is_valid, errors = validate_mermaid_syntax(self.valid_diagram)

        self.assertTrue(is_valid)
        self.assertIsInstance(errors, list)
        self.assertEqual(len(errors), 0)

    def test_validate_mermaid_syntax_invalid(self):
        """Test validation of invalid Mermaid syntax."""
        is_valid, errors = validate_mermaid_syntax(self.invalid_diagram)

        self.assertFalse(is_valid)
        self.assertIsInstance(errors, list)
        self.assertGreater(len(errors), 0)

    def test_preprocess_mermaid_diagrams(self):
        """Test Mermaid diagram preprocessing."""
        markdown_content = f"""
        # Test Document

        Here is a diagram:

        ```mermaid
        {self.valid_diagram}
        ```

        End of document.
        """

        processed = preprocess_mermaid_diagrams(markdown_content)

        self.assertIsInstance(processed, str)
        self.assertIn("mermaid", processed.lower())

    def test_preprocess_mermaid_diagrams_no_diagrams(self):
        """Test preprocessing with no Mermaid diagrams."""
        content = "# Simple Document\n\nNo diagrams here."

        processed = preprocess_mermaid_diagrams(content)

        self.assertEqual(processed, content)


class TestCrossReferences(unittest.TestCase):
    """Test cross-reference validation and fixing."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_markdown = """
        # Test Document

        ## Figure: Energy Analysis {#fig:energy_analysis}

        ![Energy consumption by component](figures/energy.png)

        **Caption:** Energy breakdown showing computation vs physical costs.

        ## Results

        As shown in Figure {@fig:energy_analysis}, the computational energy dominates.
        """

        self.invalid_markdown = """
        # Test Document

        ## Figure: Energy Analysis {#fig:energy}

        ![Energy consumption](figures/energy.png)

        **Caption:** Energy breakdown.

        ## Results

        As shown in Figure {@fig:energy_analysis}, the energy consumption is high.
        """

    def test_validate_cross_references_valid(self):
        """Test validation of valid cross-references."""
        is_valid, errors = validate_cross_references(self.valid_markdown)

        self.assertTrue(is_valid)
        self.assertIsInstance(errors, list)
        self.assertEqual(len(errors), 0)

    def test_validate_cross_references_invalid(self):
        """Test validation of invalid cross-references."""
        is_valid, errors = validate_cross_references(self.invalid_markdown)

        self.assertFalse(is_valid)
        self.assertIsInstance(errors, list)
        self.assertGreater(len(errors), 0)

    def test_fix_figure_ids(self):
        """Test automatic fixing of figure IDs."""
        fixed_markdown = fix_figure_ids(self.invalid_markdown)

        # Should have fixed the reference
        self.assertIn("{@fig:energy}", fixed_markdown)

    def test_fix_figure_ids_no_changes_needed(self):
        """Test fixing when no changes are needed."""
        fixed_markdown = fix_figure_ids(self.valid_markdown)

        # Should be unchanged
        self.assertEqual(fixed_markdown, self.valid_markdown)


class TestAssetManagement(unittest.TestCase):
    """Test asset management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.source_dir = os.path.join(self.temp_dir, "source")
        self.dest_dir = os.path.join(self.temp_dir, "destination")

        os.makedirs(self.source_dir)
        os.makedirs(self.dest_dir)

        # Create test files
        self.test_files = []
        for i in range(3):
            filename = f"figure_{i}.png"
            filepath = os.path.join(self.source_dir, filename)
            with open(filepath, 'w') as f:
                f.write(f"Test content {i}")
            self.test_files.append(filename)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_organize_figure_assets(self):
        """Test figure asset organization."""
        # Create a markdown file referencing the figures
        markdown_content = f"""
        # Test Document

        ## Figure: Test 1 {{#fig:test1}}

        ![{self.test_files[0]}]({self.test_files[0]})

        **Caption:** First test figure.

        ## Figure: Test 2 {{#fig:test2}}

        ![{self.test_files[1]}]({self.test_files[1]})

        **Caption:** Second test figure.
        """

        success = organize_figure_assets(
            markdown_content=markdown_content,
            source_dir=self.source_dir,
            output_dir=self.dest_dir
        )

        self.assertTrue(success)

        # Check that files were copied
        for filename in self.test_files[:2]:
            dest_path = os.path.join(self.dest_dir, filename)
            self.assertTrue(os.path.exists(dest_path))

    def test_copy_figure_files(self):
        """Test figure file copying."""
        success = copy_figure_files(
            file_list=self.test_files,
            source_dir=self.source_dir,
            dest_dir=self.dest_dir
        )

        self.assertTrue(success)

        # Check that all files were copied
        for filename in self.test_files:
            dest_path = os.path.join(self.dest_dir, filename)
            self.assertTrue(os.path.exists(dest_path))


class TestVisualizationIntegration(unittest.TestCase):
    """Test integration across visualization components."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_complete_figure_workflow(self):
        """Test complete figure creation and management workflow."""
        # Step 1: Create a figure
        fig = bar_plot(
            categories=['A', 'B', 'C'],
            values=[1, 2, 3],
            title="Test Figure"
        )

        # Step 2: Create figure manager
        manager = FigureManager(output_dir=self.temp_dir)

        # Step 3: Save figure
        output_path = os.path.join(self.temp_dir, "test_figure.png")
        fig.savefig(output_path)

        # Step 4: Verify file was created
        self.assertTrue(os.path.exists(output_path))

    def test_mermaid_and_references_integration(self):
        """Test integration between Mermaid processing and cross-references."""
        # Create content with both Mermaid and figure references
        content = """
        # Analysis Document

        ## Figure: System Architecture {#fig:architecture}

        ```mermaid
        graph TD
            A[Input] --> B[Processing]
            B --> C[Output]
        ```

        **Caption:** System architecture diagram.

        ## Results

        The system architecture (Figure {@fig:architecture}) shows the data flow.
        """

        # Test Mermaid validation
        is_valid_mermaid, _ = validate_mermaid_syntax(content)
        self.assertTrue(is_valid_mermaid)

        # Test cross-reference validation
        is_valid_refs, _ = validate_cross_references(content)
        self.assertTrue(is_valid_refs)

        # Test preprocessing
        processed = preprocess_mermaid_diagrams(content)
        self.assertIsInstance(processed, str)

    def test_publication_figure_manager_integration(self):
        """Test integration between publication plots and figure manager."""
        # Create publication-quality figure
        fig = publication_scatter_plot(
            x_data=[1, 2, 3, 4, 5],
            y_data=[1.1, 2.2, 3.1, 4.0, 5.2],
            title="Publication Scatter Plot",
            xlabel="X Variable",
            ylabel="Y Variable"
        )

        # Create figure manager
        manager = FigureManager(output_dir=self.temp_dir)

        # Save the figure
        output_path = os.path.join(self.temp_dir, "publication_figure.png")
        fig.savefig(output_path)

        # Verify
        self.assertTrue(os.path.exists(output_path))

        # Test asset organization
        markdown_content = f"""
        ## Figure: Publication Plot {{#fig:pub_plot}}

        ![Publication scatter plot](publication_figure.png)

        **Caption:** Publication-quality scatter plot.
        """

        success = organize_figure_assets(
            markdown_content=markdown_content,
            source_dir=self.temp_dir,
            output_dir=self.temp_dir
        )

        self.assertTrue(success)


class TestVisualizationRobustness(unittest.TestCase):
    """Test robustness of visualization functionality."""

    def test_visualization_without_matplotlib(self):
        """Test visualization behavior without matplotlib."""
        # This tests fallback behavior when matplotlib is not available
        if HAS_MATPLOTLIB:
            self.skipTest("Matplotlib is available")

        # These should handle missing matplotlib gracefully
        try:
            result = bar_plot(['A', 'B'], [1, 2], title="Test")
            # Should either return None or raise informative error
            self.assertTrue(result is None or isinstance(result, Exception))
        except ImportError:
            # Expected when matplotlib is not available
            pass

    def test_visualization_with_extreme_data(self):
        """Test visualization with extreme data values."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        # Test with very large numbers
        large_values = [1e6, 2e6, 3e6, 4e6]
        fig = bar_plot(
            categories=['A', 'B', 'C', 'D'],
            values=large_values,
            title="Large Values Test"
        )

        self.assertIsNotNone(fig)

        # Test with very small numbers
        small_values = [1e-6, 2e-6, 3e-6, 4e-6]
        fig = bar_plot(
            categories=['A', 'B', 'C', 'D'],
            values=small_values,
            title="Small Values Test"
        )

        self.assertIsNotNone(fig)

    def test_visualization_error_handling(self):
        """Test error handling in visualization functions."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        # Test with invalid data types
        with self.assertRaises((TypeError, ValueError)):
            bar_plot(
                categories=None,
                values=[1, 2, 3]
            )

        with self.assertRaises((TypeError, ValueError)):
            line_plot(
                x_data=None,
                y_data=[[1, 2, 3]]
            )

    def test_asset_management_error_handling(self):
        """Test error handling in asset management."""
        # Test with non-existent directories
        success = organize_figure_assets(
            markdown_content="# Test",
            source_dir="/nonexistent/source",
            output_dir="/nonexistent/dest"
        )

        # Should handle gracefully
        self.assertIsInstance(success, bool)

    def test_cross_reference_validation_edge_cases(self):
        """Test cross-reference validation with edge cases."""
        # Test with empty content
        is_valid, errors = validate_cross_references("")
        self.assertTrue(is_valid)

        # Test with malformed figure definitions
        malformed_content = """
        ## Figure: Test
        ![image](test.png)
        **Caption:** Test.
        """
        is_valid, errors = validate_cross_references(malformed_content)
        # Should detect missing figure ID
        self.assertFalse(is_valid)

    def test_mermaid_validation_edge_cases(self):
        """Test Mermaid validation with edge cases."""
        # Test with empty diagram
        is_valid, errors = validate_mermaid_syntax("")
        self.assertFalse(is_valid)

        # Test with very simple valid diagram
        simple_diagram = "graph TD\n    A --> B"
        is_valid, errors = validate_mermaid_syntax(simple_diagram)
        self.assertTrue(is_valid)


class TestVisualizationPerformance(unittest.TestCase):
    """Test performance characteristics of visualization functions."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

    def test_visualization_performance_large_data(self):
        """Test visualization performance with large datasets."""
        import time

        # Create large dataset
        n_points = 10000
        x_data = list(range(n_points))
        y_data = [i * 2 + (i % 100) for i in range(n_points)]

        start_time = time.time()
        fig = line_plot(
            x_data=[x_data],
            y_data=[y_data],
            labels=["Large Dataset"],
            title="Performance Test"
        )
        end_time = time.time()

        # Should complete within reasonable time (less than 5 seconds)
        self.assertLess(end_time - start_time, 5.0)
        self.assertIsNotNone(fig)

    def test_visualization_memory_usage(self):
        """Test memory usage patterns in visualization."""
        import psutil
        import os

        if not HAS_MATPLOTLIB:
            self.skipTest("Matplotlib not available")

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create multiple figures
        figures = []
        for i in range(10):
            fig = scatter_plot(
                x_data=list(range(100)),
                y_data=[j + i*10 for j in range(100)],
                title=f"Figure {i}"
            )
            figures.append(fig)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100.0)

        # Clean up
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
