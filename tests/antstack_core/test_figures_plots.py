#!/usr/bin/env python3
"""Comprehensive tests for antstack_core.figures.plots module.

Tests basic plotting functions including bar plots, line plots, and scatter plots
with matplotlib integration and file output validation.

Following .cursorrules principles:
- Real plotting functionality testing (no mocks)
- File output validation
- Statistical data visualization testing
- Professional documentation with plotting references
"""

import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for testing
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from antstack_core.figures.plots import bar_plot, line_plot, scatter_plot


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestBarPlot(unittest.TestCase):
    """Test bar plot functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.labels = ['A', 'B', 'C', 'D']
        self.values = [10, 20, 15, 25]
        self.ylabel = "Test Values"

    def test_bar_plot_basic(self):
        """Test basic bar plot creation."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            bar_plot(self.labels, self.values, "Test Bar Plot", output_path, self.ylabel)

            # Check that file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_bar_plot_with_errors(self):
        """Test bar plot with error bars."""
        yerr = [1, 2, 1.5, 3]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            bar_plot(self.labels, self.values, "Test Bar Plot with Errors",
                    output_path, self.ylabel, yerr=yerr)

            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_bar_plot_custom_ylabel(self):
        """Test bar plot with custom ylabel."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            bar_plot(self.labels, self.values, "Test Bar Plot",
                    output_path, ylabel="Custom Units")

            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_bar_plot_empty_data(self):
        """Test bar plot with empty data."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            bar_plot([], [], "Empty Plot", output_path)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestLinePlot(unittest.TestCase):
    """Test line plot functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.x = [1, 2, 3, 4, 5]
        self.ys = [[10, 20, 15, 25, 30], [5, 15, 10, 20, 25]]
        self.labels = ['Series 1', 'Series 2']
        self.xlabel = "X Values"
        self.ylabel = "Y Values"

    def test_line_plot_basic(self):
        """Test basic line plot creation."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            line_plot(self.x, self.ys, self.labels, "Test Line Plot",
                     self.xlabel, self.ylabel, output_path)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_line_plot_single_series(self):
        """Test line plot with single data series."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            line_plot(self.x, [self.ys[0]], [self.labels[0]],
                     "Single Series Plot", self.xlabel, self.ylabel, output_path)

            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_line_plot_with_bands(self):
        """Test line plot with confidence bands."""
        bands = [
            [(8, 12), (18, 22), (13, 17), (23, 27), (28, 32)],
            [(3, 7), (13, 17), (8, 12), (18, 22), (23, 27)]
        ]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            line_plot(self.x, self.ys, self.labels, "Test Line Plot with Bands",
                     self.xlabel, self.ylabel, output_path, bands=bands)

            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_line_plot_empty_data(self):
        """Test line plot with empty data."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            line_plot([], [], [], "Empty Plot", "X", "Y", output_path)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_line_plot_mismatched_lengths(self):
        """Test line plot with mismatched data lengths."""
        # matplotlib raises ValueError for mismatched lengths
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            with self.assertRaises(ValueError):
                line_plot([1, 2, 3], [[1, 2]], ["Series"], "Mismatched Plot",
                         "X", "Y", output_path)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestScatterPlot(unittest.TestCase):
    """Test scatter plot functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.x = [1, 2, 3, 4, 5]
        self.y = [10, 20, 15, 25, 30]
        self.xlabel = "X Values"
        self.ylabel = "Y Values"

    def test_scatter_plot_basic(self):
        """Test basic scatter plot creation."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            scatter_plot(self.x, self.y, "Test Scatter Plot",
                        self.xlabel, self.ylabel, output_path)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_scatter_plot_empty_data(self):
        """Test scatter plot with empty data."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            scatter_plot([], [], "Empty Scatter Plot", "X", "Y", output_path)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_scatter_plot_single_point(self):
        """Test scatter plot with single data point."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            scatter_plot([1], [10], "Single Point Plot", "X", "Y", output_path)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestPlotIntegration(unittest.TestCase):
    """Test integration scenarios for plotting functions."""

    def test_multiple_plot_types(self):
        """Test creating multiple plot types in sequence."""
        plot_configs = [
            ("bar", lambda path: bar_plot(['A', 'B'], [1, 2], "Bar", path)),
            ("line", lambda path: line_plot([1, 2], [[1, 2]], ["Line"], "Line", "X", "Y", path)),
            ("scatter", lambda path: scatter_plot([1, 2], [1, 2], "Scatter", "X", "Y", path))
        ]

        for plot_type, plot_func in plot_configs:
            with self.subTest(plot_type=plot_type):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    output_path = tmp.name

                try:
                    plot_func(output_path)
                    self.assertTrue(os.path.exists(output_path))
                    self.assertGreater(os.path.getsize(output_path), 100)  # Reasonable size

                finally:
                    if os.path.exists(output_path):
                        os.unlink(output_path)

    def test_plot_file_formats(self):
        """Test plotting with different file formats."""
        formats = ['.png', '.pdf', '.svg']

        for fmt in formats:
            with self.subTest(format=fmt):
                with tempfile.NamedTemporaryFile(suffix=fmt, delete=False) as tmp:
                    output_path = tmp.name

                try:
                    # Use a simple plot that should work with all formats
                    bar_plot(['A', 'B'], [1, 2], f"Test {fmt}", output_path)
                    self.assertTrue(os.path.exists(output_path))

                finally:
                    if os.path.exists(output_path):
                        os.unlink(output_path)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestPlotDataValidation(unittest.TestCase):
    """Test data validation in plotting functions."""

    def test_bar_plot_data_consistency(self):
        """Test that bar plot handles data consistently."""
        labels = ['A', 'B', 'C']
        values = [1, 2, 3]
        yerr = [0.1, 0.2, 0.3]

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            bar_plot(labels, values, "Consistent Data", output_path, yerr=yerr)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_line_plot_data_consistency(self):
        """Test that line plot handles data consistently."""
        x = [1, 2, 3]
        ys = [[1, 2, 3], [2, 3, 4]]
        labels = ['Line 1', 'Line 2']

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            line_plot(x, ys, labels, "Consistent Data", "X", "Y", output_path)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_scatter_plot_data_consistency(self):
        """Test that scatter plot handles data consistently."""
        x = [1, 2, 3, 4, 5]
        y = [1, 4, 9, 16, 25]  # x^2

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name

        try:
            scatter_plot(x, y, "Consistent Data", "X", "Y", output_path)
            self.assertTrue(os.path.exists(output_path))

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib not available")
class TestPlotErrorHandling(unittest.TestCase):
    """Test error handling in plotting functions."""

    def test_invalid_file_path(self):
        """Test plotting with invalid file path."""
        invalid_path = "/invalid/path/plot.png"

        # These should not raise exceptions (matplotlib handles file errors gracefully)
        try:
            bar_plot(['A'], [1], "Test", invalid_path)
        except Exception:
            # Expected to fail with invalid path
            pass

    def test_none_values(self):
        """Test plotting with None values."""
        # These should handle None gracefully or raise appropriate errors
        with self.assertRaises((TypeError, AttributeError)):
            bar_plot(None, [1], "Test", "/tmp/test.png")

        with self.assertRaises((TypeError, AttributeError)):
            line_plot([1], None, ["Test"], "Test", "X", "Y", "/tmp/test.png")


class TestMatplotlibAvailability(unittest.TestCase):
    """Test matplotlib availability and configuration."""

    def test_matplotlib_import(self):
        """Test that matplotlib can be imported and configured."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        # Test that we can create a basic plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Plot")

        # Should be able to create and close figure
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_backend_configuration(self):
        """Test that matplotlib backend is properly configured."""
        if not HAS_MATPLOTLIB:
            self.skipTest("matplotlib not available")

        # Should be using non-interactive backend
        current_backend = matplotlib.get_backend()
        self.assertIn('Agg', current_backend)  # Agg is non-interactive


if __name__ == '__main__':
    unittest.main()
