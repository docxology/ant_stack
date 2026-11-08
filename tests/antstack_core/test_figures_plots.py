"""Consumer-visible interval rendering, label fidelity and import-state invariants."""
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest

from antstack_core.figures import PlotConfig, interval_plot, line_plot


def test_wide_positive_intervals_preserve_true_bounds_and_nonzero_labels(tmp_path):
    figure = interval_plot(["Tiny", "Large"], [5.8e-5, 5417.8],
                           confidence_intervals=[(1e-5, 2e-5), (6000, 7000)],
                           out_path=str(tmp_path / "intervals.svg"), config=PlotConfig(close_on_save=False))
    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        np.testing.assert_allclose(axis.collections[1].get_segments(),
                                   [[[1e-5, 0], [2e-5, 0]], [[6000, 1], [7000, 1]]])
        np.testing.assert_allclose(axis.collections[0].get_offsets()[:, 0], [5.8e-5, 5417.8])
        assert all(float(text.get_text()) > 0 for text in axis.texts)
    finally:
        plt.close(figure)


def test_zero_values_and_nonpositive_bounds_remain_visible_on_linear_axis():
    figure = interval_plot(["Zero", "Positive"], [0, 10], confidence_intervals=[(-1, 1), (9, 11)])
    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == "linear"
        np.testing.assert_allclose(axis.collections[0].get_offsets()[:, 0], [0, 10])
        np.testing.assert_allclose(axis.collections[1].get_segments()[0], [[-1, 0], [1, 0]])
    finally:
        plt.close(figure)


def test_invalid_interval_and_mismatched_series_preserve_prior_output(tmp_path):
    path = tmp_path / "reviewed.svg"
    prior = '<svg xmlns="http://www.w3.org/2000/svg"><title>Reviewed</title></svg>'
    path.write_text(prior)
    with pytest.raises(ValueError):
        interval_plot(["x"], [1], out_path=str(path), confidence_intervals=[(2, 1)])
    assert path.read_text() == prior
    with pytest.raises(ValueError):
        line_plot([1, 2, 3], [[1, 2]], ["mismatched"], out_path=str(path))
    assert path.read_text() == prior


def test_import_preserves_callers_backend_style_and_existing_figure(tmp_path):
    code = """import json,matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.facecolor']='#123456'
figure=plt.figure()
import antstack_core
print(json.dumps({'backend':matplotlib.get_backend(),'facecolor':matplotlib.rcParams['axes.facecolor'],'open':plt.fignum_exists(figure.number)}))
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path,
                            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=40, check=True)
    assert json.loads(result.stdout) == {"backend": "svg", "facecolor": "#123456", "open": True}
