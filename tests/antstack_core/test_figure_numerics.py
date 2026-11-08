"""Numerical pairing, accounting and honest scientific plot regressions."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from antstack_core.figures import plots, publication_plots, advanced_plots
from antstack_core.analysis import ce_publication_figures as producer
from antstack_core.analysis import ce_tables_and_numbers as tables


@pytest.mark.parametrize('plot', [plots.line_plot, publication_plots.publication_line_plot])
def test_line_fits_keep_joint_finite_observations(plot):
    x = [1, 2, np.nan, 4, 5, 6]
    y = [1, np.nan, 9, 16, 25, 36]
    if plot is plots.line_plot:
        figure = plot(x, [y], ['square'])
    else:
        figure = plot(x, [y], ['square'])
    try:
        axis = figure.axes[0]
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), [1, 4, 5, 6])
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [1, 16, 25, 36])
        annotations = ' '.join(text.get_text() for text in axis.texts)
        assert '2.00' in annotations
    finally:
        plt.close(figure)


@pytest.mark.parametrize('plot', [plots.scatter_plot, publication_plots.publication_scatter_plot])
def test_scatter_joint_filter_keeps_real_pairs_and_fit(plot):
    figure = plot([1, 2, np.nan, 4], [3, np.nan, 100, 9])
    try:
        axis = figure.axes[0]
        np.testing.assert_array_equal(axis.collections[0].get_offsets(), [[1, 3], [4, 9]])
        fitted = axis.lines[0]
        np.testing.assert_allclose(fitted.get_ydata(), 2 * fitted.get_xdata() + 1)
    finally:
        plt.close(figure)


@pytest.mark.parametrize('plot', [plots.scatter_plot, publication_plots.publication_scatter_plot])
def test_pair_length_and_shape_errors_are_explicit(plot):
    with pytest.raises(ValueError, match='equal lengths'):
        plot([1, 2], [3])
    with pytest.raises(ValueError, match='one-dimensional'):
        plot([[1, 2]], [[3, 4]])


def test_line_rejects_series_cardinality_and_reversed_bands():
    with pytest.raises(ValueError, match='one x sequence and one label'):
        plots.line_plot([1, 2], [[3, 4], [5, 6]], ['missing'])
    with pytest.raises(ValueError, match='lower bounds'):
        plots.line_plot([1, 2], [[3, 4]], ['x'], bands=[[(4, 2), (3, 5)]])


def test_line_bands_follow_same_finite_pairs():
    figure = plots.line_plot([1, 2, 3], [[1, np.nan, 9]], ['x'],
                             bands=[[(0, 2), (3, 5), (8, 10)]])
    try:
        np.testing.assert_array_equal(figure.axes[0].lines[0].get_xdata(), [1, 3])
        assert '95%' not in figure.axes[0].collections[0].get_label()
    finally:
        plt.close(figure)


def test_publication_captions_do_not_invent_inference(tmp_path):
    manager = publication_plots.FigureManager(tmp_path)
    publication_plots.publication_bar_plot(['a', 'b'], [1, 3], figure_manager=manager)
    publication_plots.publication_line_plot([1, 2, 3], [[1, 4, 9]], ['x'], figure_manager=manager)
    captions = ' '.join(item['caption'] for item in manager.figures.values())
    assert 'significant differences' not in captions
    assert 'Error bars and confidence intervals' not in captions
    assert 'population SD' in captions
    plt.close('all')


def test_singular_kde_is_observed_scatter_without_density_claim(tmp_path):
    manager = publication_plots.FigureManager(tmp_path)
    publication_plots.publication_scatter_plot([1, 1, 1], [2, 2, 2], figure_manager=manager)
    caption = next(iter(manager.figures.values()))['caption']
    assert 'Point density coloring' not in caption
    assert 'Regression analysis' not in caption
    advanced_plots.complexity_entropy_diagram([0.5] * 3, [1] * 3, 'Unfit', str(tmp_path / 'ce.png'))
    assert (tmp_path / 'ce.png').stat().st_size > 0
    plt.close('all')


def test_collinear_interpolation_stays_observed_scatter_and_domain_rejected(monkeypatch, tmp_path):
    manager = publication_plots.FigureManager(tmp_path)
    real_close = plt.close
    monkeypatch.setattr(plt, 'close', lambda *_args: None)
    try:
        advanced_plots.thermodynamic_phase_diagram([1, 2, 3], [10, 20, 30], [0.2, 0.4, 0.6],
                                                   out_path=str(tmp_path / 'map.png'), figure_manager=manager)
        axis = plt.gcf().axes[0]
        np.testing.assert_array_equal(axis.collections[0].get_offsets(), [[1, 10], [2, 20], [3, 30]])
        assert not axis.lines  # No critical-temperature line or phase boundary was analyzed.
        assert (tmp_path / 'map.png').stat().st_size > 0
        with pytest.raises(ValueError, match='Temperature'):
            advanced_plots.thermodynamic_phase_diagram([0], [10], [0.2])
    finally:
        real_close('all')


def test_isolated_nodes_render_equal_without_zero_division(tmp_path):
    result = advanced_plots.network_visualization([[0, 0], [0, 0]], layout='circular', out_path=str(tmp_path / 'network.png'))
    assert result == 'fig:network'
    assert (tmp_path / 'network.png').stat().st_size > 0


def test_component_accounting_includes_all_modules_and_converts_once():
    breakdowns = [SimpleNamespace(actuation=1, sensing=2, compute_flops=3, compute_memory=4,
                                 compute_spikes=5, baseline=6) for _ in range(3)]
    components = producer._component_totals_mj(breakdowns)
    assert components == {'Actuation': 3000, 'Sensing': 6000, 'Compute': 36000, 'Baseline': 18000}
    data = producer._standard_module_data()
    real_breakdowns = [item['breakdown'] for item in data.values()]
    assert real_breakdowns[0].compute_flops > 0
    totals = producer._component_totals_mj(real_breakdowns)
    assert totals['Baseline'] == pytest.approx(15)
    assert sum(totals.values()) == pytest.approx(sum(energy.total for energy in real_breakdowns) * 1000)
    assert totals['Actuation'] == 0
    assert totals['Sensing'] == 0


def test_scaling_stats_come_from_supplied_values_without_fixed_uncertainty():
    stats = producer._descriptive_scaling_stats([1, 2, 4, 8], [1, 8, 64, 512])
    assert float(stats['Descriptive power-law exponent']) == pytest.approx(3)
    assert float(stats['Log-log fit R²']) == pytest.approx(1)
    assert all('±' not in str(value) for value in stats.values())


def test_contact_table_contains_independently_calculated_counters():
    # At J=18, C=15: PGS=15*120*15, SI=15*18*20+18**3/3,
    # LDLT=(18+3*15)**3/3. Memory follows the declared dense model.
    table = tables.generate_contact_solver_table()
    assert '| PGS | 18 | 15 | 27000 | 3312 | 15 |' in table
    assert '| SI | 18 | 15 | 7344 | 4032 | 1 |' in table
    assert '| LDLT | 18 | 15 | 83349 | 31752 | 1 |' in table



@pytest.mark.parametrize('module_name', ['ce_comprehensive_analysis', 'ce_module_scaling_analysis'])
def test_adapter_manifest_load_failure_is_not_replaced_by_defaults(monkeypatch, tmp_path, module_name):
    import importlib
    module = importlib.import_module('antstack_core.analysis.' + module_name)
    def unavailable(_path):
        raise ValueError('invalid supplied scientific manifest')
    monkeypatch.setattr(module.ExperimentManifest, 'load', unavailable)
    args = SimpleNamespace(output=str(tmp_path), manifest='invalid.yaml', modules=['body'])
    with pytest.raises(ValueError, match='invalid supplied scientific manifest'):
        module.run(args)
    assert not list(tmp_path.glob('*results.json'))


def test_analytical_report_preserves_points_without_empirical_validation(tmp_path):
    import json
    from antstack_core.analysis.ce_comprehensive_analysis import generate_veridical_report
    points = [{'parameter_value': value, 'energy': value ** 2, 'flops': value * 10} for value in (1, 2, 4)]
    generate_veridical_report({'module_analyses': [{'module': 'body', 'parameter': 'J',
                              'computational_data': points}], 'configuration': {'coefficients': {'baseline_w': 0.5}}}, tmp_path)
    summary = json.loads((tmp_path / 'data_driven_summary.json').read_text())
    assert summary['modules'][0]['points'] == points
    assert summary['modules'][0]['descriptive_power_fit']['scaling_exponent'] == pytest.approx(2)


@pytest.mark.parametrize('image_format,signature', [('png', b'\x89PNG'), ('svg', b'<?xml'), ('pdf', b'%PDF-')])
def test_publication_producer_emits_requested_real_asset_format(tmp_path, image_format, signature):
    manager = publication_plots.FigureManager(tmp_path)
    producer.generate_energy_breakdown_figures(tmp_path, manager, image_format=image_format)
    for stem in ('module_energy_comparison', 'energy_component_breakdown'):
        assert (tmp_path / f'{stem}.{image_format}').read_bytes().startswith(signature)


