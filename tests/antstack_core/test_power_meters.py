"""Deterministic meter arithmetic and availability contracts; no hardware probes."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from antstack_core.analysis import power_meters as meters


def rapl_counter(tmp_path, value=0, counter_range=None):
    path = tmp_path / 'energy_uj'
    path.write_text(str(value), encoding='utf-8')
    if counter_range is not None:
        (tmp_path / 'max_energy_range_uj').write_text(str(counter_range), encoding='utf-8')
    return meters.RaplPowerMeter(str(path))


@pytest.mark.parametrize('watts', [-1, float('nan'), float('inf')])
def test_invalid_power_is_not_a_measurement(watts):
    with pytest.raises(ValueError):
        meters.PowerSample(0, watts)


def test_initial_rapl_sample_is_not_an_observed_zero(tmp_path):
    meter = rapl_counter(tmp_path, 0)
    sample = meter.read()
    assert sample.available is False
    assert sample.watts is None
    assert sample.error


@pytest.mark.parametrize('counter', [0, 1000000])
def test_unchanged_counter_has_exactly_zero_power(tmp_path, counter):
    meter = rapl_counter(tmp_path, counter)
    with patch.object(meters.time, 'monotonic', side_effect=[0.0, 1.0]):
        meter.read()
        sample = meter.read()
    assert sample.available
    assert sample.watts == 0.0


def test_power_uses_monotonic_elapsed_not_wall_clock(tmp_path):
    meter = rapl_counter(tmp_path)
    with patch.object(meter, 'read_energy_uj', side_effect=[1000000, 3000000]), \
         patch.object(meters.time, 'monotonic', side_effect=[10.0, 10.5]), \
         patch.object(meters.time, 'time', side_effect=[100.0, 90.0]):
        meter.read()
        sample = meter.read()
    assert sample.watts == 4.0
    assert sample.timestamp == 90.0


@pytest.mark.parametrize('known_range', [True, False])
def test_single_wrap_uses_declared_range(tmp_path, known_range):
    meter = rapl_counter(tmp_path, counter_range=None if known_range else 1000)
    if known_range:
        meter.max_energy_range_uj = 1000
    with patch.object(meter, 'read_energy_uj', side_effect=[990, 10]), \
         patch.object(meters.time, 'monotonic', side_effect=[0.0, 1.0]):
        meter.read()
        sample = meter.read()
    assert sample.watts == pytest.approx(20e-6)


def test_unknown_wrap_range_is_unavailable(tmp_path):
    meter = rapl_counter(tmp_path)
    with pytest.raises(meters.PowerMeasurementUnavailable):
        meter.energy_delta_uj(990, 10)


@pytest.mark.parametrize('counter', [-1, 'nan', 'inf', 'not a counter'])
def test_invalid_rapl_file_is_unavailable(tmp_path, counter):
    meter = rapl_counter(tmp_path, counter)
    with pytest.raises(meters.PowerMeasurementUnavailable):
        meter.read_energy_uj()


def test_missing_and_unreadable_rapl_are_not_zero(tmp_path):
    meter = meters.RaplPowerMeter(str(tmp_path / 'missing'))
    with pytest.raises(meters.PowerMeasurementUnavailable):
        meter.read()
    with patch('builtins.open', side_effect=PermissionError('denied')):
        with pytest.raises(meters.PowerMeasurementUnavailable):
            meter.read_energy_uj()


def test_nonpositive_rapl_elapsed_is_unavailable(tmp_path):
    meter = rapl_counter(tmp_path)
    with patch.object(meters.time, 'monotonic', side_effect=[1.0, 1.0]):
        meter.read()
        with pytest.raises(meters.PowerMeasurementUnavailable):
            meter.read()


def test_out_of_range_counter_is_unavailable(tmp_path):
    meter = rapl_counter(tmp_path)
    meter.max_energy_range_uj = 1000
    with pytest.raises(meters.PowerMeasurementUnavailable):
        meter.energy_delta_uj(1000, 10)


def test_cumulative_context_counts_wrap_and_elapsed(tmp_path):
    meter = rapl_counter(tmp_path)
    meter.max_energy_range_uj = 1000
    with patch.object(meter, 'read_energy_uj', side_effect=[990, 10]), \
         patch.object(meters.time, 'monotonic', side_effect=[4.0, 6.0]):
        with meters.measure_energy(meter) as measurement:
            assert measurement['energy_joules'] is None
    assert measurement['status'] == 'measured'
    assert measurement['measured'] is True
    assert measurement['duration_seconds'] == 2.0
    assert measurement['energy_joules'] == pytest.approx(20e-6)
    assert measurement['average_power_watts'] == pytest.approx(10e-6)


def test_instantaneous_context_trapezoid():
    meter = Mock(spec=meters.PowerMeter)
    meter.read.side_effect = [meters.PowerSample(100, 2), meters.PowerSample(90, 6)]
    with patch.object(meters.time, 'monotonic', side_effect=[1.0, 4.0]):
        with meters.measure_energy(meter) as measurement:
            pass
    assert measurement['energy_joules'] == 12.0
    assert measurement['average_power_watts'] == 4.0
    assert len(measurement['samples']) == 2


@pytest.mark.parametrize('fail_at_end', [False, True])
def test_hardware_failure_does_not_become_observed_zero(fail_at_end):
    meter = Mock(spec=meters.PowerMeter)
    fault = meters.PowerMeasurementUnavailable('device unavailable')
    meter.read.side_effect = [meters.PowerSample(1, 2), fault] if fail_at_end else fault
    with pytest.raises(meters.PowerMeasurementUnavailable):
        with meters.measure_energy(meter) as measurement:
            pass
    if fail_at_end:
        assert measurement['measured'] is False
        assert measurement['energy_joules'] is None


def test_null_context_is_explicitly_disabled_not_empirical():
    with patch.object(meters.time, 'monotonic', side_effect=[0.0, 1.0]):
        with meters.measure_energy() as measurement:
            pass
    assert measurement['status'] == 'disabled'
    assert measurement['measured'] is False
    assert measurement['energy_joules'] == 0.0


def test_nvml_conversion_and_lazy_initialization(monkeypatch):
    backend = SimpleNamespace(nvmlInit=Mock(), nvmlDeviceGetHandleByIndex=Mock(return_value='gpu'),
                              nvmlDeviceGetPowerUsage=Mock(return_value=150000))
    monkeypatch.setattr(meters, 'pynvml', backend)
    meter = meters.NvmlPowerMeter(1)
    assert backend.nvmlInit.call_count == 0
    assert meter.read().watts == 150.0
    assert meter.read().watts == 150.0
    backend.nvmlInit.assert_called_once_with()
    backend.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)


def test_missing_nvml_is_unavailable(monkeypatch):
    monkeypatch.setattr(meters, 'pynvml', None)
    with pytest.raises(meters.PowerMeasurementUnavailable):
        meters.NvmlPowerMeter().read()


@pytest.mark.parametrize('failure', ['initialization', 'reading'])
def test_nvml_errors_are_unavailable(monkeypatch, failure):
    backend = SimpleNamespace(nvmlInit=Mock(), nvmlDeviceGetHandleByIndex=Mock(return_value='gpu'),
                              nvmlDeviceGetPowerUsage=Mock(return_value=150000))
    if failure == 'initialization':
        backend.nvmlInit.side_effect = RuntimeError('driver error')
    else:
        backend.nvmlDeviceGetPowerUsage.side_effect = RuntimeError('read error')
    monkeypatch.setattr(meters, 'pynvml', backend)
    with pytest.raises(meters.PowerMeasurementUnavailable):
        meters.NvmlPowerMeter().read()


def test_factories_do_not_access_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(meters, 'pynvml', None)
    assert isinstance(meters.create_power_meter('null'), meters.NullPowerMeter)
    rapl = meters.create_power_meter('rapl', energy_uj_path=str(tmp_path / 'counter'), max_energy_range_uj=1000)
    assert rapl.max_energy_range_uj == 1000
    assert isinstance(meters.create_power_meter('nvml'), meters.NvmlPowerMeter)
    with pytest.raises(ValueError):
        meters.create_power_meter('unsupported')
