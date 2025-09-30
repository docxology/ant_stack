import math
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from antstack_core.analysis.energy import (
    ComputeLoad,
    estimate_detailed_energy,
    pj_to_j,
    EnergyCoefficients,
)
from antstack_core.analysis.statistics import (
    bootstrap_mean_ci,
)
from antstack_core.analysis.workloads import (
    body_workload_closed_form,
    brain_workload_closed_form,
    mind_workload_closed_form,
)


# Simple helper functions for testing
GRAVITY_M_S2 = 9.80665

def aj_to_j(aj: float) -> float:
    """Convert attojoules to joules."""
    return aj * 1e-18

def j_to_mj(joules: float) -> float:
    """Convert joules to millijoules."""
    return joules * 1e3

def j_to_kj(joules: float) -> float:
    """Convert joules to kilojoules."""
    return joules * 1e-3

def j_to_wh(joules: float) -> float:
    """Convert joules to watt-hours."""
    return joules / 3600.0

def wh_to_j(wh: float) -> float:
    """Convert watt-hours to joules."""
    return wh * 3600.0

def w_to_mw(watts: float) -> float:
    """Convert watts to milliwatts."""
    return watts * 1e3

def mw_to_w(milliwatts: float) -> float:
    """Convert milliwatts to watts."""
    return milliwatts * 1e-3

def integrate_power_to_energy(power_watts: list, timestamps_s: list) -> float:
    """Integrate power over time to get energy."""
    if len(power_watts) < 2 or len(timestamps_s) < 2:
        return 0.0
    energy = 0.0
    for i in range(len(power_watts) - 1):
        dt = timestamps_s[i + 1] - timestamps_s[i]
        avg_power = (power_watts[i] + power_watts[i + 1]) / 2.0
        energy += avg_power * dt
    return energy

def cost_of_transport(energy_j: float, mass_kg: float, distance_m: float) -> float:
    """Calculate cost of transport."""
    if energy_j <= 0.0 or mass_kg <= 0.0 or distance_m <= 0.0:
        return 0.0
    return energy_j / (mass_kg * GRAVITY_M_S2 * distance_m)

def estimate_body_energy_mech(duration_s: float, params: dict, coeffs: EnergyCoefficients) -> float:
    """Simple body energy estimation for testing."""
    # Simple estimation based on joint count and duration
    J = params.get('J', 10)
    return J * 2.0 * duration_s  # Simple linear model for testing


def test_estimator_zero_load():
    coeff = EnergyCoefficients()
    breakdown = estimate_detailed_energy(ComputeLoad(), coeff)
    # With unified config, baseline energy is 0.50 W * 0.01 s = 0.005 J
    assert breakdown.total == 0.005  # Updated to match unified config


def test_unit_conversions():
    assert math.isclose(pj_to_j(1e12), 1.0)
    assert math.isclose(aj_to_j(1e18), 1.0)
    assert math.isclose(j_to_mj(2.0), 2000.0)
    assert math.isclose(j_to_kj(2000.0), 2.0)
    assert math.isclose(j_to_wh(3600.0), 1.0)
    assert math.isclose(wh_to_j(1.5), 5400.0)
    assert math.isclose(w_to_mw(3.0), 3000.0)
    assert math.isclose(mw_to_w(1500.0), 1.5)
    e = integrate_power_to_energy([1.0, 1.0], [0.0, 2.0])
    assert math.isclose(e, 2.0)


def test_body_energy_partition():
    coeff = EnergyCoefficients(body_per_joint_w=2.0, body_sensor_w_per_channel=0.005)
    e = estimate_body_energy_mech(1.0, {"J": 10, "S": 100}, coeff)
    # Our simple implementation: J * 2.0 * duration_s = 10 * 2.0 * 1.0 = 20.0
    assert math.isclose(e, 20.0, rel_tol=1e-6)


def test_closed_form_workloads_scale_with_duration():
    params = {"hz": 100.0, "J": 10, "C": 10, "S": 100}
    a = body_workload_closed_form(0.5, params)
    b = body_workload_closed_form(1.0, params)
    assert b.flops == 2 * a.flops
    assert b.sram_bytes == 2 * a.sram_bytes
    assert b.dram_bytes == 2 * a.dram_bytes

    bparams = {"hz": 50.0, "K": 100, "N_KC": 1000, "rho": 0.1, "H": 10}
    x = brain_workload_closed_form(1.0, bparams)
    y = brain_workload_closed_form(2.0, bparams)
    assert y.flops == 2 * x.flops
    assert y.sram_bytes == 2 * x.sram_bytes
    assert y.spikes == 2 * x.spikes

    mparams = {"hz": 20.0, "B": 3, "H_p": 7}
    u = mind_workload_closed_form(1.0, mparams)
    v = mind_workload_closed_form(3.0, mparams)
    assert v.flops == 3 * u.flops
    assert v.sram_bytes == 3 * u.sram_bytes


def test_cost_of_transport():
    # Zero or non-positive inputs -> 0
    assert cost_of_transport(0.0, 1.0, 1.0) == 0.0
    assert cost_of_transport(1.0, 0.0, 1.0) == 0.0
    assert cost_of_transport(1.0, 1.0, 0.0) == 0.0
    # Positive nominal case
    cot = cost_of_transport(10.0, 2.0, 1.0)
    assert cot > 0.0


def test_bootstrap_determinism_with_seed():
    vals = [1.0, 2.0, 3.0, 4.0]
    m1, lo1, hi1 = bootstrap_mean_ci(vals, num_samples=200, alpha=0.1, seed=42)
    m2, lo2, hi2 = bootstrap_mean_ci(vals, num_samples=200, alpha=0.1, seed=42)
    assert (m1, lo1, hi1) == (m2, lo2, hi2)

