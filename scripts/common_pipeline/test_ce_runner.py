#!/usr/bin/env python3
import math

from antstack_core.analysis import (
    ComputeLoad,
    EnergyCoefficients,
    aj_to_j,
    estimate_compute_energy,
    estimate_body_energy_mech,
    pj_to_j,
)


def main() -> int:
    coeff = EnergyCoefficients()
    # Zero load
    assert estimate_compute_energy(ComputeLoad(), coeff) == 0.0
    # Unit conversions
    assert math.isclose(pj_to_j(1e12), 1.0)
    assert math.isclose(aj_to_j(1e18), 1.0)
    # Body energy partition sanity
    coeff2 = EnergyCoefficients(body_per_joint_w=2.0, body_sensor_w_per_channel=0.005)
    e_body = estimate_body_energy_mech(1.0, {"J": 10, "S": 100}, coeff2)
    assert math.isclose(e_body, 20.5, rel_tol=1e-6)
    print("ce sanity tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

