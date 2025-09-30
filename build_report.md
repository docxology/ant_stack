# Build Report

**Generated:** 2025-09-30T15:34:47.637830
**Status:** ✅ SUCCESS
**Papers Built:** 4
**Papers Failed:** 0

## complexity_energetics

✅ **Status:** SUCCESS

## documentation

✅ **Status:** SUCCESS

## ant_stack

✅ **Status:** SUCCESS

## cohereAnts

✅ **Status:** SUCCESS

## Test Results

✅ **Tests:** PASSED

**Test Output:**
```
============================= test session starts ==============================
platform darwin -- Python 3.13.7, pytest-8.4.2, pluggy-1.6.0 -- /opt/homebrew/opt/python@3.13/bin/python3.13
cachedir: .pytest_cache
hypothesis profile 'default'
benchmark: 5.1.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /Users/4d/Documents/GitHub/ant
plugins: mock-3.15.1, asyncio-1.2.0, anyio-4.9.0, xdist-3.8.0, hypothesis-6.138.15, benchmark-5.1.0, cov-7.0.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 11 items

tests/core_rendering/test_core_refactor.py::TestCoreAnalysis::test_compute_load PASSED [  9%]
tests/core_rendering/test_core_refactor.py::TestCoreAnalysis::test_detailed_energy_estimation PASSED [ 18%]
tests/core_rendering/test_core_refactor.py::TestCoreAnalysis::test_energy_breakdown PASSED [ 27%]
tests/core_rendering/test_core_refactor.py::TestCoreAnalysis::test_energy_coefficients PASSED [ 36%]
tests/core_rendering/test_core_refactor.py::TestCoreAnalysis::test_energy_efficiency_metrics PASSED [ 45%]
tests/core_rendering/test_core_refactor.py::TestCoreAnalysis::test_unit_conversions PASSED [ 54%]
tests/core_rendering/test_core_refactor.py::TestCoreFigures::test_bar_plot_generation PASSED [ 63%]
tests/core_rendering/test_core_refactor.py::TestCoreFigures::test_line_plot_generation PASSED [ 72%]
tests/core_rendering/test_core_refactor.py::TestCoreFigures::test_scatter_plot_generation PASSED [ 81%]
tests/core_rendering/test_core_refactor.py::TestPaperConfiguration::test_paper_config_loading PASSED [ 90%]
tests/core_rendering/test_core_refactor.py::TestIntegrationWorkflow::test_analysis_to_visualization_pipeline PASSED [100%]

============================== 11 passed in 1.40s ==============================

```
