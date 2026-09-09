"""Run nominal benchmark acceptance/regression checks without training or stress events."""
from __future__ import annotations

# Permit both direct CLI execution and package imports from any working directory.
import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from importlib.metadata import version

CODE_DIR = Path(__file__).resolve().parents[1]
TEST_DIR = CODE_DIR / "tests"
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(CODE_DIR))

# Deferred by the benchmark_v1 scope: do not instantiate any extreme event.
DEFERRED_CASES = {
    "test_load_surge_telemetry_uses_effective_wh1_load",
    "test_phase_imbalance_telemetry_changes_office_not_zoe",
    "test_compound_telemetry_applies_load_multiplier_once",
    "test_episode_render_is_observational_and_equivalent_to_nominal_mode",
    "test_figure_has_five_panels_event_shading_and_no_show",
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main():
    started = time.perf_counter()
    loader = unittest.TestLoader()
    selected, deferred = [], []
    for path in sorted(TEST_DIR.glob("test_*.py")):
        if path.stem.startswith("test_extreme"):
            deferred.append(path.name + ": whole event suite deferred")
            continue
        for test in flatten(loader.loadTestsFromName(path.stem)):
            if getattr(test, "_testMethodName", "") in DEFERRED_CASES:
                deferred.append(test.id())
            else:
                selected.append(test)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.TestSuite(selected))
    log = output.getvalue()
    (CODE_DIR / "benchmark_v1_checks.log").write_text(log, encoding="utf-8")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "acceptance and existing nominal/electrical regression; no training, no holdout rollouts",
        "extreme_events": None,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "numpy", "pandas", "gymnasium", "opendssdirect.py", "dss-python", "dss-python-backend")},
        "elapsed_seconds": time.perf_counter() - started,
        "tests_run": result.testsRun,
        "passed": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
        "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "selected_test_ids": [test.id() for test in selected],
        "deferred": deferred,
        "failures": [{"test": test.id(), "traceback": trace} for test, trace in result.failures],
        "errors_detail": [{"test": test.id(), "traceback": trace} for test, trace in result.errors],
        "scientific_c_unserved": 1.0,
        "coefficient_note": "Phase 2 approved 1.0 USD/kWh; other test coefficients are controlled fixtures.",
    }
    (CODE_DIR / "benchmark_v1_checks.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(log[-14000:])
    print(json.dumps({key: report[key] for key in ("tests_run", "passed", "failed", "errors", "skipped", "elapsed_seconds")}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

