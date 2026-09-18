#!/usr/bin/env python3

"""Calculate 9 September medium-dynamic VHE for 20s, 30s, and 60s windows."""

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2]
EVALUATIONS_DIR = Path(__file__).resolve().parent


def _load_runner():
	spec = importlib.util.spec_from_file_location(
		"configurable_velocity_direction_error_runner",
		EVALUATIONS_DIR / "run_10th_september_velocity_direction_error.py",
	)
	if spec is None or spec.loader is None:
		raise ImportError("Could not load the shared velocity direction-error runner")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


RUNNER = _load_runner()
RUNNER.DEFAULT_RESULTS_DIR = SCRIPT_DIR / "9sept_medium_dynamic_results"
RUNNER.DEFAULT_OUTPUT_DIR = RUNNER.DEFAULT_RESULTS_DIR / "VHE"
RUNNER.DURATIONS = (20, 30, 60)
RUNNER.GROUPS = ("AI", "RAW")
RUNNER.FLIGHTS = (1, 2, 3)
RUNNER.DATASET_DESCRIPTION = "9 September medium-dynamic velocity direction error"
RUNNER.DATASET_SLUG = "9th_september_medium_dynamic"
RUNNER.INPUT_SUMMARY_SLUG = "9th_september_medium_dynamic"


if __name__ == "__main__":
	RUNNER.run()
