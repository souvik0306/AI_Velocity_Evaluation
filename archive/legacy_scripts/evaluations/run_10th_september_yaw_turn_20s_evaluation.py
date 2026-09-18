#!/usr/bin/env python3

"""Evaluate 10 September yaw-turn AI and RAW bags over 20 seconds.

Each evaluation window starts five seconds after its flight cutoff. The shared
September runner handles velocity extraction, cleaning, alignment, clipping,
RMSE/bias evaluation, drift-rate calculation, plots, and CSV summaries.
"""

import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATIONS_DIR = Path(__file__).resolve().parent


def _load_runner():
	spec = importlib.util.spec_from_file_location(
		"configurable_september_velocity_runner",
		EVALUATIONS_DIR / "run_10th_september_high_dynamic_evaluation.py",
	)
	if spec is None or spec.loader is None:
		raise ImportError("Could not load the shared September velocity evaluation runner")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


RUNNER = _load_runner()
RUNNER.DEFAULT_BAGS_DIR = REPO_ROOT / "10th_September_Yaw_Turn"
RUNNER.DEFAULT_OUTPUT_DIR = REPO_ROOT / "10th_September_Yaw_Turn_results"
RUNNER.DATASET_DESCRIPTION = "10 September yaw-turn AI/RAW bags"
RUNNER.DATASET_SLUG = "10th_september_yaw_turn"
RUNNER.DURATIONS = (20,)

# Bags are mapped to flight numbers in chronological filename order.
RUNNER.FLIGHTS = {
	"AI": {
		1: {"bag": "flight_2026-07-01-12-57-53.bag", "start": 44.93},
		2: {"bag": "flight_2026-07-01-13-00-24.bag", "start": 38.07},
		3: {"bag": "flight_2026-07-01-13-02-26.bag", "start": 39.44},
		4: {"bag": "flight_2026-07-01-13-04-27.bag", "start": 36.31},
		5: {"bag": "flight_2026-07-01-13-06-42.bag", "start": 31.78},
	},
	"RAW": {
		1: {"bag": "flight_2026-07-01-13-55-52.bag", "start": 38.05},
		2: {"bag": "flight_2026-07-01-13-58-03.bag", "start": 34.77},
		3: {"bag": "flight_2026-07-01-14-00-00.bag", "start": 38.71},
		4: {"bag": "flight_2026-07-01-14-01-56.bag", "start": 32.82},
	},
}

# Retain the supplied endpoints so the shared runner can independently verify
# that every configured interval is exactly 20 seconds long.
RUNNER.SUPPLIED_ENDPOINTS = {
	20: {
		"AI": {1: 64.93, 2: 58.07, 3: 59.44, 4: 56.31, 5: 51.78},
		"RAW": {1: 58.05, 2: 54.77, 3: 58.71, 4: 52.82},
	},
}


if __name__ == "__main__":
	RUNNER.run()
