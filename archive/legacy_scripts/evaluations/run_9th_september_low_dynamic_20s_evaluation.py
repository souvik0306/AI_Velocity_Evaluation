#!/usr/bin/env python3

"""Evaluate 9 September low-dynamic AI and RAW bags over 20 seconds.

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
RUNNER.DEFAULT_BAGS_DIR = REPO_ROOT / "9th_September_Low_Dynamic_Rosbags"
RUNNER.DEFAULT_OUTPUT_DIR = REPO_ROOT / "9th_September_Low_Dynamic_results"
RUNNER.DATASET_DESCRIPTION = "9 September low-dynamic AI/RAW bags"
RUNNER.DATASET_SLUG = "9th_september_low_dynamic"
RUNNER.DURATIONS = (20,)

# AI bags use their explicit flight numbers. RAW bags are mapped to flight
# numbers in chronological filename order, consistent with the other runners.
RUNNER.FLIGHTS = {
	"AI": {
		1: {"bag": "flight_1.bag", "start": 44.24},
		2: {"bag": "flight_2.bag", "start": 32.99},
		3: {"bag": "flight_3.bag", "start": 35.95},
		4: {"bag": "flight_4.bag", "start": 41.47},
	},
	"RAW": {
		1: {"bag": "flight_2026-07-01-13-16-49.bag", "start": 48.51},
		2: {"bag": "flight_2026-07-01-13-19-00.bag", "start": 33.29},
		3: {"bag": "flight_2026-07-01-13-20-51.bag", "start": 34.77},
		4: {"bag": "flight_2026-07-01-13-22-50.bag", "start": 39.63},
		5: {"bag": "flight_2026-07-01-13-25-41.bag", "start": 35.16},
	},
}

# Retain the supplied endpoints so the shared runner can independently verify
# that every configured interval is exactly 20 seconds long.
RUNNER.SUPPLIED_ENDPOINTS = {
	20: {
		"AI": {1: 64.24, 2: 52.99, 3: 55.95, 4: 61.47},
		"RAW": {1: 68.51, 2: 53.29, 3: 54.77, 4: 59.63, 5: 55.16},
	},
}


if __name__ == "__main__":
	RUNNER.run()
