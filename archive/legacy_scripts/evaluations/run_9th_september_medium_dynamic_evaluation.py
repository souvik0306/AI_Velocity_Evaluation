#!/usr/bin/env python3

"""Evaluate 9 September medium-dynamic AI and RAW bags over 20, 30, or 60 seconds."""

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[2]
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
RUNNER.DEFAULT_BAGS_DIR = SCRIPT_DIR / "9sept_medium_dynamic_Rosbags"
RUNNER.DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "9sept_medium_dynamic_results"
RUNNER.DATASET_DESCRIPTION = "9 September medium-dynamic AI/RAW bags"
RUNNER.DATASET_SLUG = "9th_september_medium_dynamic"
RUNNER.DURATIONS = (20, 30, 60)

# Bags are mapped to flight numbers in chronological filename order.
RUNNER.FLIGHTS = {
	"AI": {
		1: {"bag": "flight_2026-07-01-12-09-03.bag", "start": 34.43},
		2: {"bag": "flight_2026-07-01-12-11-43.bag", "start": 41.62},
		3: {"bag": "flight_2026-07-01-12-14-18.bag", "start": 35.62},
	},
	"RAW": {
		1: {"bag": "flight_2026-07-01-13-33-34.bag", "start": 38.48},
		2: {"bag": "flight_2026-07-01-13-36-16.bag", "start": 34.54},
		3: {"bag": "flight_2026-07-01-13-38-39.bag", "start": 36.91},
	},
}

RUNNER.SUPPLIED_ENDPOINTS = {
	20: {
		"AI": {1: 54.43, 2: 61.62, 3: 55.62},
		"RAW": {1: 58.48, 2: 54.54, 3: 56.91},
	},
	30: {
		"AI": {1: 64.43, 2: 71.62, 3: 65.62},
		"RAW": {1: 68.48, 2: 64.54, 3: 66.91},
	},
	60: {
		"AI": {1: 94.43, 2: 101.62, 3: 95.62},
		"RAW": {1: 98.48, 2: 94.54, 3: 96.91},
	},
}


if __name__ == "__main__":
	RUNNER.run()
