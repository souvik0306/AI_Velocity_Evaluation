#!/usr/bin/env python3
"""Evaluate 31 July flight bags over the archived 10-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "2026-07-31"

# Migrated from:
# archive/legacy_scripts/evaluations/
# run_31st_july_10s_speed_regime_heading_magnitude.py
#
# The data folder is named UN, while the archived scripts called this group AI.
# RAW flight 2 was not part of the archived evaluation configuration.
FLIGHTS = {
	"UN": {
		1: {"bag": "flight_1.bag", "start": 36.67},
		2: {"bag": "flight_2.bag", "start": 37.12},
		3: {"bag": "flight_3.bag", "start": 37.55},
		4: {"bag": "flight_4.bag", "start": 34.63},
		5: {"bag": "flight_5.bag", "start": 33.57},
		6: {"bag": "flight_6.bag", "start": 32.35},
		7: {"bag": "flight_7.bag", "start": 33.98},
		8: {"bag": "flight_8.bag", "start": 36.93},
	},
	"RAW": {
		1: {"bag": "flight_1.bag", "start": 61.86},
		3: {"bag": "flight_3.bag", "start": 118.44},
		4: {"bag": "flight_4.bag", "start": 37.97},
		5: {"bag": "flight_5.bag", "start": 30.37},
		6: {"bag": "flight_6.bag", "start": 38.23},
		7: {"bag": "flight_7.bag", "start": 42.09},
		8: {"bag": "flight_8.bag", "start": 34.59},
		9: {"bag": "flight_9.bag", "start": 36.91},
		10: {"bag": "flight_10.bag", "start": 35.02},
	},
}


if __name__ == "__main__":
	evaluate_dataset(BAGS_DIR, FLIGHTS, default_duration="10s")
