#!/usr/bin/env python3
"""Evaluate 7 October yaw bags over configured Vicon-referenced windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "7th_Oct" / "Yaw"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 37.77},
        2: {"bag": "flight_2.bag", "start": 43.85},
        3: {"bag": "flight_3.bag", "start": 38.33},
        4: {"bag": "flight_4.bag", "start": 36.86},
        5: {"bag": "flight_5.bag", "start": 37.49},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 50.12},
        2: {"bag": "flight_2.bag", "start": 36.87},
        3: {"bag": "flight_3.bag", "start": 36.16},
        4: {"bag": "flight_4.bag", "start": 34.98},
        5: {"bag": "flight_5.bag", "start": 39.10},
        6: {"bag": "flight_6.bag", "start": 39.31},
    },
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, result_name="7th_oct_yaw")
