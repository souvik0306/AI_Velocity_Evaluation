#!/usr/bin/env python3
"""Evaluate 14 October low-dynamic bags over Vicon-referenced 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "14th_Oct" / "Low"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 46.39},
        2: {"bag": "flight_2.bag", "start": 41.23},
        3: {"bag": "flight_3.bag", "start": 38.41},
        4: {"bag": "flight_4.bag", "start": 59.76},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 37.88},
        2: {"bag": "flight_2.bag", "start": 39.02},
        3: {"bag": "flight_3.bag", "start": 44.80},
    },
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        default_duration="20s",
        result_name="14th_oct_low",
    )
