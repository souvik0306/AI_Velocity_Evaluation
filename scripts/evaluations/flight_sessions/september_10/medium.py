#!/usr/bin/env python3
"""Evaluate 10 September medium flight bags over configured 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402

BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "10th_Sept" / "Medium"
FLIGHTS = {
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

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, result_name="medium")
