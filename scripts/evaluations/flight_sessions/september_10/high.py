#!/usr/bin/env python3
"""Evaluate 10 September high flight bags over configured 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402

BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "10th_Sept" / "High"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-07-01-12-23-36.bag", "start": 41.14},
        2: {"bag": "flight_2026-07-01-12-27-17.bag", "start": 40.24},
        3: {"bag": "flight_2026-07-01-12-30-45.bag", "start": 37.5},
        4: {"bag": "flight_2026-07-01-12-34-16.bag", "start": 38.23},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-13-45-38.bag", "start": 36.19},
        2: {"bag": "flight_2026-07-01-13-47-24.bag", "start": 37.1},
        3: {"bag": "flight_2026-07-01-13-49-14.bag", "start": 37.42},
        4: {"bag": "flight_2026-07-01-13-51-01.bag", "start": 31.98},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, result_name="high")
