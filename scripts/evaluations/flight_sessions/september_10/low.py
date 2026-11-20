#!/usr/bin/env python3
"""Evaluate 10 September low flight bags over configured 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402

BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "10th_Sept" / "Low-Dynamic"
FLIGHTS = {
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

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, result_name="low")
