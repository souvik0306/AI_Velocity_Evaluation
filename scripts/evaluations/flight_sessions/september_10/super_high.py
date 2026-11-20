#!/usr/bin/env python3
"""Evaluate 10 September super-high flight bags over configured 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402

BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "10th_Sept" / "Super-High"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-07-01-12-45-01.bag", "start": 41.14},
        2: {"bag": "flight_2026-07-01-12-47-25.bag", "start": 40.24},
        3: {"bag": "flight_2026-07-01-12-49-22.bag", "start": 37.5},
        4: {"bag": "flight_2026-07-01-12-51-49.bag", "start": 38.23},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, result_name="super_high")
