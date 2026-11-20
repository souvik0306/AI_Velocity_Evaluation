#!/usr/bin/env python3
"""Evaluate 22 September hover flight bags over configured 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "22nd_Sept" / "Hover_less_RC"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-07-01-14-41-46.bag", "start": 45.05},
        2: {"bag": "flight_2026-07-01-14-45-13.bag", "start": 37.39},
        3: {"bag": "flight_2026-07-01-14-47-42.bag", "start": 36.06},
        4: {"bag": "flight_2026-07-01-14-50-12.bag", "start": 37.28},
        5: {"bag": "flight_2026-07-01-14-53-18.bag", "start": 34.52},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-14-11-04.bag", "start": 113.37},
        2: {"bag": "flight_2026-07-01-14-15-52.bag", "start": 33.27},
        3: {"bag": "flight_2026-07-01-14-18-56.bag", "start": 38.22},
        4: {"bag": "flight_2026-07-01-14-21-56.bag", "start": 36.97},
    },
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        hover_analysis=True,
        result_name="22nd_sept_hover_less_rc",
    )
