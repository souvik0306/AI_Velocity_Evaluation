#!/usr/bin/env python3
"""Evaluate 22 September RC-hovering flight bags over 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "22nd_Sept" / "Hover_RC_hovering"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-07-01-14-59-51.bag", "start": 36.31},
        2: {"bag": "flight_2026-07-01-15-02-42.bag", "start": 34.62},
        3: {"bag": "flight_2026-07-01-15-05-13.bag", "start": 35.69},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-14-28-47.bag", "start": 43.42},
        2: {"bag": "flight_2026-07-01-14-31-33.bag", "start": 39.91},
        3: {"bag": "flight_2026-07-01-14-34-13.bag", "start": 38.23},
    },
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        hover_analysis=True,
        result_name="22nd_sept_hover_rc_hover",
    )
