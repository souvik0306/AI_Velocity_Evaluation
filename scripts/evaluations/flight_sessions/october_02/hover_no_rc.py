#!/usr/bin/env python3
"""Evaluate 2 October no-RC hover bags over configured 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "2nd_Oct" / "Hover_no_RC"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 63.96},
        2: {"bag": "flight_2.bag", "start": 36.37},
        3: {"bag": "flight_3.bag", "start": 41.80},
    }
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        hover_analysis=True,
        bags_grouped=False,
        result_name="2nd_oct_hover_no_rc",
    )
