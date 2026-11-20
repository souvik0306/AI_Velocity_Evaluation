#!/usr/bin/env python3
"""Evaluate 19 November hover bags over Vicon-referenced 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "2026-11-19" / "Hover"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-11-19-03-33-01.bag", "start": 59.04},
        2: {"bag": "flight_2026-11-19-03-36-08.bag", "start": 57.06},
        3: {"bag": "flight_2026-11-19-03-38-59.bag", "start": 40.90},
        4: {"bag": "flight_2026-11-19-03-41-26.bag", "start": 57.36},
        5: {"bag": "flight_2026-11-19-03-44-24.bag", "start": 56.24},
    },
    "RAW": {
        1: {"bag": "flight_2026-11-19-04-05-10.bag", "start": 75.12},
        2: {"bag": "flight_2026-11-19-04-07-49.bag", "start": 37.57},
        3: {"bag": "flight_2026-11-19-04-09-46.bag", "start": 35.11},
        4: {"bag": "flight_2026-11-19-04-11-31.bag", "start": 42.76},
        5: {"bag": "flight_2026-11-19-04-13-28.bag", "start": 34.34},
    },
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        hover_analysis=True,
        default_duration="20s",
        result_name="19th_nov_hover",
    )
