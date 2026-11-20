#!/usr/bin/env python3
"""Evaluate 19 November medium-dynamic bags over 20-second windows."""

import sys
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parents[2]
if str(EVALUATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATIONS_DIR))

from evaluation_common import REPO_ROOT, evaluate_dataset  # noqa: E402


BAGS_DIR = REPO_ROOT / "data" / "flight_sessions" / "2026-11-19" / "Medium"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-11-19-03-51-36.bag", "start": 41.08},
        2: {"bag": "flight_2026-11-19-03-53-39.bag", "start": 47.04},
        3: {"bag": "flight_2026-11-19-03-55-42.bag", "start": 49.67},
        4: {"bag": "flight_2026-11-19-03-57-45.bag", "start": 39.66},
        5: {"bag": "flight_2026-11-19-03-59-33.bag", "start": 54.91},
        6: {"bag": "flight_2026-11-19-04-01-54.bag", "start": 41.70},
    },
    "RAW": {
        1: {"bag": "flight_2026-11-19-04-17-14.bag", "start": 41.01},
        2: {"bag": "flight_2026-11-19-04-19-12.bag", "start": 41.47},
        3: {"bag": "flight_2026-11-19-04-21-04.bag", "start": 41.20},
        4: {"bag": "flight_2026-11-19-04-22-45.bag", "start": 39.23},
        5: {"bag": "flight_2026-11-19-04-24-24.bag", "start": 39.84},
    },
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        default_duration="20s",
        result_name="19th_nov_medium",
    )
