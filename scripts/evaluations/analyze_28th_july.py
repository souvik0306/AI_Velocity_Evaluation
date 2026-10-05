#!/usr/bin/env python3
"""Evaluate 28 July flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "2026-07-28"
FLIGHTS = {
    "UN": {
        1: {"bag": "flight_2026-07-01-10-20-35.bag", "start": 66.20},
        2: {"bag": "flight_2026-07-01-10-31-13.bag", "start": 59.65},
        3: {"bag": "flight_2026-07-01-10-34-48.bag", "start": 48.81},
        4: {"bag": "flight_2026-07-01-10-37-53.bag", "start": 45.39},
        5: {"bag": "flight_2026-07-01-10-40-49.bag", "start": 47.90},
        6: {"bag": "flight_2026-07-01-10-43-37.bag", "start": 41.65},
        7: {"bag": "flight_2026-07-01-10-46-31.bag", "start": 42.18},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-10-18-50.bag", "start": 36.89},
        2: {"bag": "flight_2026-07-01-10-21-43.bag", "start": 38.09},
        3: {"bag": "flight_2026-07-01-10-24-22.bag", "start": 38.54},
        4: {"bag": "flight_2026-07-01-10-29-23.bag", "start": 42.35},
        5: {"bag": "flight_2026-07-01-10-32-09.bag", "start": 41.89},
        6: {"bag": "flight_2026-07-01-10-35-00.bag", "start": 37.46},
        7: {"bag": "flight_2026-07-01-10-37-51.bag", "start": 48.27},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS)
