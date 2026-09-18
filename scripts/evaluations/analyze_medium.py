#!/usr/bin/env python3
"""Evaluate medium flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "Medium"
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
    evaluate_dataset(BAGS_DIR, FLIGHTS)
