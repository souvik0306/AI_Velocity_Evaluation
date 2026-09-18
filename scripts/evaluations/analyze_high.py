#!/usr/bin/env python3
"""Evaluate high flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "High"
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
    evaluate_dataset(BAGS_DIR, FLIGHTS)
