#!/usr/bin/env python3
"""Evaluate super high flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "Super-High"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-07-01-12-45-01.bag", "start": 41.14},
        2: {"bag": "flight_2026-07-01-12-47-25.bag", "start": 40.24},
        3: {"bag": "flight_2026-07-01-12-49-22.bag", "start": 37.5},
        4: {"bag": "flight_2026-07-01-12-51-49.bag", "start": 38.23},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS)
