#!/usr/bin/env python3
"""Evaluate low flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "Low-Dynamic"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 44.24},
        2: {"bag": "flight_2.bag", "start": 32.99},
        3: {"bag": "flight_3.bag", "start": 35.95},
        4: {"bag": "flight_4.bag", "start": 41.47},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-13-16-49.bag", "start": 48.51},
        2: {"bag": "flight_2026-07-01-13-19-00.bag", "start": 33.29},
        3: {"bag": "flight_2026-07-01-13-20-51.bag", "start": 34.77},
        4: {"bag": "flight_2026-07-01-13-22-50.bag", "start": 39.63},
        5: {"bag": "flight_2026-07-01-13-25-41.bag", "start": 35.16},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS)
