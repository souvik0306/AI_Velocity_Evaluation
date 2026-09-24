#!/usr/bin/env python3
"""Evaluate 22 September hover flight bags over configured 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "22nd_Sept_Hover_less_RC"
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
    evaluate_dataset(BAGS_DIR, FLIGHTS, hover_analysis=True)
