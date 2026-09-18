#!/usr/bin/env python3
"""Evaluate yaw flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "10th_Sept_AI_Yaw_Rosbags"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_2026-07-01-12-57-53.bag", "start": 44.93},
        2: {"bag": "flight_2026-07-01-13-00-24.bag", "start": 38.07},
        3: {"bag": "flight_2026-07-01-13-02-26.bag", "start": 39.44},
        4: {"bag": "flight_2026-07-01-13-04-27.bag", "start": 36.31},
        5: {"bag": "flight_2026-07-01-13-06-42.bag", "start": 31.78},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-13-55-52.bag", "start": 38.05},
        2: {"bag": "flight_2026-07-01-13-58-03.bag", "start": 34.77},
        3: {"bag": "flight_2026-07-01-14-00-00.bag", "start": 38.71},
        4: {"bag": "flight_2026-07-01-14-01-56.bag", "start": 32.82},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS)
