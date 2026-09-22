#!/usr/bin/env python3
"""Evaluate hover flight bags over configured 20-second windows."""

from pathlib import Path
from evaluation_common import evaluate_dataset

BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "9th_Sept_AI_Hover_Rosbags"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 55.15},
        2: {"bag": "flight_2.bag", "start": 54.92},
        3: {"bag": "flight_3.bag", "start": 42.06},
    },
    "RAW": {
        1: {"bag": "flight_2026-07-01-13-01-24.bag", "start": 44.32},
        2: {"bag": "flight_2026-07-01-13-04-24.bag", "start": 39.75},
        3: {"bag": "flight_2026-07-01-13-06-59.bag", "start": 30.71},
    },
}

if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, hover_analysis=True)
