#!/usr/bin/env python3
"""Evaluate 2 October Yaw bags over configured 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset




BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "2nd_Oct_Yaw"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 124.54},
        2: {"bag": "flight_2.bag", "start": 44.24},
        3: {"bag": "flight_3.bag", "start": 40.37},
    }
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS
    )
