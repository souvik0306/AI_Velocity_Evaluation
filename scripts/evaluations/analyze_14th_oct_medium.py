#!/usr/bin/env python3
"""Evaluate 14 October medium-dynamic bags over Vicon-referenced 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "14th_Oct_Medium"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 39.65},
        2: {"bag": "flight_2.bag", "start": 45.39},
        3: {"bag": "flight_3.bag", "start": 45.11},
        4: {"bag": "flight_4.bag", "start": 41.84},
        5: {"bag": "flight_5.bag", "start": 37.04},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 38.71},
        2: {"bag": "flight_2.bag", "start": 36.80},
        3: {"bag": "flight_3.bag", "start": 36.60},
        4: {"bag": "flight_4.bag", "start": 36.99},
    },
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, default_duration="20s")
