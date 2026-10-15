#!/usr/bin/env python3
"""Evaluate 14 October high-dynamic bags over Vicon-referenced 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "14th_Oct_High"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 51.08},
        2: {"bag": "flight_2.bag", "start": 41.44},
        3: {"bag": "flight_3.bag", "start": 38.30},
        4: {"bag": "flight_4.bag", "start": 39.35},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 38.97},
        2: {"bag": "flight_2.bag", "start": 43.61},
        3: {"bag": "flight_3.bag", "start": 34.50},
    },
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, default_duration="20s")
