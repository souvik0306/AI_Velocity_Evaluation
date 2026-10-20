#!/usr/bin/env python3
"""Evaluate 15 October low-dynamic bags over Vicon-referenced 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "15th_Oct_Low"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 42.03},
        2: {"bag": "flight_2.bag", "start": 38.28},
        3: {"bag": "flight_3.bag", "start": 55.69},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 38.70},
        2: {"bag": "flight_2.bag", "start": 41.77},
        3: {"bag": "flight_3.bag", "start": 39.39},
        4: {"bag": "flight_4.bag", "start": 42.76},
    },
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, default_duration="20s")
