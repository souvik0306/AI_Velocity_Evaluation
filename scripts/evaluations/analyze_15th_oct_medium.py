#!/usr/bin/env python3
"""Evaluate 15 October medium-dynamic bags over Vicon-referenced 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "15th_Oct_Medium"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 40.05},
        2: {"bag": "flight_2.bag", "start": 39.13},
        3: {"bag": "flight_3.bag", "start": 40.87},
        4: {"bag": "flight_4.bag", "start": 39.10},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 40.99},
        2: {"bag": "flight_2.bag", "start": 49.39},
        3: {"bag": "flight_3.bag", "start": 45.42},
        4: {"bag": "flight_4.bag", "start": 42.41},
    },
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, default_duration="20s")
