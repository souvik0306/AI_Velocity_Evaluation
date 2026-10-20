#!/usr/bin/env python3
"""Evaluate 15 October high-dynamic bags over Vicon-referenced 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "15th_Oct_High"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 38.85},
        2: {"bag": "flight_2.bag", "start": 39.79},
        3: {"bag": "flight_3.bag", "start": 40.53},
    },
    "RAW": {
        1: {"bag": "flight_1.bag", "start": 44.00},
        2: {"bag": "flight_2.bag", "start": 37.66},
        3: {"bag": "flight_3.bag", "start": 44.20},
    },
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, default_duration="20s")
