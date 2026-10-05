#!/usr/bin/env python3
"""Evaluate 2nd October hover flight bags over configured 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset



BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "2nd_Oct_Hover_less_RC"
FLIGHTS = {
    "AI": {
    1: {"bag": "flight_1.bag", "start": 56.24, "end": 76.24},
    2: {"bag": "flight_2.bag", "start": 39.81, "end": 59.81},
    3: {"bag": "flight_3.bag", "start": 37.99, "end": 57.99},
    }
}


if __name__ == "__main__":
    evaluate_dataset(BAGS_DIR, FLIGHTS, hover_analysis=True)
