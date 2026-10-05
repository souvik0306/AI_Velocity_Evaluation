#!/usr/bin/env python3
"""Evaluate 2 October FB bags over configured 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset



BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "2nd_Oct_FB"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 42.77},
        2: {"bag": "flight_2.bag", "start": 34.51},
        3: {"bag": "flight_3.bag", "start": 44.93},
        4: {"bag": "flight_4.bag", "start": 40.11},       
    }
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        hover_analysis=True,
        bags_grouped=False,
    )
