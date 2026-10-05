#!/usr/bin/env python3
"""Evaluate 2 October no-RC hover bags over configured 20-second windows."""

from pathlib import Path

from evaluation_common import evaluate_dataset


BAGS_DIR = Path(__file__).resolve().parents[2] / "data" / "2nd_Oct_Hover_no_RC"
FLIGHTS = {
    "AI": {
        1: {"bag": "flight_1.bag", "start": 63.96},
        2: {"bag": "flight_2.bag", "start": 36.37},
        3: {"bag": "flight_3.bag", "start": 41.80},
    }
}


if __name__ == "__main__":
    evaluate_dataset(
        BAGS_DIR,
        FLIGHTS,
        hover_analysis=True,
        bags_grouped=False,
    )
