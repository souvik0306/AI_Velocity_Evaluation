#!/usr/bin/env python3
"""Compare selected hover-less-RC flights from 2 October and 22 September.

The script uses the existing cleaned/aligned 20-second velocity windows and
reads RC roll/pitch inputs directly from each ROS bag.  Each output figure has
2 October AI flights 1 and 3 followed by 22 September RAW flight 4.  All plots
use 0--20 seconds from their respective configured window starts so that the
traces are directly time-aligned.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BAGS_DIR = REPO_ROOT / "data" / "2nd_Oct_Hover_less_RC" / "AI"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "2nd_oct_hover_less_rc" / "20s" / "AI"
DEFAULT_RAW_BAG = (
    REPO_ROOT
    / "data"
    / "22nd_Sept_Hover_less_RC"
    / "RAW"
    / "flight_2026-07-01-14-21-56.bag"
)
DEFAULT_RAW_RESULTS_DIR = (
    REPO_ROOT / "results" / "22nd_sept_hover_less_rc" / "20s" / "RAW"
)
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "comparison_ai_1_3_raw_4"

WINDOW_DURATION_S = 20.0
FLIGHTS = {
    "oct_ai_1": {
        "flight": 1,
        "group": "AI",
        "date": "2 October",
        "bag": "flight_1.bag",
        "start": 56.24,
    },
    "oct_ai_3": {
        "flight": 3,
        "group": "AI",
        "date": "2 October",
        "bag": "flight_3.bag",
        "start": 37.99,
    },
    "sept_raw_4": {
        "flight": 4,
        "group": "RAW",
        "date": "22 September",
        "start": 36.97,
    },
}

EST_TOPIC = "/mavros/local_position/velocity_local"
GT_TOPIC = "/vrpn_client_node/AIIMU1/twist"
RC_TOPIC = "/mavros/rc/in"

EKF2_COLOR = "#1f77b4"
GT_COLOR = "#ff7f0e"
ERROR_X_COLOR = "#9467bd"
ERROR_Y_COLOR = "#17becf"
RC_ROLL_COLOR = "#d62728"
RC_PITCH_COLOR = "#1ac938"
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
SUBPLOT_TITLE_FONT_SIZE = 18
FIGURE_TITLE_FONT_SIZE = 20
DATA_LINE_WIDTH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 20-second comparisons for 2 October AI flights 1 and 3 "
            "and 22 September RAW flight 4."
        )
    )
    parser.add_argument("--bags-dir", type=Path, default=DEFAULT_BAGS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--raw-bag", type=Path, default=DEFAULT_RAW_BAG)
    parser.add_argument(
        "--raw-results-dir", type=Path, default=DEFAULT_RAW_RESULTS_DIR
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_velocity_window(results_dir: Path, flight: int) -> pd.DataFrame:
    est_path = results_dir / f"flight_{flight}_vel_est_clean_20s_window.csv"
    gt_path = results_dir / f"flight_{flight}_vel_gt_clean_aligned_20s_window.csv"
    for path in (est_path, gt_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required 20-second result not found: {path}\n"
                "Run the corresponding hover-less-RC analysis with "
                "--duration 20s first."
            )

    required = ["time", "vel_x", "vel_y"]
    est = pd.read_csv(est_path)
    gt = pd.read_csv(gt_path)
    for label, frame, path in (("EKF2", est, est_path), ("GT", gt, gt_path)):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing {missing} in {label} data: {path}")
        frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
        frame.dropna(subset=required, inplace=True)
        frame.sort_values("time", inplace=True)

    merged = pd.merge(
        est[required],
        gt[required],
        on="time",
        how="inner",
        suffixes=("_ekf2", "_gt"),
    )
    if merged.empty:
        raise ValueError(f"No aligned EKF2/GT samples for flight {flight}")

    merged["horizontal_ekf2"] = np.hypot(
        merged["vel_x_ekf2"], merged["vel_y_ekf2"]
    )
    merged["horizontal_gt"] = np.hypot(
        merged["vel_x_gt"], merged["vel_y_gt"]
    )
    # Signed errors follow the evaluation pipeline convention: EKF2 - GT.
    merged["err_x"] = merged["vel_x_ekf2"] - merged["vel_x_gt"]
    merged["err_y"] = merged["vel_y_ekf2"] - merged["vel_y_gt"]
    return merged


def message_header_time(message, topic: str) -> float:
    """Return the ROS message-header time; never use rosbag record time."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        raise ValueError(f"Message on {topic} has no header timestamp")
    seconds = float(stamp.to_sec())
    if seconds <= 0:
        raise ValueError(f"Message on {topic} has invalid header timestamp {seconds}")
    return seconds


def load_rc_window(bag_path: Path, configured_start_s: float) -> pd.DataFrame:
    try:
        import rosbag
    except ImportError as exc:
        raise RuntimeError("The ROS 1 Python package 'rosbag' is required") from exc

    first_velocity_times: Dict[str, float] = {}
    rc_rows = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, _bag_stamp in bag.read_messages(
            topics=[EST_TOPIC, GT_TOPIC, RC_TOPIC]
        ):
            timestamp = message_header_time(message, topic)
            if topic in (EST_TOPIC, GT_TOPIC) and topic not in first_velocity_times:
                first_velocity_times[topic] = timestamp
            elif topic == RC_TOPIC:
                channels = list(message.channels)
                if len(channels) < 2:
                    continue
                rc_rows.append((timestamp, float(channels[0]), float(channels[1])))

    missing_topics = [
        topic for topic in (EST_TOPIC, GT_TOPIC) if topic not in first_velocity_times
    ]
    if missing_topics:
        raise ValueError(f"Missing velocity topics {missing_topics} in {bag_path}")
    if not rc_rows:
        raise ValueError(f"No RC roll/pitch samples on {RC_TOPIC} in {bag_path}")

    flight_time_zero = min(first_velocity_times.values())
    window_start = flight_time_zero + configured_start_s
    window_end = window_start + WINDOW_DURATION_S
    rc = pd.DataFrame(rc_rows, columns=["time", "roll", "pitch"])
    rc = rc[rc["time"].between(window_start, window_end)].copy()
    if rc.empty:
        raise ValueError(
            f"No RC samples in {configured_start_s:.2f}--"
            f"{configured_start_s + WINDOW_DURATION_S:.2f} s for {bag_path}"
        )
    rc["relative_time"] = rc["time"] - window_start
    return rc


def style_axis(axis: Axes, flight_key: str, ylabel: str) -> None:
    config = FLIGHTS[flight_key]
    start = float(config["start"])
    axis.set_title(
        f"{config['date']} {config['group']} Flight {config['flight']}  |  "
        f"source window {start:.2f}--"
        f"{start + WINDOW_DURATION_S:.2f} s",
        fontsize=SUBPLOT_TITLE_FONT_SIZE,
        loc="left",
    )
    axis.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    axis.tick_params(axis="both", which="major", labelsize=TICK_FONT_SIZE)
    axis.set_xlim(0.0, WINDOW_DURATION_S)
    axis.xaxis.set_major_locator(MultipleLocator(2.0))
    axis.xaxis.set_minor_locator(MultipleLocator(1.0))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.grid(True, which="major", alpha=0.32)
    axis.grid(True, which="minor", alpha=0.13)
    axis.axvline(0.0, color="0.25", linewidth=0.8, alpha=0.5)


def save_three_flight_figure(
    out_path: Path,
    title: str,
    ylabel: str,
    plot_flight,
    dpi: int,
    reference_line: Optional[float] = None,
    figsize: Tuple[float, float] = (14.0, 12.0),
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, sharey=True)
    for axis, flight_key in zip(axes, FLIGHTS):
        plot_flight(axis, flight_key)
        if reference_line is not None:
            axis.axhline(reference_line, color="0.25", linestyle="--", linewidth=1.5)
        style_axis(axis, flight_key, ylabel)
        axis.legend(
            loc="best", ncol=2, framealpha=0.92, fontsize=LEGEND_FONT_SIZE
        )

    axes[-1].set_xlabel(
        "Time from respective evaluation-window start (s)",
        fontsize=LABEL_FONT_SIZE,
    )
    fig.suptitle(title, fontsize=FIGURE_TITLE_FONT_SIZE, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    velocity: Dict[str, pd.DataFrame] = {}
    rc: Dict[str, pd.DataFrame] = {}
    absolute_window_starts: Dict[str, float] = {}
    for flight_key, config in FLIGHTS.items():
        flight = int(config["flight"])
        if config["group"] == "RAW":
            bag_path = args.raw_bag
            results_dir = args.raw_results_dir
        else:
            bag_path = args.bags_dir / str(config["bag"])
            results_dir = args.results_dir
        if not bag_path.is_file():
            raise FileNotFoundError(f"ROS bag not found: {bag_path}")

        velocity[flight_key] = load_velocity_window(results_dir, flight)
        rc[flight_key] = load_rc_window(bag_path, float(config["start"]))
        absolute_window_starts[flight_key] = float(
            rc[flight_key]["time"].iloc[0]
        ) - float(
            rc[flight_key]["relative_time"].iloc[0]
        )
        velocity[flight_key]["relative_time"] = (
            velocity[flight_key]["time"] - absolute_window_starts[flight_key]
        )
        velocity[flight_key] = velocity[flight_key][
            velocity[flight_key]["relative_time"].between(0.0, WINDOW_DURATION_S)
        ].copy()

    output_paths = []
    velocity_specs: Tuple[Tuple[str, str, str, str], ...] = (
        ("x", r"$V_x$", "vel_x_ekf2", "vel_x_gt"),
        ("y", r"$V_y$", "vel_y_ekf2", "vel_y_gt"),
        ("horizontal", r"$V_h = \sqrt{V_x^2 + V_y^2}$", "horizontal_ekf2", "horizontal_gt"),
    )
    for file_label, math_label, ekf2_column, gt_column in velocity_specs:
        out_path = args.out_dir / f"ai_1_3_raw_4_{file_label}_gt_vs_ekf2.png"

        def plot_velocity(axis: Axes, flight_key: str, ec=ekf2_column, gc=gt_column) -> None:
            data = velocity[flight_key]
            axis.plot(
                data["relative_time"].to_numpy(), data[gc].to_numpy(), color=GT_COLOR,
                linewidth=DATA_LINE_WIDTH, label=f"GT {math_label}",
            )
            axis.plot(
                data["relative_time"].to_numpy(), data[ec].to_numpy(), color=EKF2_COLOR,
                linewidth=DATA_LINE_WIDTH, alpha=0.9, label=f"EKF2 {math_label}",
            )

        save_three_flight_figure(
            out_path,
            f"Hover Less RC Comparison: GT vs EKF2 {math_label}",
            f"{math_label} (m/s)",
            plot_velocity,
            args.dpi,
            reference_line=0.0,
            figsize=(17.0, 14.0) if file_label == "horizontal" else (14.0, 12.0),
        )
        output_paths.append(out_path)

    errors_path = args.out_dir / "ai_1_3_raw_4_signed_velocity_errors.png"

    def plot_errors(axis: Axes, flight_key: str) -> None:
        data = velocity[flight_key]
        axis.plot(
            data["relative_time"].to_numpy(), data["err_x"].to_numpy(),
            color=ERROR_X_COLOR,
            linewidth=DATA_LINE_WIDTH, label=r"$e_x = V_{x,EKF2} - V_{x,GT}$",
        )
        axis.plot(
            data["relative_time"].to_numpy(), data["err_y"].to_numpy(),
            color=ERROR_Y_COLOR,
            linewidth=DATA_LINE_WIDTH, label=r"$e_y = V_{y,EKF2} - V_{y,GT}$",
        )

    save_three_flight_figure(
        errors_path,
        "Hover Less RC Comparison: Signed Velocity Errors",
        "Signed error (m/s)",
        plot_errors,
        args.dpi,
        reference_line=0.0,
    )
    output_paths.append(errors_path)

    rc_path = args.out_dir / "ai_1_3_raw_4_rc_roll_pitch.png"

    def plot_rc(axis: Axes, flight_key: str) -> None:
        data = rc[flight_key]
        axis.plot(
            data["relative_time"].to_numpy(), data["roll"].to_numpy(),
            color=RC_ROLL_COLOR,
            linewidth=DATA_LINE_WIDTH, label="RC roll (channels[0])",
        )
        axis.plot(
            data["relative_time"].to_numpy(), data["pitch"].to_numpy(),
            color=RC_PITCH_COLOR,
            linewidth=DATA_LINE_WIDTH, label="RC pitch (channels[1])",
        )

    save_three_flight_figure(
        rc_path,
        "Hover Less RC Comparison: RC Roll/Pitch Commands",
        "RC input (PWM)",
        plot_rc,
        args.dpi,
        reference_line=1500.0,
    )
    output_paths.append(rc_path)

    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
