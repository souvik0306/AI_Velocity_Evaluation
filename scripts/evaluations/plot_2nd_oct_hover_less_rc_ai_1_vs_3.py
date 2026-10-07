#!/usr/bin/env python3
"""Compare AI flights 1 and 3 from the 2nd October hover-less-RC test.

The script uses the existing cleaned/aligned 20-second velocity windows and
reads RC roll/pitch inputs directly from each ROS bag.  Each output figure has
flight 1 above flight 3 and uses 0--20 seconds from the respective configured
window start so that all traces are directly time-aligned.
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
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "comparison_flights_1_and_3"

WINDOW_DURATION_S = 20.0
FLIGHTS = {
    1: {"bag": "flight_1.bag", "start": 56.24},
    3: {"bag": "flight_3.bag", "start": 37.99},
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
            "Plot 20-second AI flight 1 versus flight 3 comparisons for the "
            "2nd October hover-less-RC dataset."
        )
    )
    parser.add_argument("--bags-dir", type=Path, default=DEFAULT_BAGS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
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
                "Run analyze_2nd_oct_hover_less_rc.py --duration 20s first."
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


def style_axis(axis: Axes, flight: int, ylabel: str) -> None:
    start = float(FLIGHTS[flight]["start"])
    axis.set_title(
        f"AI Flight {flight}  |  source window {start:.2f}--"
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


def save_two_flight_figure(
    out_path: Path,
    title: str,
    ylabel: str,
    plot_flight,
    dpi: int,
    reference_line: Optional[float] = None,
    figsize: Tuple[float, float] = (14.0, 8.5),
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, sharey=True)
    for axis, flight in zip(axes, FLIGHTS):
        plot_flight(axis, flight)
        if reference_line is not None:
            axis.axhline(reference_line, color="0.25", linestyle="--", linewidth=1.5)
        style_axis(axis, flight, ylabel)
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

    velocity: Dict[int, pd.DataFrame] = {}
    rc: Dict[int, pd.DataFrame] = {}
    absolute_window_starts: Dict[int, float] = {}
    for flight, config in FLIGHTS.items():
        bag_path = args.bags_dir / str(config["bag"])
        if not bag_path.is_file():
            raise FileNotFoundError(f"ROS bag not found: {bag_path}")

        velocity[flight] = load_velocity_window(args.results_dir, flight)
        rc[flight] = load_rc_window(bag_path, float(config["start"]))
        absolute_window_starts[flight] = float(rc[flight]["time"].iloc[0]) - float(
            rc[flight]["relative_time"].iloc[0]
        )
        velocity[flight]["relative_time"] = (
            velocity[flight]["time"] - absolute_window_starts[flight]
        )
        velocity[flight] = velocity[flight][
            velocity[flight]["relative_time"].between(0.0, WINDOW_DURATION_S)
        ].copy()

    output_paths = []
    velocity_specs: Tuple[Tuple[str, str, str, str], ...] = (
        ("x", r"$V_x$", "vel_x_ekf2", "vel_x_gt"),
        ("y", r"$V_y$", "vel_y_ekf2", "vel_y_gt"),
        ("horizontal", r"$V_h = \sqrt{V_x^2 + V_y^2}$", "horizontal_ekf2", "horizontal_gt"),
    )
    for file_label, math_label, ekf2_column, gt_column in velocity_specs:
        out_path = args.out_dir / f"ai_flights_1_3_{file_label}_gt_vs_ekf2.png"

        def plot_velocity(axis: Axes, flight: int, ec=ekf2_column, gc=gt_column) -> None:
            data = velocity[flight]
            axis.plot(
                data["relative_time"].to_numpy(), data[gc].to_numpy(), color=GT_COLOR,
                linewidth=DATA_LINE_WIDTH, label=f"GT {math_label}",
            )
            axis.plot(
                data["relative_time"].to_numpy(), data[ec].to_numpy(), color=EKF2_COLOR,
                linewidth=DATA_LINE_WIDTH, alpha=0.9, label=f"EKF2 {math_label}",
            )

        save_two_flight_figure(
            out_path,
            f"2nd October Hover Less RC: GT vs EKF2 {math_label}",
            f"{math_label} (m/s)",
            plot_velocity,
            args.dpi,
            reference_line=0.0,
            figsize=(17.0, 10.5) if file_label == "horizontal" else (14.0, 8.5),
        )
        output_paths.append(out_path)

    errors_path = args.out_dir / "ai_flights_1_3_signed_velocity_errors.png"

    def plot_errors(axis: Axes, flight: int) -> None:
        data = velocity[flight]
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

    save_two_flight_figure(
        errors_path,
        "2nd October Hover Less RC: Signed Velocity Errors",
        "Signed error (m/s)",
        plot_errors,
        args.dpi,
        reference_line=0.0,
    )
    output_paths.append(errors_path)

    rc_path = args.out_dir / "ai_flights_1_3_rc_roll_pitch.png"

    def plot_rc(axis: Axes, flight: int) -> None:
        data = rc[flight]
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

    save_two_flight_figure(
        rc_path,
        "2nd October Hover Less RC: RC Roll/Pitch Commands",
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
