#!/usr/bin/env python3
"""Compare AI flights 2 and 4 with RAW flight 3 from the 7 October yaw test.

The script loads the existing cleaned/aligned 20-second velocity windows and
reads the yaw command (``channels[3]``) directly from each ROS bag.  Every
figure uses 0--20 seconds from the respective configured evaluation-window
start so the three runs are directly time-aligned.
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
DEFAULT_AI_BAGS_DIR = REPO_ROOT / "data" / "7th_Oct_Yaw" / "AI"
DEFAULT_RAW_BAGS_DIR = REPO_ROOT / "data" / "7th_Oct_Yaw" / "RAW"
DEFAULT_AI_RESULTS_DIR = REPO_ROOT / "results" / "7th_oct_yaw" / "20s" / "AI"
DEFAULT_RAW_RESULTS_DIR = REPO_ROOT / "results" / "7th_oct_yaw" / "20s" / "RAW"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "7th_oct_yaw"
    / "20s"
    / "comparison_ai_2_4_and_raw_3"
)

WINDOW_DURATION_S = 20.0
RUNS = {
    "ai_2": {
        "dataset": "AI",
        "flight": 2,
        "bag": "flight_2.bag",
        "start": 43.85,
    },
    "ai_4": {
        "dataset": "AI",
        "flight": 4,
        "bag": "flight_4.bag",
        "start": 36.86,
    },
    "raw_3": {
        "dataset": "RAW",
        "flight": 3,
        "bag": "flight_3.bag",
        "start": 36.16,
    },
}

EST_TOPIC = "/mavros/local_position/velocity_local"
GT_TOPIC = "/vrpn_client_node/AIIMU1/twist"
RC_TOPIC = "/mavros/rc/in"
YAW_CHANNEL_INDEX = 3

EKF2_COLOR = "#1f77b4"
GT_COLOR = "#ff7f0e"
ERROR_X_COLOR = "#9467bd"
ERROR_Y_COLOR = "#17becf"
RC_YAW_COLOR = "#d62728"
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
SUBPLOT_TITLE_FONT_SIZE = 18
FIGURE_TITLE_FONT_SIZE = 20
DATA_LINE_WIDTH = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 20-second comparisons for 7 October yaw AI flights 2 and 4 "
            "and RAW flight 3."
        )
    )
    parser.add_argument("--bags-dir", type=Path, default=DEFAULT_AI_BAGS_DIR)
    parser.add_argument("--raw-bags-dir", type=Path, default=DEFAULT_RAW_BAGS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_AI_RESULTS_DIR)
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
                "Run analyze_7th_oct_yaw.py --duration 20s first."
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
    merged["err_x"] = merged["vel_x_ekf2"] - merged["vel_x_gt"]
    merged["err_y"] = merged["vel_y_ekf2"] - merged["vel_y_gt"]
    return merged


def message_header_time(message, topic: str) -> float:
    """Return the ROS message-header time rather than rosbag record time."""
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        raise ValueError(f"Message on {topic} has no header timestamp")
    seconds = float(stamp.to_sec())
    if seconds <= 0:
        raise ValueError(f"Message on {topic} has invalid header timestamp {seconds}")
    return seconds


def load_yaw_command_window(
    bag_path: Path, configured_start_s: float
) -> pd.DataFrame:
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
                if len(channels) <= YAW_CHANNEL_INDEX:
                    continue
                rc_rows.append((timestamp, float(channels[YAW_CHANNEL_INDEX])))

    missing_topics = [
        topic for topic in (EST_TOPIC, GT_TOPIC) if topic not in first_velocity_times
    ]
    if missing_topics:
        raise ValueError(f"Missing velocity topics {missing_topics} in {bag_path}")
    if not rc_rows:
        raise ValueError(
            f"No RC yaw samples from channels[{YAW_CHANNEL_INDEX}] in {bag_path}"
        )

    flight_time_zero = min(first_velocity_times.values())
    window_start = flight_time_zero + configured_start_s
    window_end = window_start + WINDOW_DURATION_S
    rc = pd.DataFrame(rc_rows, columns=["time", "yaw"])
    rc = rc[rc["time"].between(window_start, window_end)].copy()
    if rc.empty:
        raise ValueError(
            f"No RC yaw samples in {configured_start_s:.2f}--"
            f"{configured_start_s + WINDOW_DURATION_S:.2f} s for {bag_path}"
        )
    rc["relative_time"] = rc["time"] - window_start
    return rc


def style_axis(axis: Axes, run: str, ylabel: str) -> None:
    config = RUNS[run]
    dataset = str(config["dataset"])
    flight = int(config["flight"])
    start = float(config["start"])
    axis.set_title(
        f"{dataset} Flight {flight}  |  source window {start:.2f}--"
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


def save_comparison_figure(
    out_path: Path,
    title: str,
    ylabel: str,
    plot_flight,
    dpi: int,
    reference_line: Optional[float] = None,
    figsize: Tuple[float, float] = (14.0, 8.5),
) -> None:
    fig, axes = plt.subplots(
        len(RUNS), 1, figsize=figsize, sharex=True, sharey=True
    )
    for axis, run in zip(axes, RUNS):
        plot_flight(axis, run)
        if reference_line is not None:
            axis.axhline(
                reference_line, color="0.25", linestyle="--", linewidth=1.5
            )
        style_axis(axis, run, ylabel)
        axis.legend(
            loc="best", ncol=2, framealpha=0.92, fontsize=LEGEND_FONT_SIZE
        )

    axes[-1].set_xlabel(
        "Time from respective evaluation-window start (s)",
        fontsize=LABEL_FONT_SIZE,
    )
    fig.suptitle(
        title,
        fontsize=FIGURE_TITLE_FONT_SIZE,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    velocity: Dict[str, pd.DataFrame] = {}
    yaw_command: Dict[str, pd.DataFrame] = {}
    for run, config in RUNS.items():
        dataset = str(config["dataset"])
        flight = int(config["flight"])
        bags_dir = args.bags_dir if dataset == "AI" else args.raw_bags_dir
        results_dir = args.results_dir if dataset == "AI" else args.raw_results_dir
        bag_path = bags_dir / str(config["bag"])
        if not bag_path.is_file():
            raise FileNotFoundError(f"ROS bag not found: {bag_path}")

        velocity[run] = load_velocity_window(results_dir, flight)
        yaw_command[run] = load_yaw_command_window(
            bag_path, float(config["start"])
        )
        absolute_window_start = float(yaw_command[run]["time"].iloc[0]) - float(
            yaw_command[run]["relative_time"].iloc[0]
        )
        velocity[run]["relative_time"] = (
            velocity[run]["time"] - absolute_window_start
        )
        velocity[run] = velocity[run][
            velocity[run]["relative_time"].between(0.0, WINDOW_DURATION_S)
        ].copy()

    output_paths = []
    velocity_specs: Tuple[Tuple[str, str, str, str], ...] = (
        ("x", r"$V_x$", "vel_x_ekf2", "vel_x_gt"),
        ("y", r"$V_y$", "vel_y_ekf2", "vel_y_gt"),
        (
            "horizontal",
            r"$V_h = \sqrt{V_x^2 + V_y^2}$",
            "horizontal_ekf2",
            "horizontal_gt",
        ),
    )
    for file_label, math_label, ekf2_column, gt_column in velocity_specs:
        out_path = args.out_dir / f"ai_2_4_raw_3_{file_label}_gt_vs_ekf2.png"

        def plot_velocity(
            axis: Axes, run: str, ec=ekf2_column, gc=gt_column
        ) -> None:
            data = velocity[run]
            axis.plot(
                data["relative_time"].to_numpy(),
                data[gc].to_numpy(),
                color=GT_COLOR,
                linewidth=DATA_LINE_WIDTH,
                label=f"GT {math_label}",
            )
            axis.plot(
                data["relative_time"].to_numpy(),
                data[ec].to_numpy(),
                color=EKF2_COLOR,
                linewidth=DATA_LINE_WIDTH,
                alpha=0.9,
                label=f"EKF2 {math_label}",
            )

        save_comparison_figure(
            out_path,
            f"7 October Yaw: GT vs EKF2 {math_label}",
            f"{math_label} (m/s)",
            plot_velocity,
            args.dpi,
            reference_line=0.0,
            figsize=(
                (17.0, 14.0)
                if file_label == "horizontal"
                else (14.0, 12.0)
            ),
        )
        output_paths.append(out_path)

    errors_path = args.out_dir / "ai_2_4_raw_3_signed_velocity_errors.png"

    def plot_errors(axis: Axes, run: str) -> None:
        data = velocity[run]
        axis.plot(
            data["relative_time"].to_numpy(),
            data["err_x"].to_numpy(),
            color=ERROR_X_COLOR,
            linewidth=DATA_LINE_WIDTH,
            label=r"$e_x = V_{x,EKF2} - V_{x,GT}$",
        )
        axis.plot(
            data["relative_time"].to_numpy(),
            data["err_y"].to_numpy(),
            color=ERROR_Y_COLOR,
            linewidth=DATA_LINE_WIDTH,
            label=r"$e_y = V_{y,EKF2} - V_{y,GT}$",
        )

    save_comparison_figure(
        errors_path,
        "7 October Yaw: Signed Velocity Errors",
        "Signed error (m/s)",
        plot_errors,
        args.dpi,
        reference_line=0.0,
        figsize=(14.0, 12.0),
    )
    output_paths.append(errors_path)

    yaw_path = args.out_dir / "ai_2_4_raw_3_rc_yaw_command.png"

    def plot_yaw_command(axis: Axes, run: str) -> None:
        data = yaw_command[run]
        axis.plot(
            data["relative_time"].to_numpy(),
            data["yaw"].to_numpy(),
            color=RC_YAW_COLOR,
            linewidth=DATA_LINE_WIDTH,
            label=f"RC yaw (channels[{YAW_CHANNEL_INDEX}])",
        )

    save_comparison_figure(
        yaw_path,
        "7 October Yaw: RC Yaw Command",
        "RC input (PWM)",
        plot_yaw_command,
        args.dpi,
        reference_line=1500.0,
        figsize=(14.0, 12.0),
    )
    output_paths.append(yaw_path)

    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
