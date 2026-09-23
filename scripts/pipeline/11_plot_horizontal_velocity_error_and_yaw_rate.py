#!/usr/bin/env python3
"""Plot RMSE-window horizontal velocity error against Vicon yaw rate.

The velocity error is read from the bias-error CSV produced by the existing
RMSE pipeline. Its raw ``err_x`` and ``err_y`` columns are the exact aligned
samples used by that calculation; the bias-corrected columns are intentionally
not used here. Vicon orientation is read independently from the source bag and
clipped to the first and last RMSE sample timestamps. Direct finite differences
remain the diagnostic metric, while a uniformly resampled 11-point cubic
Savitzky-Golay derivative is used only for the time plot.
"""

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_CSV = REPO_ROOT / "results" / "yaw" / "20s" / "yaw_20s_summary_by_flight.csv"
DEFAULT_BAGS_DIR = REPO_ROOT / "data" / "10th_Sept_AI_Yaw_Rosbags"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "yaw" / "20s" / "yaw_rate_analysis"
DEFAULT_POSE_TOPIC = "/vrpn_client_node/AIIMU1/pose"
REQUIRED_SUMMARY_COLUMNS = ("group", "bag", "flight", "bias_errors_file")
REQUIRED_ERROR_COLUMNS = ("time", "err_x", "err_y")
QUATERNION_COLUMNS = ("x", "y", "z", "w")
SAVGOL_WINDOW_LENGTH = 11
SAVGOL_POLYORDER = 3


def _require_columns(df: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
	missing = [column for column in columns if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {source}")


def load_horizontal_velocity_error(path: Path) -> Tuple[np.ndarray, np.ndarray]:
	"""Use the uncorrected, aligned errors already consumed by RMSE."""
	df = pd.read_csv(path)
	_require_columns(df, REQUIRED_ERROR_COLUMNS, path)
	for column in REQUIRED_ERROR_COLUMNS:
		df[column] = pd.to_numeric(df[column], errors="coerce")
	if df["time"].isna().any():
		raise ValueError(f"Invalid velocity-error timestamps in {path}")
	if df.empty:
		raise ValueError(f"No velocity-error samples in {path}")

	time = df["time"].to_numpy(dtype=float)
	if not np.all(np.diff(time) > 0.0):
		raise ValueError(f"Velocity-error timestamps are not strictly increasing in {path}")
	error = np.hypot(
		df["err_x"].to_numpy(dtype=float),
		df["err_y"].to_numpy(dtype=float),
	)
	return time, error


def extract_vicon_quaternions(bag_path: Path, pose_topic: str) -> Tuple[pd.DataFrame, float]:
	"""Read pose quaternions with their corresponding Vicon header stamps."""
	rows = []
	with rosbag.Bag(str(bag_path), "r") as bag:
		flight_time_zero = float(bag.get_start_time())
		for _, message, _ in bag.read_messages(topics=[pose_topic]):
			orientation = message.pose.orientation
			rows.append({
				"time": float(message.header.stamp.to_sec()),
				"x": float(orientation.x),
				"y": float(orientation.y),
				"z": float(orientation.z),
				"w": float(orientation.w),
			})
	if not rows:
		raise ValueError(f"No Vicon pose messages found on {pose_topic} in {bag_path}")
	return pd.DataFrame(rows), flight_time_zero


def calculate_absolute_yaw_rate(
	quaternions: pd.DataFrame,
	evaluation_start: float,
	evaluation_end: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
	"""Calculate direct metric yaw rate and a separate SG plotting signal."""
	df = quaternions.copy()
	for column in ("time", *QUATERNION_COLUMNS):
		df[column] = pd.to_numeric(df[column], errors="coerce")

	in_interval = df["time"].between(evaluation_start, evaluation_end, inclusive="both")
	df = df.loc[in_interval, ["time", *QUATERNION_COLUMNS]].copy()
	interval_samples = len(df)
	finite = np.isfinite(df[["time", *QUATERNION_COLUMNS]].to_numpy(dtype=float)).all(axis=1)
	nonfinite_samples = int((~finite).sum())
	df = df.loc[finite].reset_index(drop=True)

	quaternion = df[list(QUATERNION_COLUMNS)].to_numpy(dtype=float)
	norm = np.linalg.norm(quaternion, axis=1)
	valid_norm = np.isfinite(norm) & (norm > 0.0)
	invalid_quaternions = int((~valid_norm).sum())
	df = df.loc[valid_norm].reset_index(drop=True)
	quaternion = quaternion[valid_norm] / norm[valid_norm, np.newaxis]

	# Reject timestamp bursts with the same half-median interval rule used by the
	# earlier yaw metric calculation. Comparing against the last retained sample
	# also rejects duplicate and backward timestamps.
	time = df["time"].to_numpy(dtype=float)
	positive_intervals = np.diff(time)
	positive_intervals = positive_intervals[positive_intervals > 0.0]
	if positive_intervals.size == 0:
		raise ValueError("No valid positive Vicon timestamp intervals")
	median_dt = float(np.median(positive_intervals))
	minimum_interval = 0.5 * median_dt
	minimum_positive_interval = float(np.min(positive_intervals))
	samples_before_timestamp_filtering = int(time.size)

	keep = np.zeros(time.size, dtype=bool)
	keep[0] = True
	last_kept_index = 0
	for index in range(1, time.size):
		if time[index] - time[last_kept_index] >= minimum_interval:
			keep[index] = True
			last_kept_index = index
	timestamp_samples_removed = int((~keep).sum())
	time = time[keep]
	quaternion = quaternion[keep]

	if time.size < 2:
		raise ValueError(
			"Fewer than two valid, monotonically increasing Vicon pose samples "
			f"inside evaluation interval [{evaluation_start:.9f}, {evaluation_end:.9f}]"
		)
	monotonic = bool(np.all(np.diff(time) > 0.0))
	if not monotonic:
		raise AssertionError("Vicon timestamps are not strictly increasing after cleanup")

	x, y, z, w = quaternion.T
	yaw_rad = np.arctan2(
		2.0 * (w * z + x * y),
		1.0 - 2.0 * (y * y + z * z),
	)
	yaw_unwrapped = np.unwrap(yaw_rad)
	delta_time = np.diff(time)
	yaw_rate_time = (time[1:] + time[:-1]) / 2.0
	absolute_yaw_rate_deg_s = np.abs(np.degrees(np.diff(yaw_unwrapped) / delta_time))
	maximum_absolute_yaw_rate = float(np.max(absolute_yaw_rate_deg_s))
	p99_absolute_yaw_rate = float(np.percentile(absolute_yaw_rate_deg_s, 99))

	# Savitzky-Golay assumes equally spaced samples. Resample only the plotting
	# signal; the validated direct finite-difference metric above stays unchanged.
	uniform_dt = float(np.median(delta_time))
	uniform_sample_count = int((time[-1] - time[0]) / uniform_dt) + 1
	uniform_time = time[0] + np.arange(uniform_sample_count, dtype=float) * uniform_dt
	uniform_yaw = np.interp(uniform_time, time, yaw_unwrapped)
	if uniform_yaw.size < SAVGOL_WINDOW_LENGTH:
		raise ValueError(
			f"Need at least {SAVGOL_WINDOW_LENGTH} uniformly resampled yaw samples "
			"for the Savitzky-Golay plotting derivative"
		)
	yaw_rate_sg_rad_s = savgol_filter(
		uniform_yaw,
		window_length=SAVGOL_WINDOW_LENGTH,
		polyorder=SAVGOL_POLYORDER,
		deriv=1,
		delta=uniform_dt,
		mode="interp",
	)
	absolute_yaw_rate_sg_deg_s = np.abs(np.degrees(yaw_rate_sg_rad_s))

	diagnostics = {
		"pose_samples_in_interval": interval_samples,
		"nonfinite_pose_samples_removed": nonfinite_samples,
		"zero_norm_quaternions_removed": invalid_quaternions,
		"median_vicon_dt_s": median_dt,
		"minimum_accepted_dt_s": minimum_interval,
		"minimum_positive_dt_before_filtering_s": minimum_positive_interval,
		"samples_before_timestamp_filtering": samples_before_timestamp_filtering,
		"samples_after_timestamp_filtering": int(time.size),
		"timestamp_samples_removed": timestamp_samples_removed,
		"yaw_timestamps_monotonically_increasing": monotonic,
		"valid_yaw_samples": int(time.size),
		"yaw_rate_samples": int(yaw_rate_time.size),
		"maximum_absolute_yaw_rate_deg_s": maximum_absolute_yaw_rate,
		"p99_absolute_yaw_rate_deg_s": p99_absolute_yaw_rate,
		"p99_yaw_rate_method": "direct_finite_difference",
		"savgol_uniform_dt_s": uniform_dt,
		"savgol_window_length": SAVGOL_WINDOW_LENGTH,
		"savgol_polyorder": SAVGOL_POLYORDER,
		"savgol_yaw_rate_samples": int(uniform_time.size),
		"maximum_absolute_savgol_yaw_rate_deg_s": float(np.max(absolute_yaw_rate_sg_deg_s)),
	}
	return (
		yaw_rate_time,
		absolute_yaw_rate_deg_s,
		uniform_time,
		absolute_yaw_rate_sg_deg_s,
		diagnostics,
	)


def plot_signals(
	e_xy_time: np.ndarray,
	e_xy: np.ndarray,
	yaw_rate_time: np.ndarray,
	absolute_yaw_rate_deg_s: np.ndarray,
	yaw_rate_sg_time: np.ndarray,
	absolute_yaw_rate_sg_deg_s: np.ndarray,
	p99_yaw_rate_deg_s: float,
	evaluation_start: float,
	evaluation_end: float,
	title: str,
	output_path: Path,
	dpi: int,
	show: bool,
) -> None:
	"""Plot velocity error and both yaw-rate forms with one time origin."""
	fig, (ax_error, ax_yaw) = plt.subplots(
		2,
		1,
		figsize=(16, 11),
		sharex=True,
		constrained_layout=True,
	)
	ax_error.plot(e_xy_time - evaluation_start, e_xy, color="#2ca02c", linewidth=4.0)
	ax_error.set_ylabel("Horizontal velocity error (m/s)", fontsize=19)
	ax_error.set_title(title, fontsize=22)

	ax_yaw.plot(
		yaw_rate_time - evaluation_start,
		absolute_yaw_rate_deg_s,
		color="#8ecae6",
		linewidth=4.0,
		alpha=0.25,
		label="Raw yaw rate",
	)
	ax_yaw.plot(
		yaw_rate_sg_time - evaluation_start,
		absolute_yaw_rate_sg_deg_s,
		color="#1f77b4",
		linewidth=4.0,
		label="SG filtered yaw rate",
	)
	ax_yaw.axhline(
		p99_yaw_rate_deg_s,
		color="#333333",
		linestyle="--",
		linewidth=4.0,
		alpha=0.8,
		label=f"P99 = {p99_yaw_rate_deg_s:.1f} deg/s",
	)
	ax_yaw.set_ylabel("Absolute Vicon yaw rate (deg/s)", fontsize=19)
	ax_yaw.set_xlabel("Time (s)", fontsize=19)
	ax_yaw.legend(loc="best", fontsize=14)

	for axis in (ax_error, ax_yaw):
		axis.grid(True, which="major", alpha=0.7)
		axis.grid(True, which="minor", alpha=0.25)
		axis.xaxis.set_major_locator(MaxNLocator(nbins=12))
		axis.xaxis.set_minor_locator(AutoMinorLocator(2))
		axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
		axis.yaxis.set_minor_locator(AutoMinorLocator(2))
		axis.tick_params(axis="both", labelsize=17)
	ax_yaw.set_xlim(0.0, evaluation_end - evaluation_start)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	if show:
		plt.show()
	else:
		plt.close(fig)


def _resolve_bias_errors_path(configured_path: str, summary_path: Path, group: str) -> Path:
	path = Path(configured_path)
	if path.exists():
		return path
	fallback = summary_path.parent / group / path.name
	if fallback.exists():
		return fallback
	raise FileNotFoundError(f"RMSE bias-error CSV not found: {path} (or {fallback})")


def analyze_flight(
	row,
	summary_path: Path,
	bags_dir: Path,
	output_dir: Path,
	pose_topic: str,
	dpi: int,
	show: bool,
) -> dict:
	group = str(row.group)
	flight = str(row.flight)
	bag_path = bags_dir / group / str(row.bag)
	if not bag_path.exists():
		raise FileNotFoundError(f"ROS bag not found: {bag_path}")
	bias_path = _resolve_bias_errors_path(str(row.bias_errors_file), summary_path, group)

	e_xy_time, e_xy = load_horizontal_velocity_error(bias_path)
	evaluation_start = float(e_xy_time[0])
	evaluation_end = float(e_xy_time[-1])
	quaternions, flight_time_zero = extract_vicon_quaternions(bag_path, pose_topic)
	(
		yaw_rate_time,
		yaw_rate,
		yaw_rate_sg_time,
		yaw_rate_sg,
		diagnostics,
	) = calculate_absolute_yaw_rate(
		quaternions,
		evaluation_start,
		evaluation_end,
	)

	flight_dir = output_dir / group / flight
	flight_dir.mkdir(parents=True, exist_ok=True)
	prefix = f"{group.lower()}_{flight}"
	error_csv = flight_dir / f"{prefix}_horizontal_velocity_error.csv"
	yaw_csv = flight_dir / f"{prefix}_absolute_yaw_rate.csv"
	yaw_sg_csv = flight_dir / f"{prefix}_absolute_yaw_rate_sg.csv"
	plot_path = flight_dir / f"{prefix}_horizontal_velocity_error_and_yaw_rate.png"
	diagnostics_path = flight_dir / f"{prefix}_diagnostics.txt"

	pd.DataFrame({"e_xy_time": e_xy_time, "e_xy": e_xy}).to_csv(error_csv, index=False)
	pd.DataFrame({
		"yaw_rate_time": yaw_rate_time,
		"absolute_yaw_rate_deg_s": yaw_rate,
	}).to_csv(yaw_csv, index=False)
	pd.DataFrame({
		"yaw_rate_sg_time": yaw_rate_sg_time,
		"absolute_yaw_rate_sg_deg_s": yaw_rate_sg,
	}).to_csv(yaw_sg_csv, index=False)
	plot_signals(
		e_xy_time,
		e_xy,
		yaw_rate_time,
		yaw_rate,
		yaw_rate_sg_time,
		yaw_rate_sg,
		diagnostics["p99_absolute_yaw_rate_deg_s"],
		evaluation_start,
		evaluation_end,
		f"{group} {flight}: Horizontal Velocity Error and Vicon Yaw Rate",
		plot_path,
		dpi,
		show,
	)

	diagnostics.update({
		"group": group,
		"flight": flight,
		"bag": str(row.bag),
		"evaluation_start_timestamp_s": evaluation_start,
		"evaluation_end_timestamp_s": evaluation_end,
		"evaluation_start_flight_time_s": evaluation_start - flight_time_zero,
		"evaluation_end_flight_time_s": evaluation_end - flight_time_zero,
		"velocity_error_samples": int(e_xy_time.size),
		"horizontal_velocity_error_csv": str(error_csv),
		"direct_absolute_yaw_rate_csv": str(yaw_csv),
		"savgol_absolute_yaw_rate_csv": str(yaw_sg_csv),
		"plot_file": str(plot_path),
	})
	lines = [f"{key}: {value}" for key, value in diagnostics.items()]
	diagnostics_path.write_text("\n".join(lines) + "\n")

	print(f"{group} {flight}")
	print(
		f"  Evaluation interval: {evaluation_start:.9f} -> {evaluation_end:.9f} s "
		f"(flight time {evaluation_start - flight_time_zero:.6f} -> "
		f"{evaluation_end - flight_time_zero:.6f} s)"
	)
	print(f"  Median Vicon dt: {diagnostics['median_vicon_dt_s']:.9f} s")
	print(f"  Minimum accepted dt: {diagnostics['minimum_accepted_dt_s']:.9f} s")
	print(
		"  Minimum dt before filtering: "
		f"{diagnostics['minimum_positive_dt_before_filtering_s']:.9f} s"
	)
	print(
		"  Samples before timestamp filtering: "
		f"{diagnostics['samples_before_timestamp_filtering']}"
	)
	print(
		"  Samples after timestamp filtering: "
		f"{diagnostics['samples_after_timestamp_filtering']}"
	)
	print(f"  Yaw timestamps monotonically increasing: {diagnostics['yaw_timestamps_monotonically_increasing']}")
	print(f"  Velocity error samples: {e_xy_time.size}")
	print(f"  Direct finite-difference yaw rate samples: {yaw_rate_time.size}")
	print(f"  SG plotting yaw rate samples: {yaw_rate_sg_time.size}")
	print(
		"  Maximum absolute yaw rate: "
		f"{diagnostics['maximum_absolute_yaw_rate_deg_s']:.6f} deg/s"
	)
	print(f"  P99 absolute yaw rate: {diagnostics['p99_absolute_yaw_rate_deg_s']:.6f} deg/s")
	print(
		"  Maximum absolute SG plotting yaw rate: "
		f"{diagnostics['maximum_absolute_savgol_yaw_rate_deg_s']:.6f} deg/s"
	)
	print(f"  Saved plot: {plot_path}")
	return diagnostics


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Plot horizontal velocity error and absolute Vicon yaw rate for every "
			"flight in an existing RMSE summary."
		)
	)
	parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY_CSV)
	parser.add_argument("--bags_dir", type=Path, default=DEFAULT_BAGS_DIR)
	parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	parser.add_argument("--pose_topic", default=DEFAULT_POSE_TOPIC)
	parser.add_argument("--group", choices=("all", "AI", "RAW"), default="all")
	parser.add_argument("--flight", help="Optional flight label such as flight_1")
	parser.add_argument("--dpi", type=int, default=300)
	parser.add_argument("--show", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_arguments()
	if not args.summary_csv.exists():
		raise FileNotFoundError(f"RMSE summary CSV not found: {args.summary_csv}")
	summary = pd.read_csv(args.summary_csv)
	_require_columns(summary, REQUIRED_SUMMARY_COLUMNS, args.summary_csv)
	if args.group != "all":
		summary = summary[summary["group"] == args.group]
	if args.flight:
		summary = summary[summary["flight"] == args.flight]
	if summary.empty:
		raise ValueError("No flights match the requested group/flight selection")

	all_diagnostics = []
	for row in summary.itertuples(index=False):
		all_diagnostics.append(analyze_flight(
			row,
			args.summary_csv,
			args.bags_dir,
			args.out_dir,
			args.pose_topic,
			args.dpi,
			args.show,
		))

	args.out_dir.mkdir(parents=True, exist_ok=True)
	combined_path = args.out_dir / "yaw_rate_analysis_summary.csv"
	pd.DataFrame(all_diagnostics).to_csv(combined_path, index=False)
	print(f"Saved combined diagnostics for {len(all_diagnostics)} flights: {combined_path}")


if __name__ == "__main__":
	main()
