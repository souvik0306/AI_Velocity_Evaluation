#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


REQUIRED_VEL_COLS = ["time", "vel_x", "vel_y"]
REQUIRED_ERR_COLS = ["time", "err_x", "err_y"]
DEFAULT_POSE_TOPIC = "/mavros/local_position/pose"


def _load_csv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {path}")

	df = df.copy()
	df["time"] = pd.to_numeric(df["time"], errors="coerce")
	df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	return df


def _derive_default_path(path: Path, suffix: str) -> Path:
	stem = path.stem
	for old_suffix in ("_vel_est_clean_window_bias_errors", "_vel_est_clean_window"):
		if stem.endswith(old_suffix):
			stem = stem[: -len(old_suffix)]
			break
	return path.with_name(f"{stem}{suffix}{path.suffix}")


def _read_pose_z_from_bag(
	bag_path: Path,
	start_time: float,
	end_time: float,
	topic: str,
) -> Tuple[pd.DataFrame, float]:
	try:
		import rosbag
	except ImportError as exc:
		raise SystemExit(
			"Could not import rosbag. Run this script in a ROS environment "
			"where the rosbag Python package is available.",
		) from exc

	rows = []
	with rosbag.Bag(str(bag_path), "r") as bag:
		bag_start_time = float(bag.get_start_time())
		for _, msg, stamp in bag.read_messages(topics=[topic]):
			t = stamp.to_sec()
			if t < start_time:
				continue
			if t > end_time:
				break
			rows.append(
				{
					"time": t,
					"z": msg.pose.position.z,
				},
			)

	if not rows:
		raise ValueError(
			f"No pose Z samples found on {topic} in window "
			f"{start_time:.6f} -> {end_time:.6f}",
		)

	return pd.DataFrame(rows).sort_values("time").reset_index(drop=True), bag_start_time


def _merge_plot_data(
	err_df: pd.DataFrame,
	est_df: pd.DataFrame,
	gt_df: pd.DataFrame,
) -> pd.DataFrame:
	merged = pd.merge(
		err_df[["time", "err_x", "err_y"]],
		est_df[["time", "vel_x", "vel_y"]],
		on="time",
		how="inner",
	)
	merged = pd.merge(
		merged,
		gt_df[["time", "vel_x", "vel_y"]],
		on="time",
		how="inner",
		suffixes=("_est", "_gt"),
	)
	if merged.empty:
		raise ValueError("No overlapping timestamps across error, estimate, and GT CSVs")
	return merged.sort_values("time").reset_index(drop=True)


def _style_axis(ax) -> None:
	ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.grid(True, which="major", alpha=0.75)
	ax.grid(True, which="minor", alpha=0.25)


def _window_xlabel(
	window_start_time: float,
	window_end_time: float,
	flight_time_zero: float,
) -> str:
	flight_rel_start = window_start_time - flight_time_zero
	flight_rel_end = window_end_time - flight_time_zero
	return (
		"time from window start (s) "
		f"[flight: {flight_rel_start:.2f} -> {flight_rel_end:.2f} s]"
	)


def _plot_axis(
	merged: pd.DataFrame,
	z_df: pd.DataFrame,
	axis: str,
	out_dir: Path,
	prefix: str,
	dpi: int,
	show: bool,
	flight_time_zero: float,
) -> Path:
	time0 = float(merged["time"].iloc[0])
	time1 = float(merged["time"].iloc[-1])
	rel_time = merged["time"].to_numpy(dtype=float) - time0
	z_rel_time = z_df["time"].to_numpy(dtype=float) - time0

	err_col = f"err_{axis}"
	est_col = f"vel_{axis}_est"
	gt_col = f"vel_{axis}_gt"

	fig, (ax_vel, ax_z) = plt.subplots(
		2,
		1,
		figsize=(16, 10),
		sharex=True,
		gridspec_kw={"height_ratios": [2.2, 1.0]},
	)

	ax_vel.plot(
		rel_time,
		merged[err_col].abs().to_numpy(dtype=float),
		label=f"|{err_col}|",
		color="#1f42b4",
		linewidth=2.6,
	)
	ax_vel.plot(
		rel_time,
		merged[est_col].to_numpy(dtype=float),
		label=f"est_{axis}",
		color="#d62728",
		linewidth=2.6,
	)
	ax_vel.plot(
		rel_time,
		merged[gt_col].to_numpy(dtype=float),
		label=f"gt_{axis}",
		color="#2ca02c",
		linewidth=2.6,
	)
	ax_vel.set_title(f"Absolute Velocity Error and Velocity Comparison: {axis.upper()}")
	ax_vel.set_ylabel("velocity / error (m/s)")
	ax_vel.legend(loc="best")
	_style_axis(ax_vel)

	ax_z.plot(
		z_rel_time,
		z_df["z"].to_numpy(dtype=float),
		label="local_position_z",
		color="#4b4b4b",
		linewidth=2.6,
	)
	ax_z.set_title("Local Position Z Over Evaluation Window", fontsize=14)
	ax_z.set_xlabel(
		_window_xlabel(time0, time1, flight_time_zero),
		fontsize=12,
	)
	ax_z.set_ylabel("z position (m)")
	ax_z.legend(loc="best")
	_style_axis(ax_z)

	fig.tight_layout()
	output_path = out_dir / f"{prefix}_abs_error_vel_{axis}_with_z.png"
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	if not show:
		plt.close(fig)
	return output_path


def _plot_xy_error(
	merged: pd.DataFrame,
	z_df: pd.DataFrame,
	out_dir: Path,
	prefix: str,
	dpi: int,
	show: bool,
	flight_time_zero: float,
) -> Path:
	time0 = float(merged["time"].iloc[0])
	time1 = float(merged["time"].iloc[-1])
	rel_time = merged["time"].to_numpy(dtype=float) - time0
	z_rel_time = z_df["time"].to_numpy(dtype=float) - time0
	xy_error = (
		merged["err_x"].to_numpy(dtype=float) ** 2
		+ merged["err_y"].to_numpy(dtype=float) ** 2
	) ** 0.5

	fig, (ax_err, ax_z) = plt.subplots(
		2,
		1,
		figsize=(16, 10),
		sharex=True,
		gridspec_kw={"height_ratios": [2.2, 1.0]},
	)

	ax_err.plot(
		rel_time,
		xy_error,
		label="sqrt(err_x^2 + err_y^2)",
		color="#2ca02c",
		linewidth=2.6,
	)
	ax_err.set_title("Horizontal Velocity Error Magnitude")
	ax_err.set_ylabel("XY error magnitude (m/s)")
	ax_err.legend(loc="best")
	_style_axis(ax_err)

	ax_z.plot(
		z_rel_time,
		z_df["z"].to_numpy(dtype=float),
		label="local_position_z",
		color="#4b4b4b",
		linewidth=2.6,
	)
	ax_z.set_title("Local Position Z Over Evaluation Window", fontsize=14)
	ax_z.set_xlabel(
		_window_xlabel(time0, time1, flight_time_zero),
		fontsize=12,
	)
	ax_z.set_ylabel("z position (m)")
	ax_z.legend(loc="best")
	_style_axis(ax_z)

	fig.tight_layout()
	output_path = out_dir / f"{prefix}_xy_error_magnitude_with_z.png"
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	if not show:
		plt.close(fig)
	return output_path


def _plot_drift_fit(
	merged: pd.DataFrame,
	z_df: pd.DataFrame,
	out_dir: Path,
	prefix: str,
	dpi: int,
	show: bool,
	flight_time_zero: float,
) -> Path:
	"""Plot horizontal error magnitude and fitted linear drift (a*t + b).

	Returns the plot path. Does not write parameter files.
	"""
	time0 = float(merged["time"].iloc[0])
	time1 = float(merged["time"].iloc[-1])
	rel_time = merged["time"].to_numpy(dtype=float) - time0
	z_rel_time = z_df["time"].to_numpy(dtype=float) - time0

	time_arr = merged["time"].to_numpy(dtype=float)
	t = time_arr - time_arr[0]
	e_v = (
		merged["err_x"].to_numpy(dtype=float) ** 2
		+ merged["err_y"].to_numpy(dtype=float) ** 2
	) ** 0.5

	if len(t) >= 2:
		drift_rate, drift_intercept = np.polyfit(t, e_v, 1)
	else:
		drift_rate, drift_intercept = float("nan"), float("nan")
	fit = drift_rate * t + drift_intercept

	fig, (ax_err, ax_z) = plt.subplots(
		2,
		1,
		figsize=(16, 10),
		sharex=True,
		gridspec_kw={"height_ratios": [2.2, 1.0]},
	)

	ax_err.plot(
		rel_time,
		e_v,
		label="sqrt(err_x^2 + err_y^2)",
		color="#2ca02c",
		linewidth=2.6,
	)
	ax_err.plot(
		rel_time,
		fit,
		label=f"fit: drift_rate={drift_rate:.6e}, drift_intercept={drift_intercept:.6e}",
		color="#1f77b4",
		linewidth=2.6,
	)
	ax_err.set_title("Drift Fit: e_v(t) ≈ drift_rate*t + drift_intercept")
	ax_err.set_ylabel("error (m/s)")
	ax_err.legend(loc="best")
	_style_axis(ax_err)

	ax_z.plot(
		z_rel_time,
		z_df["z"].to_numpy(dtype=float),
		label="local_position_z",
		color="#4b4b4b",
		linewidth=2.6,
	)
	ax_z.set_title("Local Position Z Over Evaluation Window", fontsize=14)
	ax_z.set_xlabel(
		_window_xlabel(time0, time1, flight_time_zero),
		fontsize=12,
	)
	ax_z.set_ylabel("z position (m)")
	ax_z.legend(loc="best")
	_style_axis(ax_z)

	fig.tight_layout()
	output_path = out_dir / f"{prefix}_drift_fit_with_z.png"
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	if not show:
		plt.close(fig)

	return output_path


def _resolve_input_path(value: Optional[str], default: Path) -> Path:
	if value:
		return Path(value)
	if default.exists():
		return default

	matches = sorted(Path.cwd().rglob(default.name))
	if not matches:
		return default
	return matches[0]


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Plot absolute velocity error with estimate/GT velocity, plus local "
			"position Z from rosbag for the same evaluation window."
		),
	)
	parser.add_argument(
		"--bias_errors_csv",
		default="flight_2026-06-03-09-30-41_vel_est_clean_window_bias_errors.csv",
		help="CSV from 6_bias_zeroing_and_rmse.py containing err_x and err_y",
	)
	parser.add_argument(
		"--est_csv",
		help="Windowed estimate velocity CSV. Defaults are derived from bias_errors_csv.",
	)
	parser.add_argument(
		"--gt_csv",
		help="Windowed aligned GT velocity CSV. Defaults are derived from bias_errors_csv.",
	)
	parser.add_argument(
		"--bag",
		default="2026-06-05/flight_2026-06-03-09-30-41.bag",
		help="Rosbag containing /mavros/local_position/pose",
	)
	parser.add_argument(
		"--pose_topic",
		default=DEFAULT_POSE_TOPIC,
		help="Pose topic with pose/position/z",
	)
	parser.add_argument(
		"--out_dir",
		help="Directory to save plots. Defaults to plots_<bias CSV prefix>.",
	)
	parser.add_argument(
		"--dpi",
		type=int,
		default=600,
		help="Output image DPI",
	)
	parser.add_argument(
		"--flight_time_zero",
		type=float,
		help=(
			"Absolute timestamp to use as flight-relative t=0. "
			"Defaults to the rosbag start time."
		),
	)
	parser.add_argument("--show", action="store_true", help="Show plots interactively")
	args = parser.parse_args()

	bias_path = Path(args.bias_errors_csv)
	if not bias_path.exists():
		raise SystemExit(f"Bias-error CSV not found: {bias_path}")

	est_path = _resolve_input_path(
		args.est_csv,
		_derive_default_path(bias_path, "_vel_est_clean_window"),
	)
	gt_path = _resolve_input_path(
		args.gt_csv,
		_derive_default_path(bias_path, "_vel_gt_clean_aligned_window"),
	)
	bag_path = Path(args.bag)

	for label, path in (
		("Estimate CSV", est_path),
		("GT CSV", gt_path),
		("Rosbag", bag_path),
	):
		if not path.exists():
			raise SystemExit(f"{label} not found: {path}")

	default_out = Path(f"plots_{bias_path.stem}")
	out_dir = Path(args.out_dir) if args.out_dir else default_out
	out_dir.mkdir(parents=True, exist_ok=True)

	err_df = _load_csv(bias_path, REQUIRED_ERR_COLS)
	est_df = _load_csv(est_path, REQUIRED_VEL_COLS)
	gt_df = _load_csv(gt_path, REQUIRED_VEL_COLS)
	merged = _merge_plot_data(err_df, est_df, gt_df)

	# Horizontal velocity error magnitude
	xy_error = (
		merged["err_x"].to_numpy(dtype=float) ** 2
		+ merged["err_y"].to_numpy(dtype=float) ** 2
	) ** 0.5

	# Estimate drift rate (least-squares linear fit e_v(t) ~= drift_rate*t + drift_intercept)
	# where e_v is horizontal velocity error magnitude and t is time since window start
	time_arr = merged["time"].to_numpy(dtype=float)
	t = time_arr - time_arr[0]
	e_v = xy_error
	if len(t) >= 2:
		drift_rate, drift_intercept = np.polyfit(t, e_v, 1)
		drift_rate = float(drift_rate)
		drift_intercept = float(drift_intercept)
	else:
		drift_rate = float("nan")
		drift_intercept = float("nan")

	start_time = float(merged["time"].iloc[0])
	end_time = float(merged["time"].iloc[-1])
	z_df, bag_start_time = _read_pose_z_from_bag(
		bag_path,
		start_time,
		end_time,
		args.pose_topic,
	)
	flight_time_zero = args.flight_time_zero
	if flight_time_zero is None:
		flight_time_zero = bag_start_time

	prefix = bias_path.stem.replace("_vel_est_clean_window_bias_errors", "")
	z_csv_path = out_dir / f"{prefix}_local_position_z_window.csv"
	z_df.to_csv(z_csv_path, index=False)

	output_paths = [
		_plot_axis(
			merged,
			z_df,
			"x",
			out_dir,
			prefix,
			args.dpi,
			args.show,
			flight_time_zero,
		),
		_plot_axis(
			merged,
			z_df,
			"y",
			out_dir,
			prefix,
			args.dpi,
			args.show,
			flight_time_zero,
		),
		_plot_xy_error(
			merged,
			z_df,
			out_dir,
			prefix,
			args.dpi,
			args.show,
			flight_time_zero,
		),
	]

	# Add drift-fit plot
	drift_plot_path = _plot_drift_fit(
		merged, z_df, out_dir, prefix, args.dpi, args.show, flight_time_zero
	)
	output_paths.append(drift_plot_path)


	print(f"Window: {start_time:.6f} -> {end_time:.6f} ({end_time - start_time:.6f} s)")
	print(
		"Flight-relative window: "
		f"{start_time - flight_time_zero:.2f} -> {end_time - flight_time_zero:.2f} s"
	)
	print(f"Drift Rate: {drift_rate:.6f} m/s^2")
	print(f"Drift Intercept: {drift_intercept:.6f} m/s")
	print(f"Saved Z window CSV to {z_csv_path} (rows: {len(z_df)})")
	for output_path in output_paths:
		print(f"Saved plot to {output_path}")

	if args.show:
		plt.show()
	else:
		plt.close("all")


if __name__ == "__main__":
	main()

# Example:
# python3 7_plot_absolute_velocity_error.py \
#   --bias_errors_csv flight_2026-06-03-09-30-41_vel_est_clean_window_bias_errors.csv \
#   --est_csv 2026-06-05/flight_2026-06-03-09-30-41_vel_est_clean_window.csv \
#   --gt_csv 2026-06-05/flight_2026-06-03-09-30-41_vel_gt_clean_aligned_window.csv \
#   --bag 2026-06-05/flight_2026-06-03-09-30-41.bag
