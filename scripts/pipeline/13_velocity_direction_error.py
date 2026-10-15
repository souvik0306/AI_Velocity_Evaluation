#!/usr/bin/env python3

"""Calculate and plot horizontal velocity direction error for one window.

The estimate and ground-truth CSVs must already be cleaned, time-aligned, and
clipped to the desired evaluation window. Direction error is deliberately
undefined when the GT horizontal speed is below the configured minimum or the
estimated horizontal speed is zero.
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MIN_FONT_SIZE = 18
DATA_LINE_WIDTH = 4.0
REFERENCE_LINE_WIDTH = 3.5
LOW_SPEED_LINE_ALPHA = 0.38
LOW_SPEED_REGION_COLOR = "#d8e2ec"
LOW_SPEED_REGION_EDGE_COLOR = "#9aabba"
LOW_SPEED_REGION_ALPHA = 0.48


def configure_plot_style() -> None:
	"""Apply readable defaults shared by direction-error plots."""
	plt.rcParams.update(
		{
			"font.size": MIN_FONT_SIZE,
			"axes.titlesize": 22,
			"axes.labelsize": 20,
			"xtick.labelsize": 20,
			"ytick.labelsize": MIN_FONT_SIZE,
			"legend.fontsize": MIN_FONT_SIZE,
			"figure.titlesize": 24,
			"lines.linewidth": DATA_LINE_WIDTH,
		}
	)


def _load_script_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


# Script filenames begin with numbers, so a normal Python import is not valid.
# Reuse script 9's validation, exact timestamp merge, and percentile behavior.
VHE = _load_script_module("9_velocity_heading_error.py", "velocity_heading_error")


def calculate_direction_error(merged: pd.DataFrame, min_gt_speed_mps: float) -> pd.DataFrame:
	"""Return per-sample unsigned horizontal direction error in [0, 180] degrees."""
	est_x = merged["vel_x_est"].to_numpy(dtype=float)
	est_y = merged["vel_y_est"].to_numpy(dtype=float)
	gt_x = merged["vel_x_gt"].to_numpy(dtype=float)
	gt_y = merged["vel_y_gt"].to_numpy(dtype=float)

	est_speed = np.hypot(est_x, est_y)
	gt_speed = np.hypot(gt_x, gt_y)
	est_heading = np.arctan2(est_y, est_x)
	gt_heading = np.arctan2(gt_y, gt_x)
	heading_difference = est_heading - gt_heading
	wrapped_difference = np.arctan2(np.sin(heading_difference), np.cos(heading_difference))
	raw_direction_error_deg = np.abs(np.degrees(wrapped_difference))
	raw_direction_error_deg[est_speed <= 0.0] = np.nan

	valid_direction = (gt_speed >= min_gt_speed_mps) & (est_speed > 0.0)
	direction_error_deg = raw_direction_error_deg.copy()
	direction_error_deg[~valid_direction] = np.nan
	time = merged["time"].to_numpy(dtype=float)

	return pd.DataFrame(
		{
			"time": time,
			"time_from_window_start_s": time - time[0],
			"vel_x_est": est_x,
			"vel_y_est": est_y,
			"vel_x_gt": gt_x,
			"vel_y_gt": gt_y,
			"est_horizontal_speed_mps": est_speed,
			"gt_horizontal_speed_mps": gt_speed,
			"est_heading_deg": np.degrees(est_heading),
			"gt_heading_deg": np.degrees(gt_heading),
			"velocity_direction_error_raw_deg": raw_direction_error_deg,
			"velocity_direction_error_deg": direction_error_deg,
			"direction_error_evaluated": valid_direction,
		}
	)


def summarize_direction_error(samples: pd.DataFrame, min_gt_speed_mps: float) -> pd.DataFrame:
	error = pd.to_numeric(samples["velocity_direction_error_deg"], errors="coerce")
	valid = error.dropna()
	return pd.DataFrame(
		[
			{
				"minimum_gt_horizontal_speed_mps": min_gt_speed_mps,
				"total_samples": int(len(samples)),
				"evaluated_direction_samples": int(error.notna().sum()),
				"ignored_direction_samples": int(error.isna().sum()),
				"mean_direction_error_deg": float(valid.mean()) if len(valid) else float("nan"),
				"median_direction_error_deg": float(valid.median()) if len(valid) else float("nan"),
				"p90_direction_error_deg": VHE._nearest_rank_percentile(valid, 90.0),
				"max_direction_error_deg": float(valid.max()) if len(valid) else float("nan"),
			}
		]
	)


def _style_axis(axis: plt.Axes) -> None:
	axis.grid(True, which="major", alpha=0.45)
	axis.grid(True, which="minor", alpha=0.2)
	axis.xaxis.set_major_locator(MaxNLocator(nbins=12))
	axis.xaxis.set_minor_locator(AutoMinorLocator(2))
	axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
	axis.yaxis.set_minor_locator(AutoMinorLocator(2))


def _shade_low_speed_regions(
	axis: plt.Axes,
	time: np.ndarray,
	low_speed: np.ndarray,
	min_gt_speed_mps: float,
) -> None:
	if len(time) == 0 or not np.any(low_speed):
		return

	changes = np.diff(low_speed.astype(np.int8))
	starts = list(np.flatnonzero(changes == 1) + 1)
	ends = list(np.flatnonzero(changes == -1))
	if low_speed[0]:
		starts.insert(0, 0)
	if low_speed[-1]:
		ends.append(len(low_speed) - 1)

	for region_number, (start, end) in enumerate(zip(starts, ends)):
		left = time[start] if start == 0 else (time[start - 1] + time[start]) / 2.0
		right = time[end] if end == len(time) - 1 else (time[end] + time[end + 1]) / 2.0
		axis.axvspan(
			left,
			right,
			facecolor=LOW_SPEED_REGION_COLOR,
			edgecolor=LOW_SPEED_REGION_EDGE_COLOR,
			alpha=LOW_SPEED_REGION_ALPHA,
			linewidth=1.2,
			linestyle="--",
			zorder=0,
			label=(
				f"Low-speed region: GT < {min_gt_speed_mps:g} m/s"
				if region_number == 0
				else None
			),
		)


def plot_direction_and_speed(
	samples: pd.DataFrame,
	output_path: Path,
	min_gt_speed_mps: float,
	title: str,
	dpi: int,
) -> None:
	configure_plot_style()
	time = samples["time_from_window_start_s"].to_numpy(dtype=float)
	error = samples["velocity_direction_error_deg"].to_numpy(dtype=float)
	raw_error = samples.get(
		"velocity_direction_error_raw_deg",
		samples["velocity_direction_error_deg"],
	).to_numpy(dtype=float)
	est_speed = samples["est_horizontal_speed_mps"].to_numpy(dtype=float)
	gt_speed = samples["gt_horizontal_speed_mps"].to_numpy(dtype=float)
	low_gt_speed = gt_speed < min_gt_speed_mps
	mean_error = float(np.nanmean(error)) if np.any(np.isfinite(error)) else float("nan")

	fig, (direction_axis, speed_axis) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
	_shade_low_speed_regions(direction_axis, time, low_gt_speed, min_gt_speed_mps)
	direction_axis.plot(
		time,
		error,
		color="#1f4bb3",
		linewidth=DATA_LINE_WIDTH,
		label="Direction error (evaluated)",
	)
	low_speed_error = np.where(low_gt_speed, raw_error, np.nan)
	direction_axis.plot(
		time,
		low_speed_error,
		color="#1f4bb3",
		linewidth=DATA_LINE_WIDTH,
		alpha=LOW_SPEED_LINE_ALPHA,
		label=f"Direction error where GT < {min_gt_speed_mps:g} m/s",
	)
	if np.isfinite(mean_error):
		direction_axis.axhline(
			mean_error,
			color="#ff7f0e",
			linestyle="--",
			linewidth=REFERENCE_LINE_WIDTH,
			label=f"Mean direction error ({mean_error:.2f} deg)",
		)
	direction_axis.set_title(title)
	direction_axis.set_ylabel("Direction error (deg)")
	direction_axis.legend(loc="best")
	_style_axis(direction_axis)

	_shade_low_speed_regions(speed_axis, time, low_gt_speed, min_gt_speed_mps)
	speed_axis.plot(
		time,
		est_speed,
		color="#d62728",
		linewidth=DATA_LINE_WIDTH,
		label="Estimate",
	)
	speed_axis.plot(
		time,
		gt_speed,
		color="#7b2cbf",
		linewidth=DATA_LINE_WIDTH,
		label="Ground truth",
	)
	speed_axis.axhline(
		min_gt_speed_mps,
		color="#666666",
		linestyle="--",
		linewidth=REFERENCE_LINE_WIDTH,
		label=f"GT evaluation threshold ({min_gt_speed_mps:g} m/s)",
	)
	speed_axis.set_title("Horizontal Speed: Estimate vs Ground Truth")
	speed_axis.set_xlabel("Time from evaluation-window start (s)", fontsize=24)
	speed_axis.set_ylabel("Horizontal speed (m/s)")
	speed_axis.legend(loc="best")
	_style_axis(speed_axis)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def _derive_output_paths(est_path: Path) -> Tuple[Path, Path, Path]:
	stem = est_path.parent / f"{est_path.stem}_velocity_direction_error"
	return (
		stem.with_name(f"{stem.name}_samples.csv"),
		stem.with_name(f"{stem.name}_summary.csv"),
		stem.with_suffix(".png"),
	)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Calculate and plot unsigned horizontal velocity direction error for one "
			"cleaned, aligned evaluation window."
		),
	)
	parser.add_argument("--est_csv", required=True, help="Windowed estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="Windowed aligned GT velocity CSV")
	parser.add_argument("--out_samples_csv", help="Per-sample direction-error CSV")
	parser.add_argument("--out_summary_csv", help="Direction-error summary CSV")
	parser.add_argument("--out_plot", help="Direction-error and speed plot")
	parser.add_argument(
		"--min_gt_speed",
		type=float,
		default=0.2,
		help="Minimum GT horizontal speed at which direction is evaluated (m/s)",
	)
	parser.add_argument("--title", default="Horizontal Velocity Direction Error")
	parser.add_argument("--dpi", type=int, default=300)
	args = parser.parse_args()
	if args.min_gt_speed < 0.0:
		parser.error("--min_gt_speed must be non-negative")
	if args.dpi < 1:
		parser.error("--dpi must be positive")
	return args


def run() -> None:
	args = parse_arguments()
	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	if not est_path.exists():
		raise SystemExit(f"Estimate CSV not found: {est_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	default_samples, default_summary, default_plot = _derive_output_paths(est_path)
	samples_path = Path(args.out_samples_csv) if args.out_samples_csv else default_samples
	summary_path = Path(args.out_summary_csv) if args.out_summary_csv else default_summary
	plot_path = Path(args.out_plot) if args.out_plot else default_plot

	est_df = VHE._load_velocity_csv(est_path, VHE.REQUIRED_COLS)
	gt_df = VHE._load_velocity_csv(gt_path, VHE.REQUIRED_COLS)
	merged = VHE._merge_velocity_data(est_df, gt_df)
	samples = calculate_direction_error(merged, args.min_gt_speed)
	summary = summarize_direction_error(samples, args.min_gt_speed)

	samples_path.parent.mkdir(parents=True, exist_ok=True)
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	samples.to_csv(samples_path, index=False)
	summary.to_csv(summary_path, index=False)
	plot_direction_and_speed(samples, plot_path, args.min_gt_speed, args.title, args.dpi)

	print(summary.to_string(index=False))
	print(f"Saved per-sample direction error to {samples_path}")
	print(f"Saved direction-error summary to {summary_path}")
	print(f"Saved direction-error plot to {plot_path}")


if __name__ == "__main__":
	run()
