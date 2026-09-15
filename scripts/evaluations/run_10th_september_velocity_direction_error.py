#!/usr/bin/env python3

"""Calculate 10 September horizontal velocity direction error.

The calculation has no acceptance limits. Direction error is deliberately
undefined (NaN) wherever GT horizontal speed is below the configured minimum.
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "10th_September_High_Dyn_results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "VHE"
DURATIONS = (10, 20, 30)
GROUPS = ("AI", "RAW")
FLIGHTS = (1, 2, 3, 4)
DATASET_DESCRIPTION = "10 September high-dynamic velocity direction error"
DATASET_SLUG = "10th_september"
INPUT_SUMMARY_SLUG = "10th_september_high_dynamic"
REQUIRED_COLUMNS = ("time", "vel_x", "vel_y")


def _load_velocity(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {path}")
	df = df.loc[:, REQUIRED_COLUMNS].copy()
	for column in REQUIRED_COLUMNS:
		df[column] = pd.to_numeric(df[column], errors="coerce")
	df = df.dropna(subset=REQUIRED_COLUMNS).sort_values("time").reset_index(drop=True)
	if df.empty:
		raise ValueError(f"No valid velocity samples in {path}")
	return df


def _merge_velocity(est_path: Path, gt_path: Path) -> pd.DataFrame:
	est = _load_velocity(est_path)
	gt = _load_velocity(gt_path)
	merged = pd.merge(est, gt, on="time", how="inner", suffixes=("_est", "_gt"))
	if merged.empty:
		raise ValueError(f"No common timestamps between {est_path} and {gt_path}")
	return merged.sort_values("time").reset_index(drop=True)


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
	wrapped_difference = np.arctan2(
		np.sin(est_heading - gt_heading),
		np.cos(est_heading - gt_heading),
	)
	direction_error_deg = np.abs(np.degrees(wrapped_difference))

	# A zero estimate also has no defined direction. Most importantly, every
	# sample below the requested GT threshold remains NaN rather than zero.
	valid_direction = (gt_speed >= min_gt_speed_mps) & (est_speed > 0.0)
	direction_error_deg[~valid_direction] = np.nan

	result = pd.DataFrame(
		{
			"time": merged["time"].to_numpy(dtype=float),
			"time_from_window_start_s": merged["time"].to_numpy(dtype=float)
			- float(merged["time"].iloc[0]),
			"vel_x_est": est_x,
			"vel_y_est": est_y,
			"vel_x_gt": gt_x,
			"vel_y_gt": gt_y,
			"est_horizontal_speed_mps": est_speed,
			"gt_horizontal_speed_mps": gt_speed,
			"est_heading_deg": np.degrees(est_heading),
			"gt_heading_deg": np.degrees(gt_heading),
			"velocity_direction_error_deg": direction_error_deg,
			"direction_error_evaluated": valid_direction,
		}
	)
	return result


def _nearest_rank(values: pd.Series, percentile: float) -> float:
	valid = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
	if len(valid) == 0:
		return float("nan")
	index = int(np.ceil(percentile / 100.0 * len(valid))) - 1
	return float(valid[min(max(index, 0), len(valid) - 1)])


def summarize_samples(
	samples: pd.DataFrame,
	group: str,
	flight: int,
	duration_s: int,
	min_gt_speed_mps: float,
	sampled_start_s: float,
	sampled_end_s: float,
) -> Dict[str, object]:
	error = pd.to_numeric(samples["velocity_direction_error_deg"], errors="coerce")
	valid = error.dropna()
	return {
		"group": group,
		"flight": f"flight_{flight}",
		"configured_duration_s": duration_s,
		"sampled_window_start_s": sampled_start_s,
		"sampled_window_end_s": sampled_end_s,
		"sampled_window_duration_s": sampled_end_s - sampled_start_s,
		"minimum_gt_horizontal_speed_mps": min_gt_speed_mps,
		"total_samples": int(len(samples)),
		"evaluated_direction_samples": int(error.notna().sum()),
		"ignored_direction_samples": int(error.isna().sum()),
		"mean_direction_error_deg": float(valid.mean()) if len(valid) else float("nan"),
		"median_direction_error_deg": float(valid.median()) if len(valid) else float("nan"),
		"p90_direction_error_deg": _nearest_rank(valid, 90.0),
		"max_direction_error_deg": float(valid.max()) if len(valid) else float("nan"),
	}


def _style_axis(axis: plt.Axes) -> None:
	axis.grid(True, which="major", alpha=0.45)
	axis.grid(True, which="minor", alpha=0.2)
	axis.xaxis.set_major_locator(MaxNLocator(nbins=12))
	axis.xaxis.set_minor_locator(AutoMinorLocator(2))
	axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
	axis.yaxis.set_minor_locator(AutoMinorLocator(2))
	axis.tick_params(axis="both", which="major", labelsize=17)
	axis.tick_params(axis="both", which="minor", labelsize=15)


def _shade_low_speed_regions(
	axis: plt.Axes,
	time: np.ndarray,
	low_speed: np.ndarray,
	min_gt_speed_mps: float,
) -> None:
	"""Shade contiguous sample regions excluded by the GT-speed threshold."""
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
			color="#9e9e9e",
			alpha=0.30,
			linewidth=0,
			label=(
				f"GT speed < {min_gt_speed_mps:g} m/s (not evaluated)"
				if region_number == 0
				else None
			),
		)


def plot_direction_and_speed(
	samples: pd.DataFrame,
	output_path: Path,
	group: str,
	flight: int,
	min_gt_speed_mps: float,
	sampled_start_s: float,
	sampled_end_s: float,
	dpi: int,
) -> None:
	time = samples["time_from_window_start_s"].to_numpy(dtype=float)
	error = samples["velocity_direction_error_deg"].to_numpy(dtype=float)
	est_speed = samples["est_horizontal_speed_mps"].to_numpy(dtype=float)
	gt_speed = samples["gt_horizontal_speed_mps"].to_numpy(dtype=float)
	low_gt_speed = gt_speed < min_gt_speed_mps
	mean_error = float(np.nanmean(error)) if np.any(np.isfinite(error)) else float("nan")

	fig, (direction_axis, speed_axis) = plt.subplots(
		2,
		1,
		figsize=(16, 12),
		sharex=True,
		gridspec_kw={"height_ratios": [1, 1]},
	)
	_shade_low_speed_regions(direction_axis, time, low_gt_speed, min_gt_speed_mps)
	direction_axis.plot(time, error, color="#1f4bb3", linewidth=4.0, label="Direction error")
	if np.isfinite(mean_error):
		direction_axis.axhline(
			mean_error,
			color="#ff7f0e",
			linestyle="--",
			linewidth=3.2,
			label=f"Mean direction error ({mean_error:.2f} deg)",
		)
	direction_axis.set_title(
		f"{group} Flight {flight}: Velocity Direction Error (GT speed >= {min_gt_speed_mps:g} m/s)",
		fontsize=22,
	)
	direction_axis.set_ylabel("Direction error (deg)", fontsize=19)
	direction_axis.legend(loc="best", fontsize=16)
	_style_axis(direction_axis)

	_shade_low_speed_regions(speed_axis, time, low_gt_speed, min_gt_speed_mps)
	speed_axis.plot(time, est_speed, color="#d62728", linewidth=4.0, label="Estimate")
	speed_axis.plot(time, gt_speed, color="#2ca02c", linewidth=4.0, label="Ground truth")
	speed_axis.axhline(
		min_gt_speed_mps,
		color="#666666",
		linestyle="--",
		linewidth=2.5,
		label=f"GT evaluation threshold ({min_gt_speed_mps:g} m/s)",
	)
	speed_axis.set_title("Horizontal Velocity Magnitude: Estimate vs Ground Truth", fontsize=22)
	speed_axis.set_xlabel(
		"Time from evaluation-window start (s) "
		f"[flight time: {sampled_start_s:.2f} -> {sampled_end_s:.2f} s]",
		fontsize=19,
	)
	speed_axis.set_ylabel("Velocity magnitude (m/s)", fontsize=19)
	speed_axis.legend(loc="best", fontsize=16)
	_style_axis(speed_axis)

	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def _selected(value: str, available: Iterable[str]) -> Tuple[str, ...]:
	return tuple(available) if value == "all" else (value,)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=DATASET_DESCRIPTION)
	parser.add_argument("--results_dir", default=str(DEFAULT_RESULTS_DIR))
	parser.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--group", choices=("all", *GROUPS), default="all")
	parser.add_argument(
		"--duration",
		choices=("all", *(f"{duration}s" for duration in DURATIONS)),
		default="all",
	)
	parser.add_argument("--min_gt_speed", type=float, default=0.2)
	parser.add_argument("--dpi", type=int, default=300)
	args = parser.parse_args()
	args.results_dir = Path(args.results_dir).resolve()
	args.out_dir = Path(args.out_dir).resolve()
	if args.min_gt_speed < 0.0:
		parser.error("--min_gt_speed must be non-negative")
	return args


def write_group_summary_csvs(summary_df: pd.DataFrame, out_dir: Path) -> Path:
	"""Write one flight-plus-average CSV per duration/group and one averages table."""
	average_rows = []
	for duration_s in DURATIONS:
		for group in GROUPS:
			subset = summary_df[
				(summary_df["configured_duration_s"] == duration_s)
				& (summary_df["group"] == group)
			].copy()
			if subset.empty:
				continue

			valid_counts = subset["evaluated_direction_samples"].to_numpy(dtype=float)
			flight_means = subset["mean_direction_error_deg"].to_numpy(dtype=float)
			valid_mean = np.isfinite(flight_means) & (valid_counts > 0)
			total_evaluated = int(subset["evaluated_direction_samples"].sum())
			overall_average = (
				float(np.average(flight_means[valid_mean], weights=valid_counts[valid_mean]))
				if np.any(valid_mean)
				else float("nan")
			)

			rows = [
				{
					"group": row.group,
					"duration_s": int(row.configured_duration_s),
					"flight": row.flight,
					"average_direction_error_deg": row.mean_direction_error_deg,
					"evaluated_direction_samples": int(row.evaluated_direction_samples),
					"ignored_direction_samples": int(row.ignored_direction_samples),
				}
				for row in subset.itertuples(index=False)
			]
			average_row = {
				"group": group,
				"duration_s": duration_s,
				"flight": "AVERAGE",
				"average_direction_error_deg": overall_average,
				"evaluated_direction_samples": total_evaluated,
				"ignored_direction_samples": int(subset["ignored_direction_samples"].sum()),
			}
			rows.append(average_row)
			average_rows.append(average_row)

			group_dir = out_dir / f"{duration_s}s" / group
			group_dir.mkdir(parents=True, exist_ok=True)
			group_path = group_dir / f"{group.lower()}_{duration_s}s_direction_error_summary.csv"
			pd.DataFrame(rows).to_csv(group_path, index=False, float_format="%.6f")
			print(f"Saved group summary: {group_path}")

	averages_path = out_dir / f"{DATASET_SLUG}_ai_raw_direction_error_averages.csv"
	pd.DataFrame(average_rows).to_csv(averages_path, index=False, float_format="%.6f")
	return averages_path


def run() -> None:
	args = parse_arguments()
	groups = _selected(args.group, GROUPS)
	durations = DURATIONS if args.duration == "all" else (int(args.duration[:-1]),)
	summary_rows = []

	for duration_s in durations:
		window_summary_path = (
			args.results_dir
			/ f"{duration_s}s"
			/ f"{INPUT_SUMMARY_SLUG}_{duration_s}s_summary_by_flight.csv"
		)
		window_summary = pd.read_csv(window_summary_path)
		for group in groups:
			for flight in FLIGHTS:
				input_dir = args.results_dir / f"{duration_s}s" / group
				est_path = input_dir / f"flight_{flight}_vel_est_clean_{duration_s}s_window.csv"
				gt_path = input_dir / f"flight_{flight}_vel_gt_clean_aligned_{duration_s}s_window.csv"
				merged = _merge_velocity(est_path, gt_path)
				samples = calculate_direction_error(merged, args.min_gt_speed)

				window_row = window_summary[
					(window_summary["group"] == group)
					& (window_summary["flight"] == f"flight_{flight}")
				]
				if len(window_row) != 1:
					raise ValueError(f"Missing unique window metadata for {group} flight_{flight} {duration_s}s")
				sampled_start_s = float(window_row.iloc[0]["sampled_window_start_s"])
				sampled_end_s = float(window_row.iloc[0]["sampled_window_end_s"])

				output_dir = args.out_dir / f"{duration_s}s" / group
				stem = f"flight_{flight}_{duration_s}s_velocity_direction_error"
				samples_path = output_dir / f"{stem}_samples.csv"
				plot_path = output_dir / f"{stem}.png"
				output_dir.mkdir(parents=True, exist_ok=True)
				samples.to_csv(samples_path, index=False)
				plot_direction_and_speed(
					samples,
					plot_path,
					group,
					flight,
					args.min_gt_speed,
					sampled_start_s,
					sampled_end_s,
					args.dpi,
				)
				summary = summarize_samples(
					samples,
					group,
					flight,
					duration_s,
					args.min_gt_speed,
					sampled_start_s,
					sampled_end_s,
				)
				summary["samples_csv"] = str(samples_path)
				summary["plot_file"] = str(plot_path)
				summary_rows.append(summary)
				print(
					f"Saved {group} flight_{flight} {duration_s}s: "
					f"{summary['evaluated_direction_samples']}/{summary['total_samples']} samples evaluated"
				)

	summary_df = pd.DataFrame(summary_rows)
	args.out_dir.mkdir(parents=True, exist_ok=True)
	summary_path = args.out_dir / f"{DATASET_SLUG}_velocity_direction_error_summary.csv"
	summary_df.to_csv(summary_path, index=False)
	print(f"Saved combined summary: {summary_path}")
	averages_path = write_group_summary_csvs(summary_df, args.out_dir)
	print(f"Saved AI/RAW averages: {averages_path}")


if __name__ == "__main__":
	run()
