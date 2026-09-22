#!/usr/bin/env python3
"""Shared evaluation flow using the existing velocity pipeline implementations."""
import argparse
import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT
PIPELINE_DIR = REPO_ROOT / "scripts" / "pipeline"
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]
DEFAULT_EST_TOPIC = "/mavros/local_position/velocity_local"
DEFAULT_GT_TOPIC = "/vrpn_client_node/AIIMU1/twist"
HOVER_SPEED_THRESHOLD_MPS = 0.1


def _load_pipeline(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, PIPELINE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONVERTER = _load_pipeline("1_rosbag_to_csv_vel.py", "velocity_converter")
CLEANER = _load_pipeline("3_vel_csv_dataset_cleaner.py", "velocity_cleaner")
METRICS = _load_pipeline("6_bias_zeroing_and_rmse.py", "velocity_metrics")

def extract_bag_velocities(
	bag_path: Path,
	est_path: Path,
	gt_path: Path,
	est_topic: str,
	gt_topic: str,
) -> Tuple[int, int, float]:
	est_rows = []
	gt_rows = []
	with CONVERTER.rosbag.Bag(str(bag_path), "r") as bag:
		flight_time_zero = float(bag.get_start_time())
		for topic, message, stamp in bag.read_messages(topics=[est_topic, gt_topic]):
			row = CONVERTER.twist_message_to_row(stamp.to_sec(), message)
			if topic == est_topic:
				est_rows.append(row)
			else:
				gt_rows.append(row)

	if not est_rows:
		raise ValueError(f"No estimate messages found on {est_topic} in {bag_path}")
	if not gt_rows:
		raise ValueError(f"No GT messages found on {gt_topic} in {bag_path}")

	est_path.parent.mkdir(parents=True, exist_ok=True)
	CONVERTER.build_velocity_dataframe(est_rows).sort_values("time").to_csv(est_path, index=False)
	CONVERTER.build_velocity_dataframe(gt_rows).sort_values("time").to_csv(gt_path, index=False)
	return len(est_rows), len(gt_rows), flight_time_zero


def clean_velocity_csv(
	input_path: Path,
	output_path: Path,
	window: int,
	median_tol: Optional[float],
	bound_overrides: List[Tuple[str, Optional[float], Optional[float]]],
	fill: str,
	savgol_window: Optional[int],
	savgol_polyorder: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
	df = pd.read_csv(input_path)
	cleaned, pruned, filtered = CLEANER.clean_dataframe(
		df,
		window,
		median_tol,
		bound_overrides,
		fill,
		savgol_window=savgol_window,
		savgol_polyorder=savgol_polyorder,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	cleaned.to_csv(output_path, index=False)
	return pruned, filtered


def load_velocity_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [column for column in required_columns if column not in df.columns]
	if missing:
		raise ValueError(f"Missing columns {missing} in {path}")
	df = df.copy()
	for column in required_columns:
		df[column] = pd.to_numeric(df[column], errors="coerce")
	df = df.dropna(subset=required_columns).sort_values("time").reset_index(drop=True)
	if df.empty:
		raise ValueError(f"No valid rows in {path}")
	return df


def align_ground_truth(
	est_clean_path: Path,
	gt_clean_path: Path,
	output_path: Path,
	gt_latency: float,
	tolerance: Optional[float],
) -> Path:
	est_df = load_velocity_csv(est_clean_path, ["time"])
	gt_df = load_velocity_csv(gt_clean_path, ["time"])
	if gt_latency:
		gt_df = gt_df.copy()
		gt_df["time"] = gt_df["time"] - gt_latency
		gt_df = gt_df.sort_values("time").reset_index(drop=True)

	merge_args = {"on": "time", "direction": "nearest", "suffixes": ("_est", "_gt")}
	if tolerance is not None:
		merge_args["tolerance"] = tolerance
	aligned = pd.merge_asof(est_df, gt_df, **merge_args)

	gt_columns = []
	rename = {}
	for column in gt_df.columns:
		if column in aligned.columns:
			gt_columns.append(column)
		elif f"{column}_gt" in aligned.columns:
			gt_columns.append(f"{column}_gt")
			rename[f"{column}_gt"] = column
	aligned[gt_columns].rename(columns=rename).to_csv(output_path, index=False)
	return output_path


def clip_window(
	est_clean_path: Path,
	gt_aligned_path: Path,
	start_s: float,
	duration_s: int,
	output_dir: Path,
	flight_name: str,
) -> Tuple[Path, Path]:
	est_df = load_velocity_csv(est_clean_path, ["time"])
	gt_df = load_velocity_csv(gt_aligned_path, ["time"])
	reference_time = float(gt_df["time"].iloc[0])
	start_time = reference_time + start_s
	end_time = start_time + duration_s
	est_window = est_df[est_df["time"].between(start_time, end_time)].copy()
	gt_window = gt_df[gt_df["time"].between(start_time, end_time)].copy()
	if est_window.empty or gt_window.empty:
		raise ValueError(
			f"Empty {duration_s}s window for {flight_name}: "
			f"estimate rows={len(est_window)}, GT rows={len(gt_window)}"
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	est_out = output_dir / f"{flight_name}_vel_est_clean_{duration_s}s_window.csv"
	gt_out = output_dir / f"{flight_name}_vel_gt_clean_aligned_{duration_s}s_window.csv"
	est_window.to_csv(est_out, index=False)
	gt_window.to_csv(gt_out, index=False)
	return est_out, gt_out


def evaluate_window(
	est_window_path: Path,
	gt_window_path: Path,
	flight_name: str,
	start_s: float,
	duration_s: int,
	plot_dpi: int,
	flight_time_zero: float,
	hover_analysis: bool = False,
) -> Dict[str, object]:
	rmse_metrics, bias_path, rmse_summary_path = calculate_rmse(
		est_window_path, gt_window_path, est_window_path.parent
	)
	hover_regime_metrics = {}
	if hover_analysis:
		hover_regime_metrics = calculate_hover_regime_rmse(bias_path, gt_window_path)
	drift_rate, drift_intercept, plots_dir = calculate_drift_and_plots(
		bias_path=bias_path,
		est_window_path=est_window_path,
		gt_window_path=gt_window_path,
		plot_dpi=plot_dpi,
		flight_time_zero=flight_time_zero,
		plot_hover_horizontal_diagnostic=hover_analysis,
	)
	bias_df = pd.read_csv(bias_path)
	sample_start_s = float(bias_df["time"].iloc[0]) - flight_time_zero
	sample_end_s = float(bias_df["time"].iloc[-1]) - flight_time_zero

	return {
		"flight": flight_name,
		"configured_window_start_s": start_s,
		"configured_window_end_s": start_s + duration_s,
		"configured_window_duration_s": duration_s,
		"sampled_window_start_s": sample_start_s,
		"sampled_window_end_s": sample_end_s,
		"sampled_window_duration_s": sample_end_s - sample_start_s,
		"samples": int(len(bias_df)),
		**rmse_metrics,
		**hover_regime_metrics,
		"drift_rate_mps2": drift_rate,
		"drift_intercept_mps": drift_intercept,
		"rmse_summary_file": str(rmse_summary_path),
		"bias_errors_file": str(bias_path),
		"drift_plots_dir": str(plots_dir),
	}


def summarize_duration(rows: List[Dict[str, object]]) -> Dict[str, object]:
	df = pd.DataFrame(rows)
	metric_columns = [
		"rmse_x_mps",
		"rmse_y_mps",
		"rmse_z_mps",
		"rmse_xy_mps",
		"bias_corrected_rmse_x_mps",
		"bias_corrected_rmse_y_mps",
		"bias_corrected_rmse_z_mps",
		"bias_corrected_rmse_xy_mps",
		"bias_corrected_rmse_xy_stationary_mps",
		"bias_corrected_rmse_xy_moving_mps",
		"drift_rate_mps2",
		"drift_intercept_mps",
	]
	result: Dict[str, object] = {
		"trajectories": len(rows),
		"total_samples": int(df["samples"].sum()),
	}
	for column in metric_columns:
		if column not in df.columns:
			continue
		result[f"mean_{column}"] = float(pd.to_numeric(df[column], errors="coerce").mean())
	for column in ("stationary_samples", "moving_samples"):
		if column in df.columns:
			result[f"total_{column}"] = int(pd.to_numeric(df[column], errors="coerce").sum())
	return result


def calculate_rmse(est_window_path: Path, gt_window_path: Path, output_dir: Path):
    """Calculate the same first-sample bias and RMSE as pipeline script 6."""
    est = METRICS._load_velocity_csv(est_window_path)
    gt = METRICS._load_velocity_csv(gt_window_path)
    merged = pd.merge(est, gt, on="time", how="inner", suffixes=("_est", "_gt"))
    if merged.empty:
        raise ValueError("No overlapping timestamps between estimate and GT CSVs")
    merged = METRICS._compute_bias_errors(merged)
    raw = {axis: METRICS._rmse(merged[f"err_{axis}"]) for axis in "xyz"}
    raw["xy"] = METRICS._rmse_xy(merged["err_x"], merged["err_y"])
    corrected = {axis: METRICS._rmse(merged[f"err_{axis}_bias"]) for axis in "xyz"}
    corrected["xy"] = METRICS._rmse_xy(merged["err_x_bias"], merged["err_y_bias"])
    bias_path = output_dir / f"{est_window_path.stem}_bias_errors.csv"
    summary_path = output_dir / f"{est_window_path.stem}_rmse_summary.txt"
    columns = ["time", "err_x", "err_y", "err_z", "err_x_bias", "err_y_bias", "err_z_bias"]
    merged[columns].to_csv(bias_path, index=False)
    lines = ["RMSE (Not including bias): ", "", *[f"RMSE_{axis}={raw[axis]:.6f}" for axis in ("x", "y", "z", "xy")],
             "", "RMSE (bias corrected):", "", *[f"RMSE_{axis}={corrected[axis]:.6f}" for axis in ("x", "y", "z", "xy")]]
    summary_path.write_text("\n".join(lines) + "\n")
    values = {f"rmse_{axis}_mps": round(raw[axis], 6) for axis in ("x", "y", "z", "xy")}
    values.update({f"bias_corrected_rmse_{axis}_mps": round(corrected[axis], 6) for axis in ("x", "y", "z", "xy")})
    return values, bias_path, summary_path


def calculate_hover_regime_rmse(
	bias_path: Path,
	gt_window_path: Path,
	threshold_mps: float = HOVER_SPEED_THRESHOLD_MPS,
) -> Dict[str, object]:
	"""Calculate bias-corrected horizontal RMSE above and below the hover threshold."""
	errors = load_velocity_csv(
		bias_path,
		["time", "err_x_bias", "err_y_bias"],
	)
	gt = load_velocity_csv(gt_window_path, ["time", "vel_x", "vel_y"])
	merged = pd.merge(errors, gt, on="time", how="inner")
	if merged.empty:
		raise ValueError("No overlapping timestamps between bias errors and GT CSVs")

	gt_xy = np.hypot(
		merged["vel_x"].to_numpy(dtype=float),
		merged["vel_y"].to_numpy(dtype=float),
	)
	squared_xy_error = (
		merged["err_x_bias"].to_numpy(dtype=float) ** 2
		+ merged["err_y_bias"].to_numpy(dtype=float) ** 2
	)
	stationary = gt_xy < threshold_mps
	moving = ~stationary

	def regime_rmse(mask: np.ndarray) -> float:
		if not mask.any():
			return float("nan")
		return float(np.sqrt(np.mean(squared_xy_error[mask])))

	return {
		"stationary_samples": int(stationary.sum()),
		"moving_samples": int(moving.sum()),
		"bias_corrected_rmse_xy_stationary_mps": round(regime_rmse(stationary), 6),
		"bias_corrected_rmse_xy_moving_mps": round(regime_rmse(moving), 6),
	}


def calculate_drift_and_plots(bias_path: Path, est_window_path: Path, gt_window_path: Path,
                              plot_dpi: int, flight_time_zero: float,
                              plot_hover_horizontal_diagnostic: bool = False):
    """Run the existing report script and return its drift rate and plots."""
    plots_dir = bias_path.parent / "plots" / bias_path.stem.replace("_bias_errors", "")
    command = [
        sys.executable, str(PIPELINE_DIR / "7_plot_absolute_velocity_error.py"),
        "--bias_errors_csv", str(bias_path),
        "--est_csv", str(est_window_path),
        "--gt_csv", str(gt_window_path),
        "--out_dir", str(plots_dir),
        "--dpi", str(plot_dpi),
        "--flight_time_zero", f"{flight_time_zero:.9f}",
    ]
    if plot_hover_horizontal_diagnostic:
        command.append("--plot_hover_horizontal_diagnostic")
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Report failed for {bias_path}:\n{completed.stdout}{completed.stderr}")
    rate = re.search(r"^Drift Rate:\s+([0-9eE+.-]+)\s+m/s\^2$", completed.stdout, re.MULTILINE)
    intercept = re.search(r"^Drift Intercept:\s+([0-9eE+.-]+)\s+m/s$", completed.stdout, re.MULTILINE)
    if not rate or not intercept:
        raise ValueError(f"Could not read drift metrics from report output for {bias_path}")
    return float(rate.group(1)), float(intercept.group(1)), plots_dir


def resolve_path(value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else SCRIPT_DIR / path


def validate_configuration(flights, duration_s=20) -> None:
	for group, entries in flights.items():
		for flight, config in entries.items():
			if not np.isfinite(float(config["start"])) or float(config["start"]) < 0:
				raise ValueError(f"Invalid start for {group} flight_{flight}")
			if "end" in config and not np.isclose(float(config["end"]) - float(config["start"]), duration_s):
				raise ValueError(f"{group} flight_{flight} window is not {duration_s} seconds")


def selected_groups(value: str, flights) -> Tuple[str, ...]:
	return tuple(flights) if value == "all" else (value,)


def selected_durations(value: str) -> Tuple[int, ...]:
	return (20,)


def parse_arguments(bags_dir: Path, output_dir: Path, flights) -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate configured ROS bags over 20-second windows."
	)
	parser.add_argument("--bags_dir", default=str(bags_dir))
	parser.add_argument("--out_dir", default=str(output_dir))
	parser.add_argument("--group", choices=["all", *flights], default="all")
	parser.add_argument(
		"--duration",
		choices=["20s"],
		default="20s",
	)
	parser.add_argument("--velocity_topic", default=DEFAULT_EST_TOPIC)
	parser.add_argument("--gt_velocity_topic", default=DEFAULT_GT_TOPIC)
	parser.add_argument("--plot_dpi", type=int, default=300)
	parser.add_argument("--clean_window", type=int, default=9)
	parser.add_argument("--median_tol", type=float, default=None)
	parser.add_argument(
		"--bound",
		action="append",
		default=None,
		help="Cleaning bound pattern:min:max; default is vel_*:-8:8",
	)
	parser.add_argument("--fill", choices=["none", "linear"], default="linear")
	parser.add_argument("--savgol_window", type=int, default=21, help="Use 0 to disable smoothing")
	parser.add_argument("--savgol_polyorder", type=int, default=2)
	parser.add_argument("--gt_latency", type=float, default=0.025)
	parser.add_argument("--tolerance", type=float, default=None)
	args = parser.parse_args()
	args.bags_dir = resolve_path(args.bags_dir)
	args.out_dir = resolve_path(args.out_dir)
	args.clean_window = CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = CLEANER._parse_bound_overrides(
		args.bound or DEFAULT_VELOCITY_BOUNDS
	)
	if args.savgol_window <= 0:
		args.savgol_window = None
	return args


def group_rollup(rows: List[Dict[str, object]], group: str) -> Dict[str, object]:
	rollup = summarize_duration(rows)
	return {"group": group, **rollup}


def evaluate_dataset(
	bags_dir: Path,
	flights,
	hover_analysis: bool = False,
) -> List[Dict[str, object]]:
	group_name = Path(sys.argv[0]).stem[len("analyze_"):]
	args = parse_arguments(bags_dir, REPO_ROOT / "results" / group_name, flights)
	validate_configuration(flights)
	groups = selected_groups(args.group, flights)
	durations = selected_durations(args.duration)
	results: Dict[int, List[Dict[str, object]]] = {duration: [] for duration in durations}
	cleaning_rows: List[Dict[str, object]] = []

	for group in groups:
		for flight, config in flights[group].items():
			flight_name = f"flight_{flight}"
			bag_path = args.bags_dir / group / str(config["bag"])
			if not bag_path.exists():
				raise FileNotFoundError(f"ROS bag not found: {bag_path}")
			print(f"Extracting {group} {flight_name}: {bag_path.name}")

			extracted_dir = args.out_dir / "extracted_csv" / group
			cleaned_dir = args.out_dir / "cleaned_csv" / group
			est_raw = extracted_dir / f"{flight_name}_vel_est.csv"
			gt_raw = extracted_dir / f"{flight_name}_vel_gt.csv"
			est_rows, gt_rows, flight_time_zero = extract_bag_velocities(
				bag_path, est_raw, gt_raw, args.velocity_topic, args.gt_velocity_topic
			)

			est_clean = cleaned_dir / f"{flight_name}_vel_est_clean.csv"
			gt_clean = cleaned_dir / f"{flight_name}_vel_gt_clean.csv"
			gt_aligned = cleaned_dir / f"{flight_name}_vel_gt_clean_aligned.csv"
			est_pruned, est_filtered = clean_velocity_csv(
				est_raw, est_clean, args.clean_window, args.median_tol, args.bound_overrides,
				args.fill, args.savgol_window, args.savgol_polyorder,
			)
			gt_pruned, gt_filtered = clean_velocity_csv(
				gt_raw, gt_clean, args.clean_window, args.median_tol, args.bound_overrides,
				args.fill, args.savgol_window, args.savgol_polyorder,
			)
			align_ground_truth(est_clean, gt_clean, gt_aligned, args.gt_latency, args.tolerance)
			cleaning_rows.append({
				"group": group,
				"flight": flight_name,
				"bag": bag_path.name,
				"estimate_extracted_rows": est_rows,
				"gt_extracted_rows": gt_rows,
				"estimate_bound_pruned_values": sum(est_pruned.values()),
				"gt_bound_pruned_values": sum(gt_pruned.values()),
				"estimate_median_filtered_values": sum(est_filtered.values()),
				"gt_median_filtered_values": sum(gt_filtered.values()),
			})

			start_s = float(config["start"])
			for duration_s in durations:
				end_s = float(config.get("end", start_s + duration_s))
				print(
					f"Evaluating {group} {flight_name}: "
					f"{start_s:.2f} -> {end_s:.2f} s ({duration_s}s)"
				)
				output_dir = args.out_dir / f"{duration_s}s" / group
				est_window, gt_window = clip_window(
					est_clean, gt_aligned, start_s, duration_s, output_dir, flight_name
				)
				row = evaluate_window(
					est_window,
					gt_window,
					flight_name,
					start_s,
					duration_s,
					args.plot_dpi,
					flight_time_zero,
					hover_analysis,
				)
				results[duration_s].append({"group": group, "bag": bag_path.name, **row})

	args.out_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(cleaning_rows).to_csv(
		args.out_dir / f"{group_name}_velocity_cleaning_summary.csv", index=False
	)
	for duration_s in durations:
		duration_dir = args.out_dir / f"{duration_s}s"
		summary_path = duration_dir / f"{group_name}_{duration_s}s_summary_by_flight.csv"
		group_path = duration_dir / f"{group_name}_{duration_s}s_summary_by_group.csv"
		summary_df = pd.DataFrame(results[duration_s])
		group_rows = [
			group_rollup(
				[dict(row) for row in results[duration_s] if row["group"] == group], group
			)
			for group in groups
		]
		duration_dir.mkdir(parents=True, exist_ok=True)
		summary_df.to_csv(summary_path, index=False)
		pd.DataFrame(group_rows).to_csv(group_path, index=False)
		print(f"Saved {summary_path}")
		print(f"Saved {group_path}")

	return results[20]
