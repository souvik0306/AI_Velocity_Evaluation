#!/usr/bin/env python3

"""Run the 9 September hover AI and RAW 20-second velocity evaluations.

Each evaluation starts five seconds after its supplied cutoff. The configured
endpoints are retained as an independent check that every window is 20 seconds.
"""

import argparse
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[2]
PIPELINE_DIR = SCRIPT_DIR / "scripts" / "pipeline"
DEFAULT_BAGS_DIR = SCRIPT_DIR / "9th_September_Hover_Rosbags"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "9th_September_Hover_results"
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]
DEFAULT_EST_TOPIC = "/mavros/local_position/velocity_local"
DEFAULT_GT_TOPIC = "/vrpn_client_node/AIIMU1/twist"

# RAW bags are mapped to flight numbers in chronological filename order.
FLIGHTS = {
	"AI": {
		1: {"bag": "flight_1.bag", "start": 55.15, "end": 75.15},
		2: {"bag": "flight_2.bag", "start": 54.92, "end": 74.92},
		3: {"bag": "flight_3.bag", "start": 42.06, "end": 62.06},
	},
	"RAW": {
		1: {"bag": "flight_2026-07-01-13-01-24.bag", "start": 44.32, "end": 64.32},
		2: {"bag": "flight_2026-07-01-13-04-24.bag", "start": 39.75, "end": 59.75},
		3: {"bag": "flight_2026-07-01-13-06-59.bag", "start": 30.71, "end": 50.71},
	},
}
EVALUATION_DURATION_S = 20


def load_script_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, PIPELINE_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


CONVERTER = load_script_module("1_rosbag_to_csv_vel.py", "rosbag_to_csv_vel")
CLEANER = load_script_module("3_vel_csv_dataset_cleaner.py", "vel_csv_dataset_cleaner")


def resolve_path(value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else SCRIPT_DIR / path


def validate_windows() -> None:
	for group, flights in FLIGHTS.items():
		for flight, config in flights.items():
			start = float(config["start"])
			end = float(config["end"])
			if not np.isclose(end - start, EVALUATION_DURATION_S):
				raise ValueError(
					f"{group} flight_{flight} configured window is not "
					f"{EVALUATION_DURATION_S} seconds: {start} -> {end}"
				)


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
) -> Dict[str, object]:
	rmse_metrics, bias_path, rmse_summary_path = run_script_6(
		est_window_path, gt_window_path, est_window_path.parent
	)
	drift_rate, drift_intercept, plots_dir = run_script_7(
		bias_path=bias_path,
		est_window_path=est_window_path,
		gt_window_path=gt_window_path,
		plot_dpi=plot_dpi,
		flight_time_zero=flight_time_zero,
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
		"drift_rate_mps2": drift_rate,
		"drift_intercept_mps": drift_intercept,
		"rmse_summary_file": str(rmse_summary_path),
		"bias_errors_file": str(bias_path),
		"drift_plots_dir": str(plots_dir),
	}


def run_script_6(
	est_window_path: Path,
	gt_window_path: Path,
	output_dir: Path,
) -> Tuple[Dict[str, float], Path, Path]:
	"""Run script 6 and parse its RMSE summary without recalculating metrics."""
	command = [
		"python3",
		str(PIPELINE_DIR / "6_bias_zeroing_and_rmse.py"),
		"--est_csv",
		str(est_window_path),
		"--gt_csv",
		str(gt_window_path),
	]
	completed = subprocess.run(command, cwd=output_dir, text=True, capture_output=True)
	if completed.returncode != 0:
		raise RuntimeError(
			f"Script 6 failed for {est_window_path.name} ({completed.returncode}):\n"
			f"{completed.stdout}{completed.stderr}"
		)
	print(completed.stdout, end="")

	bias_path = output_dir / f"{est_window_path.stem}_bias_errors.csv"
	summary_path = output_dir / f"{est_window_path.stem}_rmse_summary.txt"
	if not bias_path.exists() or not summary_path.exists():
		raise FileNotFoundError(f"Script 6 did not create expected outputs for {est_window_path}")
	text = summary_path.read_text()
	parts = text.split("RMSE (bias corrected):")
	if len(parts) != 2:
		raise ValueError(f"Unexpected script 6 summary format: {summary_path}")

	def parse_section(section: str) -> Dict[str, float]:
		values = dict(re.findall(r"RMSE_(x|y|z|xy)=([0-9eE+.-]+)", section))
		if set(values) != {"x", "y", "z", "xy"}:
			raise ValueError(f"Incomplete script 6 RMSE values in {summary_path}")
		return {axis: float(value) for axis, value in values.items()}

	raw = parse_section(parts[0])
	bias_corrected = parse_section(parts[1])
	metrics = {
		"rmse_x_mps": raw["x"],
		"rmse_y_mps": raw["y"],
		"rmse_z_mps": raw["z"],
		"rmse_xy_mps": raw["xy"],
		"bias_corrected_rmse_x_mps": bias_corrected["x"],
		"bias_corrected_rmse_y_mps": bias_corrected["y"],
		"bias_corrected_rmse_z_mps": bias_corrected["z"],
		"bias_corrected_rmse_xy_mps": bias_corrected["xy"],
	}
	return metrics, bias_path, summary_path


def run_script_7(
	bias_path: Path,
	est_window_path: Path,
	gt_window_path: Path,
	plot_dpi: int,
	flight_time_zero: float,
) -> Tuple[float, float, Path]:
	"""Use script 7 as the single source of drift metrics and plots."""
	plots_dir = bias_path.parent / "plots" / bias_path.stem.replace("_bias_errors", "")
	command = [
		"python3",
		str(PIPELINE_DIR / "7_plot_absolute_velocity_error.py"),
		"--bias_errors_csv",
		str(bias_path),
		"--est_csv",
		str(est_window_path),
		"--gt_csv",
		str(gt_window_path),
		"--out_dir",
		str(plots_dir),
		"--dpi",
		str(plot_dpi),
		"--flight_time_zero",
		f"{flight_time_zero:.9f}",
	]
	completed = subprocess.run(command, text=True, capture_output=True)
	if completed.returncode != 0:
		raise RuntimeError(
			f"Script 7 failed for {bias_path.name} ({completed.returncode}):\n{completed.stdout}{completed.stderr}"
		)
	print(completed.stdout, end="")
	rate_match = re.search(r"^Drift Rate:\s+([0-9eE+.-]+)\s+m/s\^2$", completed.stdout, re.MULTILINE)
	intercept_match = re.search(
		r"^Drift Intercept:\s+([0-9eE+.-]+)\s+m/s$", completed.stdout, re.MULTILINE
	)
	if not rate_match or not intercept_match:
		raise ValueError(f"Could not parse script 7 drift metrics for {bias_path}")
	return float(rate_match.group(1)), float(intercept_match.group(1)), plots_dir


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
		"drift_rate_mps2",
		"drift_intercept_mps",
	]
	result: Dict[str, object] = {
		"trajectories": len(rows),
		"total_samples": int(df["samples"].sum()),
	}
	for column in metric_columns:
		result[f"mean_{column}"] = float(pd.to_numeric(df[column], errors="coerce").mean())
	return result


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Extract and evaluate the 9 September hover AI/RAW bags over 20-second windows."
	)
	parser.add_argument(
		"--bags_dir",
		default=str(DEFAULT_BAGS_DIR),
		help="Directory containing the AI and RAW bag subdirectories",
	)
	parser.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_DIR), help="Evaluation output directory")
	parser.add_argument("--group", choices=["all", "AI", "RAW"], default="all")
	parser.add_argument("--velocity_topic", default=DEFAULT_EST_TOPIC)
	parser.add_argument("--gt_velocity_topic", default=DEFAULT_GT_TOPIC)
	parser.add_argument("--plot_dpi", type=int, default=300, help="DPI for script 7 plots")
	parser.add_argument("--clean_window", type=int, default=9, help="Sliding median window size")
	parser.add_argument("--median_tol", type=float, default=None, help="Override median tolerance")
	parser.add_argument(
		"--bound",
		action="append",
		default=None,
		help="Cleaning bound pattern:min:max; default is vel_*:-8:8",
	)
	parser.add_argument("--fill", choices=["none", "linear"], default="linear")
	parser.add_argument("--savgol_window", type=int, default=21, help="Use 0 to disable smoothing")
	parser.add_argument("--savgol_polyorder", type=int, default=2)
	parser.add_argument("--gt_latency", type=float, default=0.025, help="Seconds subtracted from GT time")
	parser.add_argument("--tolerance", type=float, default=None, help="Optional maximum alignment delta")
	args = parser.parse_args()
	args.bags_dir = resolve_path(args.bags_dir)
	args.out_dir = resolve_path(args.out_dir)
	args.clean_window = CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = CLEANER._parse_bound_overrides(args.bound or DEFAULT_VELOCITY_BOUNDS)
	if args.savgol_window <= 0:
		args.savgol_window = None
	return args


def run() -> None:
	args = parse_arguments()
	validate_windows()
	groups = ("AI", "RAW") if args.group == "all" else (args.group,)
	cleaning_rows = []
	results: List[Dict[str, object]] = []

	for group in groups:
		for flight, config in FLIGHTS[group].items():
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
			end_s = float(config["end"])
			print(
				f"Evaluating {group} {flight_name}: {start_s:.2f} -> {end_s:.2f} s "
				f"({EVALUATION_DURATION_S}s)"
			)
			duration_dir = args.out_dir / f"{EVALUATION_DURATION_S}s" / group
			est_window, gt_window = clip_window(
				est_clean, gt_aligned, start_s, EVALUATION_DURATION_S, duration_dir, flight_name
			)
			row = evaluate_window(
				est_window,
				gt_window,
				flight_name,
				start_s,
				EVALUATION_DURATION_S,
				args.plot_dpi,
				flight_time_zero,
			)
			results.append({"group": group, "bag": bag_path.name, **row})

	args.out_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(cleaning_rows).to_csv(args.out_dir / "9th_september_velocity_cleaning_summary.csv", index=False)
	duration_dir = args.out_dir / f"{EVALUATION_DURATION_S}s"
	summary_path = duration_dir / "9th_september_hover_20s_summary_by_flight.csv"
	group_path = duration_dir / "9th_september_hover_20s_summary_by_group.csv"
	group_rows = []
	for group in groups:
		group_results = [dict(row) for row in results if row["group"] == group]
		group_rows.append({"group": group, **summarize_duration(group_results)})
	duration_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(results).to_csv(summary_path, index=False)
	pd.DataFrame(group_rows).to_csv(group_path, index=False)
	print(f"Saved {summary_path}")
	print(f"Saved {group_path}")


if __name__ == "__main__":
	run()
