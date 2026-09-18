#!/usr/bin/env python3

"""Evaluate 10 September high-dynamic AI and RAW ROS bags.

Velocity extraction, +/-8 m/s cleaning, and timestamp alignment precede the
evaluation. Script 6 is the sole source of RMSE/bias-error results, and script
7 is the sole source of drift-rate results and plots.
"""

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[2]
EVALUATIONS_DIR = Path(__file__).resolve().parent
DEFAULT_BAGS_DIR = SCRIPT_DIR / "10th_September_High_Dyn_Rosbags"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "10th_September_High_Dyn_results"
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]
DURATIONS = (10, 20, 30)
DATASET_DESCRIPTION = "10 September high-dynamic AI/RAW bags"
DATASET_SLUG = "10th_september_high_dynamic"


FLIGHTS = {
	"AI": {
		1: {"bag": "flight_2026-07-01-12-23-36.bag", "start": 41.14},
		2: {"bag": "flight_2026-07-01-12-27-17.bag", "start": 40.24},
		3: {"bag": "flight_2026-07-01-12-30-45.bag", "start": 37.50},
		4: {"bag": "flight_2026-07-01-12-34-16.bag", "start": 38.23},
	},
	"RAW": {
		1: {"bag": "flight_2026-07-01-13-45-38.bag", "start": 36.19},
		2: {"bag": "flight_2026-07-01-13-47-24.bag", "start": 37.10},
		3: {"bag": "flight_2026-07-01-13-49-14.bag", "start": 37.42},
		4: {"bag": "flight_2026-07-01-13-51-01.bag", "start": 31.98},
	},
}

# Supplied endpoints are retained as an independent configuration check.
SUPPLIED_ENDPOINTS = {
	10: {
		"AI": {1: 51.14, 2: 50.24, 3: 47.50, 4: 48.23},
		"RAW": {1: 46.19, 2: 47.10, 3: 47.42, 4: 41.98},
	},
	20: {
		"AI": {1: 61.14, 2: 60.24, 3: 57.50, 4: 58.23},
		"RAW": {1: 56.19, 2: 57.10, 3: 57.42, 4: 51.98},
	},
	30: {
		"AI": {1: 71.14, 2: 70.24, 3: 67.50, 4: 68.23},
		"RAW": {1: 66.19, 2: 67.10, 3: 67.42, 4: 61.98},
	},
}


def load_base_runner():
	spec = importlib.util.spec_from_file_location(
		"september_velocity_evaluation_base",
		EVALUATIONS_DIR / "run_9th_september_30s_60s_evaluation.py",
	)
	if spec is None or spec.loader is None:
		raise ImportError("Could not load the September velocity evaluation helpers")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


BASE = load_base_runner()


def resolve_path(value: str) -> Path:
	path = Path(value)
	return path if path.is_absolute() else SCRIPT_DIR / path


def validate_configuration() -> None:
	for duration_s in DURATIONS:
		for group, flights in FLIGHTS.items():
			for flight, config in flights.items():
				calculated_end = float(config["start"]) + duration_s
				supplied_end = SUPPLIED_ENDPOINTS[duration_s][group][flight]
				if not np.isclose(calculated_end, supplied_end):
					raise ValueError(
						f"{group} flight_{flight} {duration_s}s endpoint mismatch: "
						f"calculated={calculated_end}, supplied={supplied_end}"
					)


def selected_groups(value: str) -> Tuple[str, ...]:
	return ("AI", "RAW") if value == "all" else (value,)


def selected_durations(value: str) -> Tuple[int, ...]:
	return DURATIONS if value == "all" else (int(value[:-1]),)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=f"Evaluate {DATASET_DESCRIPTION} over the configured windows."
	)
	parser.add_argument("--bags_dir", default=str(DEFAULT_BAGS_DIR))
	parser.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument("--group", choices=["all", "AI", "RAW"], default="all")
	parser.add_argument(
		"--duration",
		choices=["all", *(f"{duration}s" for duration in DURATIONS)],
		default="all",
	)
	parser.add_argument("--velocity_topic", default=BASE.DEFAULT_EST_TOPIC)
	parser.add_argument("--gt_velocity_topic", default=BASE.DEFAULT_GT_TOPIC)
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
	args.clean_window = BASE.CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = BASE.CLEANER._parse_bound_overrides(
		args.bound or DEFAULT_VELOCITY_BOUNDS
	)
	if args.savgol_window <= 0:
		args.savgol_window = None
	return args


def group_rollup(rows: List[Dict[str, object]], group: str) -> Dict[str, object]:
	rollup = BASE.summarize_duration(rows)
	return {"group": group, **rollup}


def run() -> None:
	args = parse_arguments()
	validate_configuration()
	groups = selected_groups(args.group)
	durations = selected_durations(args.duration)
	results: Dict[int, List[Dict[str, object]]] = {duration: [] for duration in durations}
	cleaning_rows: List[Dict[str, object]] = []

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
			est_rows, gt_rows, flight_time_zero = BASE.extract_bag_velocities(
				bag_path, est_raw, gt_raw, args.velocity_topic, args.gt_velocity_topic
			)

			est_clean = cleaned_dir / f"{flight_name}_vel_est_clean.csv"
			gt_clean = cleaned_dir / f"{flight_name}_vel_gt_clean.csv"
			gt_aligned = cleaned_dir / f"{flight_name}_vel_gt_clean_aligned.csv"
			est_pruned, est_filtered = BASE.clean_velocity_csv(
				est_raw, est_clean, args.clean_window, args.median_tol, args.bound_overrides,
				args.fill, args.savgol_window, args.savgol_polyorder,
			)
			gt_pruned, gt_filtered = BASE.clean_velocity_csv(
				gt_raw, gt_clean, args.clean_window, args.median_tol, args.bound_overrides,
				args.fill, args.savgol_window, args.savgol_polyorder,
			)
			BASE.align_ground_truth(est_clean, gt_clean, gt_aligned, args.gt_latency, args.tolerance)
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
				end_s = SUPPLIED_ENDPOINTS[duration_s][group][flight]
				print(
					f"Evaluating {group} {flight_name}: "
					f"{start_s:.2f} -> {end_s:.2f} s ({duration_s}s)"
				)
				output_dir = args.out_dir / f"{duration_s}s" / group
				est_window, gt_window = BASE.clip_window(
					est_clean, gt_aligned, start_s, duration_s, output_dir, flight_name
				)
				row = BASE.evaluate_window(
					est_window,
					gt_window,
					flight_name,
					start_s,
					duration_s,
					args.plot_dpi,
					flight_time_zero,
				)
				results[duration_s].append({"group": group, "bag": bag_path.name, **row})

	args.out_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(cleaning_rows).to_csv(
		args.out_dir / f"{DATASET_SLUG}_velocity_cleaning_summary.csv", index=False
	)
	for duration_s in durations:
		duration_dir = args.out_dir / f"{duration_s}s"
		summary_path = duration_dir / f"{DATASET_SLUG}_{duration_s}s_summary_by_flight.csv"
		group_path = duration_dir / f"{DATASET_SLUG}_{duration_s}s_summary_by_group.csv"
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


if __name__ == "__main__":
	run()
