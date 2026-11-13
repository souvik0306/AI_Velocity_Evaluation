#!/usr/bin/env python3
"""Extract one fixed pre-flight accelerometer and gyroscope bias per flight.

``vehicle_control_mode`` and ``estimator_sensor_bias`` use the same PX4 boot
clock. The first transition of ``flag_control_position_enabled`` from 1 to 0
is therefore used directly as the end of the bias window; no ROS-to-ULog clock
alignment is required. Biases are taken from the preceding one to two seconds,
then the median of each accelerometer and gyroscope axis is written as the fixed
initial bias.

The configured ROS-bag start and ``start - 5`` Vicon cutoff are retained in the
CSV as reference metadata, but are not used to locate the ULog bias window.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from pyulog import ULog


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
	sys.path.insert(0, str(SCRIPT_DIR))

from analyze_28th_july import FLIGHTS as FLIGHTS_28  # noqa: E402
from analyze_31st_july import FLIGHTS as FLIGHTS_31  # noqa: E402
from analyze_10th_sept_high import FLIGHTS as FLIGHTS_HIGH  # noqa: E402
from analyze_10th_sept_hover import FLIGHTS as FLIGHTS_HOVER  # noqa: E402
from analyze_10th_sept_low import FLIGHTS as FLIGHTS_LOW  # noqa: E402
from analyze_10th_sept_medium import FLIGHTS as FLIGHTS_MEDIUM  # noqa: E402
from analyze_10th_sept_super_high import FLIGHTS as FLIGHTS_SUPER_HIGH  # noqa: E402
from analyze_10th_sept_yaw import FLIGHTS as FLIGHTS_YAW  # noqa: E402


DATASETS = {
	"2026-07-28": (REPO_ROOT / "data" / "2026-07-28", FLIGHTS_28),
	"2026-07-31": (REPO_ROOT / "data" / "2026-07-31", FLIGHTS_31),
	"10th_Sept_AI_Hover_Rosbags": (
		REPO_ROOT / "data" / "10th_Sept_AI_Hover_Rosbags",
		FLIGHTS_HOVER,
	),
	"10th_Sept_AI_Yaw_Rosbags": (
		REPO_ROOT / "data" / "10th_Sept_AI_Yaw_Rosbags",
		FLIGHTS_YAW,
	),
	"10th_Sept_High": (REPO_ROOT / "data" / "10th_Sept_High", FLIGHTS_HIGH),
	"10th_Sept_Low-Dynamic": (
		REPO_ROOT / "data" / "10th_Sept_Low-Dynamic",
		FLIGHTS_LOW,
	),
	"10th_Sept_Medium": (REPO_ROOT / "data" / "10th_Sept_Medium", FLIGHTS_MEDIUM),
	"10th_Sept_Super-High": (
		REPO_ROOT / "data" / "10th_Sept_Super-High",
		FLIGHTS_SUPER_HIGH,
	),
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "initial_sensor_biases"
AXES = ("x", "y", "z")
VICON_CUTOFF_BEFORE_START_S = 5.0
DEFAULT_BIAS_WINDOW_S = 2.0


def _numeric_suffix(path: Path, prefix: str) -> int:
	match = re.search(rf"{re.escape(prefix)}(\d+)", path.stem)
	if match is None:
		raise ValueError(f"Could not find numeric suffix after {prefix!r} in {path.name}")
	return int(match.group(1))


def _find_dataset(ulog: ULog, name: str, multi_id: int = 0) -> Mapping[str, np.ndarray]:
	matches = [
		dataset.data
		for dataset in ulog.data_list
		if dataset.name == name and dataset.multi_id == multi_id
	]
	if len(matches) != 1:
		raise ValueError(
			f"Expected one {name} instance {multi_id}, found {len(matches)}"
		)
	return matches[0]


def _first_position_control_falling_edge(control_mode: Mapping[str, np.ndarray]) -> float:
	"""Return PX4 time of the first position-control 1-to-0 transition."""
	timestamps = np.asarray(control_mode["timestamp"], dtype=float) * 1e-6
	enabled = np.asarray(control_mode["flag_control_position_enabled"], dtype=bool)
	falling_edges = np.flatnonzero(enabled[:-1] & ~enabled[1:]) + 1
	if falling_edges.size == 0:
		raise ValueError(
			"No flag_control_position_enabled transition from 1 to 0 in ULog"
		)
	return float(timestamps[int(falling_edges[0])])


def _bias_medians(
	bias: Mapping[str, np.ndarray],
	window_start_px4_s: float,
	window_end_px4_s: float,
) -> Tuple[Dict[str, float], int, float, float]:
	timestamps = np.asarray(bias["timestamp"], dtype=float) * 1e-6
	mask = (timestamps >= window_start_px4_s) & (timestamps <= window_end_px4_s)
	for field in ("accel_bias_valid", "gyro_bias_valid"):
		if field in bias:
			mask &= np.asarray(bias[field], dtype=bool)

	indices = np.flatnonzero(mask)
	if indices.size == 0:
		raise ValueError(
			"No valid estimator_sensor_bias samples in stable fusion window "
			f"{window_start_px4_s:.6f} -> {window_end_px4_s:.6f} s"
		)

	values: Dict[str, float] = {}
	for axis_index, axis_name in enumerate(AXES):
		accel = np.asarray(bias[f"accel_bias[{axis_index}]"], dtype=float)[indices]
		if not np.all(np.isfinite(accel)):
			raise ValueError("Non-finite estimator_sensor_bias value in selected window")
		values[f"accel_bias_{axis_name}_mps2"] = float(np.median(accel))
	for axis_index, axis_name in enumerate(AXES):
		gyro = np.asarray(bias[f"gyro_bias[{axis_index}]"], dtype=float)[indices]
		if not np.all(np.isfinite(gyro)):
			raise ValueError("Non-finite estimator_sensor_bias value in selected window")
		values[f"gyro_bias_{axis_name}_radps"] = float(np.median(gyro))

	return values, int(indices.size), float(timestamps[indices[0]]), float(timestamps[indices[-1]])


def _pair_ulg_files(
	dataset_name: str,
	group_dir: Path,
	flights: Mapping[int, Mapping[str, object]],
) -> Dict[int, Path]:
	ulg_paths = list(group_dir.glob("*.ulg"))
	if dataset_name == "2026-07-31":
		by_number = {_numeric_suffix(path, "log_"): path for path in ulg_paths}
		missing = sorted(set(flights) - set(by_number))
		if missing:
			raise FileNotFoundError(f"Missing ULog files for flight numbers {missing} in {group_dir}")
		return {flight: by_number[flight] for flight in flights}

	# The 28 July bag names contain wall-clock timestamps and the ULogs contain
	# logger sequence numbers. Their chronological order is the recorded pairing;
	# the RAW logger sequence legitimately skips log_3.
	ordered_ulgs = sorted(ulg_paths, key=lambda path: _numeric_suffix(path, "log_"))
	if len(ordered_ulgs) != len(flights):
		raise ValueError(
			f"Expected {len(flights)} ULogs for {group_dir}, found {len(ordered_ulgs)}"
		)
	return dict(zip(flights, ordered_ulgs))


def extract_flight(
	dataset_name: str,
	group: str,
	flight_number: int,
	config: Mapping[str, object],
	bag_path: Path,
	ulg_path: Path,
	window_s: float,
) -> Dict[str, object]:
	start_rosbag_s = float(config["start"])
	cutoff_rosbag_s = start_rosbag_s - VICON_CUTOFF_BEFORE_START_S

	ulog = ULog(
		str(ulg_path),
		message_name_filter_list=["estimator_sensor_bias", "vehicle_control_mode"],
	)
	bias = _find_dataset(ulog, "estimator_sensor_bias")
	control_mode = _find_dataset(ulog, "vehicle_control_mode")
	window_end_s = _first_position_control_falling_edge(control_mode)
	window_start_s = window_end_s - window_s
	ulog_start_s = float(ulog.start_timestamp) * 1e-6
	medians, bias_samples, bias_first_s, bias_last_s = _bias_medians(
		bias, window_start_s, window_end_s
	)

	return {
		"dataset": dataset_name,
		"group": group,
		"flight_name": bag_path.stem,
		"flight_number": flight_number,
		"bag_file": bag_path.name,
		"ulg_file": ulg_path.name,
		"start_rosbag_s": start_rosbag_s,
		"vicon_cutoff_rosbag_s": cutoff_rosbag_s,
		"ulg_window_anchor": "vehicle_control_mode.flag_control_position_enabled_1_to_0",
		"ulog_start_px4_s": ulog_start_s,
		"position_control_falling_edge_px4_s": window_end_s,
		"position_control_falling_edge_relative_s": window_end_s - ulog_start_s,
		"bias_window_start_px4_s": window_start_s,
		"bias_window_end_px4_s": window_end_s,
		"bias_window_start_relative_s": window_start_s - ulog_start_s,
		"bias_window_end_relative_s": window_end_s - ulog_start_s,
		"bias_first_sample_px4_s": bias_first_s,
		"bias_last_sample_px4_s": bias_last_s,
		"bias_samples": bias_samples,
		**medians,
	}


def extract_all(
	datasets: Mapping[str, Tuple[Path, Mapping[str, Mapping[int, Mapping[str, object]]]]],
	window_s: float,
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
	rows: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
	for dataset_name, (dataset_dir, groups) in datasets.items():
		rows[dataset_name] = {group: [] for group in groups}
		for group, flights in groups.items():
			group_dir = dataset_dir / group
			ulg_by_flight = _pair_ulg_files(dataset_name, group_dir, flights)
			for flight_number, config in flights.items():
				bag_path = group_dir / str(config["bag"])
				if not bag_path.exists():
					raise FileNotFoundError(f"ROS bag not found: {bag_path}")
				print(f"Extracting {dataset_name} {group} {bag_path.stem}")
				rows[dataset_name][group].append(
					extract_flight(
						dataset_name,
						group,
						flight_number,
						config,
						bag_path,
						ulg_by_flight[flight_number],
						window_s,
					)
				)
	return rows


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--out-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
	)
	parser.add_argument(
		"--bias-window",
		type=float,
		default=DEFAULT_BIAS_WINDOW_S,
		help="Stable-fusion window length in seconds (default: 2.0; must be 1.0 to 2.0)",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_arguments()
	if not 1.0 <= args.bias_window <= 2.0:
		raise ValueError("--bias-window must be between 1.0 and 2.0 seconds")

	rows = extract_all(DATASETS, args.bias_window)
	args.out_dir.mkdir(parents=True, exist_ok=True)
	for dataset_name, dataset_rows in rows.items():
		dataset_output_dir = args.out_dir / dataset_name
		dataset_output_dir.mkdir(parents=True, exist_ok=True)
		for source_group, group_rows in dataset_rows.items():
			output_group = "AI" if source_group == "UN" else source_group
			output_path = dataset_output_dir / f"initial_sensor_bias_{output_group}.csv"
			output_df = pd.DataFrame(group_rows)
			output_df["group"] = output_group
			output_df.to_csv(output_path, index=False, float_format="%.9f")
			print(f"Saved {len(group_rows)} flights to {output_path}")


if __name__ == "__main__":
	main()
