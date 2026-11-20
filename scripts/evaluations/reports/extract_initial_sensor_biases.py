#!/usr/bin/env python3
"""Extract one fixed pre-flight accelerometer and gyroscope bias per flight.

The final recorded timestamp on ``/mavros/vision_pose/pose`` is the cutoff.
Nearby MAVROS timesync samples convert that ROS timestamp to PX4 boot time so
the matching preceding one-to-two-second ``estimator_sensor_bias`` window can
be selected. The CSV contains both bag-relative and exact ROS epoch cutoff time.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
import rosbag
from pyulog import ULog


EVALUATIONS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EVALUATIONS_DIR.parents[1]
if str(EVALUATIONS_DIR) not in sys.path:
	sys.path.insert(0, str(EVALUATIONS_DIR))

from flight_sessions.july_28.analyze import FLIGHTS as FLIGHTS_28  # noqa: E402
from flight_sessions.july_31.analyze import FLIGHTS as FLIGHTS_31  # noqa: E402
from flight_sessions.october_14.high import FLIGHTS as FLIGHTS_14_HIGH  # noqa: E402
from flight_sessions.october_14.low import FLIGHTS as FLIGHTS_14_LOW  # noqa: E402
from flight_sessions.october_14.medium import FLIGHTS as FLIGHTS_14_MEDIUM  # noqa: E402
from flight_sessions.october_15.high import FLIGHTS as FLIGHTS_15_HIGH  # noqa: E402
from flight_sessions.october_15.low import FLIGHTS as FLIGHTS_15_LOW  # noqa: E402
from flight_sessions.october_15.medium import FLIGHTS as FLIGHTS_15_MEDIUM  # noqa: E402
from flight_sessions.september_10.high import FLIGHTS as FLIGHTS_HIGH  # noqa: E402
from flight_sessions.september_10.hover import FLIGHTS as FLIGHTS_HOVER  # noqa: E402
from flight_sessions.september_10.low import FLIGHTS as FLIGHTS_LOW  # noqa: E402
from flight_sessions.september_10.medium import FLIGHTS as FLIGHTS_MEDIUM  # noqa: E402
from flight_sessions.september_10.super_high import FLIGHTS as FLIGHTS_SUPER_HIGH  # noqa: E402
from flight_sessions.september_10.yaw import FLIGHTS as FLIGHTS_YAW  # noqa: E402


FLIGHT_SESSIONS_DIR = REPO_ROOT / "data" / "flight_sessions"

DATASETS = {
	"2026-07-28": (FLIGHT_SESSIONS_DIR / "2026-07-28", FLIGHTS_28),
	"2026-07-31": (FLIGHT_SESSIONS_DIR / "2026-07-31", FLIGHTS_31),
	"10th_Sept_AI_Hover_Rosbags": (
		FLIGHT_SESSIONS_DIR / "10th_Sept" / "AI_Hover_Rosbags",
		FLIGHTS_HOVER,
	),
	"10th_Sept_AI_Yaw_Rosbags": (
		FLIGHT_SESSIONS_DIR / "10th_Sept" / "AI_Yaw_Rosbags",
		FLIGHTS_YAW,
	),
	"10th_Sept_High": (FLIGHT_SESSIONS_DIR / "10th_Sept" / "High", FLIGHTS_HIGH),
	"10th_Sept_Low-Dynamic": (
		FLIGHT_SESSIONS_DIR / "10th_Sept" / "Low-Dynamic",
		FLIGHTS_LOW,
	),
	"10th_Sept_Medium": (FLIGHT_SESSIONS_DIR / "10th_Sept" / "Medium", FLIGHTS_MEDIUM),
	"10th_Sept_Super-High": (
		FLIGHT_SESSIONS_DIR / "10th_Sept" / "Super-High",
		FLIGHTS_SUPER_HIGH,
	),
	"14th_Oct_Low": (FLIGHT_SESSIONS_DIR / "14th_Oct" / "Low", FLIGHTS_14_LOW),
	"14th_Oct_Medium": (
		FLIGHT_SESSIONS_DIR / "14th_Oct" / "Medium",
		FLIGHTS_14_MEDIUM,
	),
	"14th_Oct_High": (FLIGHT_SESSIONS_DIR / "14th_Oct" / "High", FLIGHTS_14_HIGH),
	"15th_Oct_Low": (FLIGHT_SESSIONS_DIR / "15th_Oct" / "Low", FLIGHTS_15_LOW),
	"15th_Oct_Medium": (
		FLIGHT_SESSIONS_DIR / "15th_Oct" / "Medium",
		FLIGHTS_15_MEDIUM,
	),
	"15th_Oct_High": (FLIGHT_SESSIONS_DIR / "15th_Oct" / "High", FLIGHTS_15_HIGH),
}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "reports" / "initial_sensor_biases"
AXES = ("x", "y", "z")
DEFAULT_BIAS_WINDOW_S = 2.0
OUTPUT_COLUMNS = [
	"group",
	"flight_number",
	"bag_file",
	"ulg_file",
	"cutoff_rosbag_s",
	"cutoff_ros_epoch_s",
	"bias_samples",
	"accel_bias_x_mps2",
	"accel_bias_y_mps2",
	"accel_bias_z_mps2",
	"gyro_bias_x_radps",
	"gyro_bias_y_radps",
	"gyro_bias_z_radps",
]


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


def _vision_pose_cutoff(
	bag_path: Path,
	px4_min_s: float,
	px4_max_s: float,
	nearby_radius_s: float = 1.0,
) -> Tuple[float, float, float, int]:
	"""Return vision cutoff in ROS epoch, bag-relative, and PX4 boot time.

	MAVROS reports ``ros_epoch_ns = remote_px4_ns + estimated_offset_ns``.
	The median offset from timesync samples within ``nearby_radius_s`` of the
	final vision-pose message is used to convert the cutoff to PX4 time.
	"""
	remote_times = []
	offsets = []
	vision_times = []
	with rosbag.Bag(str(bag_path), "r") as bag:
		bag_start_epoch_s = float(bag.get_start_time())
		for topic, message, bag_stamp in bag.read_messages(
			topics=["/mavros/vision_pose/pose", "/mavros/timesync_status"]
		):
			if topic == "/mavros/vision_pose/pose":
				vision_times.append(float(bag_stamp.to_sec()))
				continue
			remote_s = float(message.remote_timestamp_ns) * 1e-9
			offset_s = float(message.estimated_offset_ns) * 1e-9
			if np.isfinite(remote_s) and np.isfinite(offset_s):
				remote_times.append(remote_s)
				offsets.append(offset_s)

	if not vision_times:
		raise ValueError(f"No /mavros/vision_pose/pose samples in {bag_path}")
	if not offsets:
		raise ValueError(f"No finite /mavros/timesync_status samples in {bag_path}")
	remote_array = np.asarray(remote_times, dtype=float)
	offset_array = np.asarray(offsets, dtype=float)
	timesync_ros_times = remote_array + offset_array

	# Select the final vision message belonging to the same PX4 boot session as
	# the paired ULog. This rejects a later vision segment after an FMU reboot.
	vision_array = np.asarray(vision_times, dtype=float)
	right = np.searchsorted(timesync_ros_times, vision_array, side="left")
	right = np.clip(right, 0, timesync_ros_times.size - 1)
	left = np.clip(right - 1, 0, timesync_ros_times.size - 1)
	use_left = (
		np.abs(vision_array - timesync_ros_times[left])
		<= np.abs(vision_array - timesync_ros_times[right])
	)
	nearest = np.where(use_left, left, right)
	mapped_px4 = vision_array - offset_array[nearest]
	same_boot = (mapped_px4 >= px4_min_s) & (mapped_px4 <= px4_max_s)
	valid_indices = np.flatnonzero(same_boot)
	if valid_indices.size == 0:
		raise ValueError(f"No vision-pose cutoff maps into ULog time range for {bag_path}")
	cutoff_ros_epoch_s = float(vision_array[int(valid_indices[-1])])

	nearby = np.abs(timesync_ros_times - cutoff_ros_epoch_s) <= nearby_radius_s
	same_boot_timesync = (remote_array >= px4_min_s) & (remote_array <= px4_max_s)
	nearby &= same_boot_timesync
	if not np.any(nearby):
		eligible = np.flatnonzero(same_boot_timesync)
		if eligible.size == 0:
			raise ValueError(f"No timesync samples map into ULog time range for {bag_path}")
		nearest_eligible = eligible[
			int(np.argmin(np.abs(timesync_ros_times[eligible] - cutoff_ros_epoch_s)))
		]
		nearby[int(nearest_eligible)] = True
	median_offset_s = float(np.median(offset_array[nearby]))
	cutoff_rosbag_s = cutoff_ros_epoch_s - bag_start_epoch_s
	cutoff_px4_s = cutoff_ros_epoch_s - median_offset_s
	return cutoff_ros_epoch_s, cutoff_rosbag_s, cutoff_px4_s, int(nearby.sum())


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
	if dataset_name == "2026-07-31" or "_Oct_" in dataset_name:
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
	ulog = ULog(
		str(ulg_path),
		message_name_filter_list=["estimator_sensor_bias"],
	)
	bias = _find_dataset(ulog, "estimator_sensor_bias")
	bias_timestamps = np.asarray(bias["timestamp"], dtype=float) * 1e-6
	cutoff_ros_epoch_s, cutoff_rosbag_s, window_end_s, _ = (
		_vision_pose_cutoff(
			bag_path,
			float(bias_timestamps[0]),
			float(bias_timestamps[-1]),
		)
	)
	window_start_s = window_end_s - window_s
	medians, bias_samples, _, _ = _bias_medians(
		bias, window_start_s, window_end_s
	)

	return {
		"group": group,
		"flight_number": flight_number,
		"bag_file": bag_path.name,
		"ulg_file": ulg_path.name,
		"cutoff_ros_epoch_s": cutoff_ros_epoch_s,
		"cutoff_rosbag_s": cutoff_rosbag_s,
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
			output_df[OUTPUT_COLUMNS].to_csv(
				output_path,
				index=False,
				float_format="%.9f",
			)
			print(f"Saved {len(group_rows)} flights to {output_path}")


if __name__ == "__main__":
	main()
