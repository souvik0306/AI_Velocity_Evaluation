#!/usr/bin/env python3
"""Evaluate yaw flights with Vicon-referenced body-frame velocity errors."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from scipy.spatial.transform import Rotation

import evaluation_common as common
from analyze_yaw import BAGS_DIR, FLIGHTS


DEFAULT_POSE_TOPIC = "/vrpn_client_node/AIIMU1/pose"
QUATERNION_COLUMNS = ["quat_x", "quat_y", "quat_z", "quat_w"]
BODY_METRIC_COLUMNS = [
	"rmse_forward_mps",
	"rmse_lateral_mps",
	"rmse_vertical_body_mps",
	"bias_corrected_rmse_forward_mps",
	"bias_corrected_rmse_lateral_mps",
	"bias_corrected_rmse_vertical_body_mps",
]


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Evaluate yaw bags with body-frame velocity metrics and plots."
	)
	parser.add_argument("--bags_dir", default=str(BAGS_DIR))
	parser.add_argument(
		"--out_dir",
		default=str(common.REPO_ROOT / "results" / "yaw_body_frame"),
	)
	parser.add_argument("--group", choices=["all", *FLIGHTS], default="all")
	parser.add_argument("--duration", choices=["20s"], default="20s")
	parser.add_argument("--velocity_topic", default=common.DEFAULT_EST_TOPIC)
	parser.add_argument("--gt_velocity_topic", default=common.DEFAULT_GT_TOPIC)
	parser.add_argument("--pose_topic", default=DEFAULT_POSE_TOPIC)
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
	parser.add_argument("--savgol_window", type=int, default=21)
	parser.add_argument("--savgol_polyorder", type=int, default=2)
	parser.add_argument("--gt_latency", type=float, default=0.025)
	parser.add_argument("--tolerance", type=float, default=None)
	args = parser.parse_args()
	args.bags_dir = common.resolve_path(args.bags_dir)
	args.out_dir = common.resolve_path(args.out_dir)
	args.clean_window = common.CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = common.CLEANER._parse_bound_overrides(
		args.bound or common.DEFAULT_VELOCITY_BOUNDS
	)
	if args.savgol_window <= 0:
		args.savgol_window = None
	return args


def extract_vicon_pose(bag_path: Path, output_path: Path, pose_topic: str) -> int:
	"""Extract Vicon orientation using each pose message's header timestamp."""
	rows = []
	with common.CONVERTER.rosbag.Bag(str(bag_path), "r") as bag:
		for _, message, _bag_stamp in bag.read_messages(topics=[pose_topic]):
			time = common.CONVERTER.header_time_to_sec(message)
			if not hasattr(message, "pose") or not hasattr(message.pose, "orientation"):
				raise AttributeError(f"Message on {pose_topic} has no pose.orientation")
			orientation = message.pose.orientation
			rows.append({
				"time": time,
				"quat_x": orientation.x,
				"quat_y": orientation.y,
				"quat_z": orientation.z,
				"quat_w": orientation.w,
			})
	if not rows:
		raise ValueError(f"No Vicon pose messages found on {pose_topic} in {bag_path}")
	output_path.parent.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(rows, columns=["time", *QUATERNION_COLUMNS]).sort_values(
		"time"
	).to_csv(output_path, index=False)
	return len(rows)


def match_pose_to_cleaned_gt(
	gt_clean_path: Path,
	pose_path: Path,
	output_path: Path,
) -> Path:
	"""Nearest-match pose to cleaned GT velocity without cleaning quaternions."""
	gt = common.load_velocity_csv(
		gt_clean_path, ["time", "vel_x", "vel_y", "vel_z"]
	)
	pose = pd.read_csv(pose_path)
	missing = [column for column in ["time", *QUATERNION_COLUMNS] if column not in pose]
	if missing:
		raise ValueError(f"Missing pose columns {missing} in {pose_path}")
	pose = pose.copy()
	for column in ["time", *QUATERNION_COLUMNS]:
		pose[column] = pd.to_numeric(pose[column], errors="coerce")
	pose = pose.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	if pose.empty:
		raise ValueError(f"No timestamped pose samples in {pose_path}")
	pose = pose.rename(columns={"time": "pose_time"})
	combined = pd.merge_asof(
		gt.sort_values("time"),
		pose.sort_values("pose_time"),
		left_on="time",
		right_on="pose_time",
		direction="nearest",
	)
	combined["pose_time_difference_s"] = combined["pose_time"] - combined["time"]

	quaternions = combined[QUATERNION_COLUMNS].to_numpy(dtype=float)
	norms = np.linalg.norm(quaternions, axis=1)
	valid = np.isfinite(quaternions).all(axis=1) & np.isfinite(norms) & (norms > 0.0)
	if not valid.any():
		raise ValueError(f"No valid matched Vicon quaternions for {gt_clean_path}")
	combined.loc[:, QUATERNION_COLUMNS] = np.nan
	combined.loc[valid, QUATERNION_COLUMNS] = (
		quaternions[valid] / norms[valid, np.newaxis]
	)
	combined = combined.drop(columns=["pose_time"])
	output_path.parent.mkdir(parents=True, exist_ok=True)
	combined.to_csv(output_path, index=False)
	return output_path


def calculate_body_frame_metrics(
	est_window_path: Path,
	gt_window_path: Path,
	output_dir: Path,
) -> Tuple[Dict[str, float], Path]:
	"""Calculate direct and bias-corrected body-frame errors and RMSE."""
	est = common.METRICS._load_velocity_csv(est_window_path)
	gt = common.METRICS._load_velocity_csv(gt_window_path)
	merged = pd.merge(est, gt, on="time", how="inner", suffixes=("_est", "_gt"))
	if merged.empty:
		raise ValueError("No overlapping timestamps between estimate and GT CSVs")
	merged = common.METRICS._compute_bias_errors(merged)

	missing = [
		column
		for column in [*QUATERNION_COLUMNS, "pose_time_difference_s"]
		if column not in merged
	]
	if missing:
		raise ValueError(f"Missing body-frame GT columns after alignment: {missing}")
	required = [
		*[f"vel_{axis}_est" for axis in "xyz"],
		*[f"vel_{axis}_gt" for axis in "xyz"],
		*[f"err_{axis}" for axis in "xyz"],
		*[f"err_{axis}_bias" for axis in "xyz"],
		*QUATERNION_COLUMNS,
		"pose_time_difference_s",
	]
	values = merged[required].to_numpy(dtype=float)
	quaternions = merged[QUATERNION_COLUMNS].to_numpy(dtype=float)
	norms = np.linalg.norm(quaternions, axis=1)
	valid = np.isfinite(values).all(axis=1) & np.isfinite(norms) & (norms > 0.0)
	body = merged.loc[valid].copy()
	if body.empty:
		raise ValueError("No valid samples remain for body-frame calculation")
	quaternions = quaternions[valid] / norms[valid, np.newaxis]
	body.loc[:, QUATERNION_COLUMNS] = quaternions
	rotation_gt = Rotation.from_quat(quaternions)

	error_world = body[[f"err_{axis}" for axis in "xyz"]].to_numpy(dtype=float)
	error_body = rotation_gt.inv().apply(error_world)
	error_world_bias = body[[f"err_{axis}_bias" for axis in "xyz"]].to_numpy(dtype=float)
	error_body_bias = rotation_gt.inv().apply(error_world_bias)
	gt_world = body[[f"vel_{axis}_gt" for axis in "xyz"]].to_numpy(dtype=float)
	gt_body = rotation_gt.inv().apply(gt_world)

	body["err_world_x"] = error_world[:, 0]
	body["err_world_y"] = error_world[:, 1]
	body["err_world_z"] = error_world[:, 2]
	body["err_forward"] = error_body[:, 0]
	body["err_lateral"] = error_body[:, 1]
	body["err_vertical"] = error_body[:, 2]
	body["err_forward_bias"] = error_body_bias[:, 0]
	body["err_lateral_bias"] = error_body_bias[:, 1]
	body["err_vertical_bias"] = error_body_bias[:, 2]
	body["gt_forward_velocity"] = gt_body[:, 0]
	body["gt_lateral_velocity"] = gt_body[:, 1]
	body["gt_vertical_velocity"] = gt_body[:, 2]

	metrics = {
		"rmse_forward_mps": common.METRICS._rmse(body["err_forward"]),
		"rmse_lateral_mps": common.METRICS._rmse(body["err_lateral"]),
		"rmse_vertical_body_mps": common.METRICS._rmse(body["err_vertical"]),
		"bias_corrected_rmse_forward_mps": common.METRICS._rmse(
			body["err_forward_bias"]
		),
		"bias_corrected_rmse_lateral_mps": common.METRICS._rmse(
			body["err_lateral_bias"]
		),
		"bias_corrected_rmse_vertical_body_mps": common.METRICS._rmse(
			body["err_vertical_bias"]
		),
	}
	metrics = {name: round(value, 6) for name, value in metrics.items()}
	body_path = output_dir / f"{est_window_path.stem}_body_frame_errors.csv"
	body_columns = [
		"time",
		*[f"vel_{axis}_est" for axis in "xyz"],
		*[f"vel_{axis}_gt" for axis in "xyz"],
		*QUATERNION_COLUMNS,
		"pose_time_difference_s",
		*[f"err_world_{axis}" for axis in "xyz"],
		"err_forward", "err_lateral", "err_vertical",
		"err_forward_bias", "err_lateral_bias", "err_vertical_bias",
		"gt_forward_velocity", "gt_lateral_velocity", "gt_vertical_velocity",
	]
	body[body_columns].to_csv(body_path, index=False)
	return metrics, body_path


def style_axis(axis) -> None:
	axis.xaxis.set_major_locator(MaxNLocator(nbins=12))
	axis.xaxis.set_minor_locator(AutoMinorLocator(2))
	axis.yaxis.set_major_locator(MaxNLocator(nbins=8))
	axis.yaxis.set_minor_locator(AutoMinorLocator(2))
	axis.grid(True, which="major", alpha=0.75)
	axis.grid(True, which="minor", alpha=0.25)
	axis.tick_params(axis="both", which="major", labelsize=14)


def plot_error_with_gt_velocity(
	body: pd.DataFrame,
	component: str,
	output_dir: Path,
	prefix: str,
	dpi: int,
	flight_time_zero: Optional[float],
) -> Path:
	time_start = float(body["time"].iloc[0])
	time_end = float(body["time"].iloc[-1])
	relative_time = body["time"].to_numpy(dtype=float) - time_start
	fig, axis = plt.subplots(figsize=(16, 8.5))
	axis.plot(
		relative_time,
		body[f"err_{component}"].to_numpy(dtype=float),
		label=f"{component.capitalize()} velocity error",
		color="#1f42b4",
		linewidth=3.0,
	)
	axis.plot(
		relative_time,
		body[f"gt_{component}_velocity"].to_numpy(dtype=float),
		label=f"GT {component} velocity",
		color="#2ca02c",
		linewidth=3.0,
		alpha=0.85,
	)
	axis.axhline(0.0, color="black", linewidth=1.2, alpha=0.6)
	axis.set_title(
		f"Body-frame {component.capitalize()} Error and GT Velocity",
		fontsize=21,
	)
	if flight_time_zero is None:
		xlabel = "Time from evaluation-window start (s)"
	else:
		xlabel = (
			"Time from evaluation-window start (s) "
			f"[flight time: {time_start - flight_time_zero:.2f} → "
			f"{time_end - flight_time_zero:.2f} s]"
		)
	axis.set_xlabel(xlabel, fontsize=17)
	axis.set_ylabel("Velocity / error (m/s)", fontsize=17)
	axis.legend(loc="best", fontsize=14)
	style_axis(axis)
	fig.tight_layout()
	path = output_dir / f"{prefix}_{component}_error_and_gt_velocity.png"
	fig.savefig(path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	return path


def plot_gt_body_velocity_validation(
	body: pd.DataFrame,
	output_dir: Path,
	prefix: str,
	dpi: int,
) -> Path:
	time_start = float(body["time"].iloc[0])
	relative_time = body["time"].to_numpy(dtype=float) - time_start
	fig, axis = plt.subplots(figsize=(16, 8.5))
	axis.plot(
		relative_time,
		body["gt_forward_velocity"].to_numpy(dtype=float),
		label="GT forward velocity",
		color="#d62728",
		linewidth=3.0,
	)
	axis.plot(
		relative_time,
		body["gt_lateral_velocity"].to_numpy(dtype=float),
		label="GT lateral velocity",
		color="#9467bd",
		linewidth=3.0,
	)
	axis.axhline(0.0, color="black", linewidth=1.2, alpha=0.6)
	axis.set_title("GT Body-frame Velocity Axis Validation", fontsize=21)
	axis.set_xlabel("Time from evaluation-window start (s)", fontsize=17)
	axis.set_ylabel("GT velocity (m/s)", fontsize=17)
	axis.legend(loc="best", fontsize=14)
	style_axis(axis)
	fig.tight_layout()
	path = output_dir / f"{prefix}_gt_forward_lateral_velocity.png"
	fig.savefig(path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)
	return path


def create_body_frame_plots(
	body_path: Path,
	output_dir: Path,
	dpi: int,
	flight_time_zero: float,
) -> List[Path]:
	body = pd.read_csv(body_path)
	required = [
		"time", "err_forward", "err_lateral",
		"gt_forward_velocity", "gt_lateral_velocity",
	]
	missing = [column for column in required if column not in body]
	if missing:
		raise ValueError(f"Missing body plot columns {missing} in {body_path}")
	body = body.dropna(subset=required).sort_values("time").reset_index(drop=True)
	if body.empty:
		raise ValueError(f"No valid body-frame plot rows in {body_path}")
	output_dir.mkdir(parents=True, exist_ok=True)
	prefix = body_path.stem.replace("_body_frame_errors", "")
	return [
		plot_error_with_gt_velocity(
			body, "forward", output_dir, prefix, dpi, flight_time_zero
		),
		plot_error_with_gt_velocity(
			body, "lateral", output_dir, prefix, dpi, flight_time_zero
		),
		plot_gt_body_velocity_validation(body, output_dir, prefix, dpi),
	]


def summarize_group(rows: List[Dict[str, object]], group: str) -> Dict[str, object]:
	result = common.summarize_duration(rows)
	frame = pd.DataFrame(rows)
	for column in BODY_METRIC_COLUMNS:
		result[f"mean_{column}"] = float(
			pd.to_numeric(frame[column], errors="coerce").mean()
		)
	return {"group": group, **result}


def evaluate_yaw_body_frame() -> List[Dict[str, object]]:
	args = parse_arguments()
	common.validate_configuration(FLIGHTS)
	groups = common.selected_groups(args.group, FLIGHTS)
	durations = common.selected_durations(args.duration)
	results: Dict[int, List[Dict[str, object]]] = {
		duration: [] for duration in durations
	}
	cleaning_rows: List[Dict[str, object]] = []

	for group in groups:
		for flight, config in FLIGHTS[group].items():
			flight_name = f"flight_{flight}"
			bag_path = args.bags_dir / group / str(config["bag"])
			if not bag_path.exists():
				raise FileNotFoundError(f"ROS bag not found: {bag_path}")
			print(f"Extracting yaw body-frame data for {group} {flight_name}: {bag_path.name}")

			extracted_dir = args.out_dir / "extracted_csv" / group
			cleaned_dir = args.out_dir / "cleaned_csv" / group
			est_raw = extracted_dir / f"{flight_name}_vel_est.csv"
			gt_raw = extracted_dir / f"{flight_name}_vel_gt.csv"
			pose_raw = extracted_dir / f"{flight_name}_pose_gt.csv"
			est_rows, gt_rows, flight_time_zero = common.extract_bag_velocities(
				bag_path,
				est_raw,
				gt_raw,
				args.velocity_topic,
				args.gt_velocity_topic,
			)
			pose_rows = extract_vicon_pose(bag_path, pose_raw, args.pose_topic)

			est_clean = cleaned_dir / f"{flight_name}_vel_est_clean.csv"
			gt_clean = cleaned_dir / f"{flight_name}_vel_gt_clean.csv"
			gt_with_pose = cleaned_dir / f"{flight_name}_vel_gt_clean_with_pose.csv"
			gt_aligned = cleaned_dir / f"{flight_name}_vel_gt_clean_aligned.csv"
			est_pruned, est_filtered = common.clean_velocity_csv(
				est_raw, est_clean, args.clean_window, args.median_tol,
				args.bound_overrides, args.fill, args.savgol_window,
				args.savgol_polyorder,
			)
			gt_pruned, gt_filtered = common.clean_velocity_csv(
				gt_raw, gt_clean, args.clean_window, args.median_tol,
				args.bound_overrides, args.fill, args.savgol_window,
				args.savgol_polyorder,
			)
			match_pose_to_cleaned_gt(gt_clean, pose_raw, gt_with_pose)
			common.align_ground_truth(
				est_clean, gt_with_pose, gt_aligned, args.gt_latency, args.tolerance
			)
			cleaning_rows.append({
				"group": group,
				"flight": flight_name,
				"bag": bag_path.name,
				"estimate_extracted_rows": est_rows,
				"gt_extracted_rows": gt_rows,
				"pose_extracted_rows": pose_rows,
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
				est_window, gt_window = common.clip_window(
					est_clean, gt_aligned, start_s, duration_s,
					output_dir, flight_name,
				)
				row = common.evaluate_window(
					est_window, gt_window, flight_name, start_s, duration_s,
					args.plot_dpi, flight_time_zero,
				)
				body_metrics, body_path = calculate_body_frame_metrics(
					est_window, gt_window, output_dir
				)
				body_plots_dir = output_dir / "plots" / f"{flight_name}_body_frame"
				body_plot_paths = create_body_frame_plots(
					body_path, body_plots_dir, args.plot_dpi, flight_time_zero
				)
				row.update({
					**body_metrics,
					"body_frame_errors_file": str(body_path),
					"body_frame_plots_dir": str(body_plots_dir),
					"forward_error_plot": str(body_plot_paths[0]),
					"lateral_error_plot": str(body_plot_paths[1]),
					"gt_body_velocity_validation_plot": str(body_plot_paths[2]),
				})
				results[duration_s].append({
					"group": group,
					"bag": bag_path.name,
					**row,
				})

	args.out_dir.mkdir(parents=True, exist_ok=True)
	pd.DataFrame(cleaning_rows).to_csv(
		args.out_dir / "yaw_body_frame_velocity_cleaning_summary.csv",
		index=False,
	)
	for duration_s in durations:
		duration_dir = args.out_dir / f"{duration_s}s"
		duration_dir.mkdir(parents=True, exist_ok=True)
		flight_path = duration_dir / f"yaw_body_frame_{duration_s}s_summary_by_flight.csv"
		group_path = duration_dir / f"yaw_body_frame_{duration_s}s_summary_by_group.csv"
		pd.DataFrame(results[duration_s]).to_csv(flight_path, index=False)
		pd.DataFrame([
			summarize_group(
				[row for row in results[duration_s] if row["group"] == group],
				group,
			)
			for group in groups
		]).to_csv(group_path, index=False)
		print(f"Saved {flight_path}")
		print(f"Saved {group_path}")
	return results[20]


if __name__ == "__main__":
	evaluate_yaw_body_frame()
