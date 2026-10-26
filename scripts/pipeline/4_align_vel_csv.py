#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


DEFAULT_ALIGNMENT_TOLERANCE_S = 0.025


def _load_sorted_csv(path: Path, time_col: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	if time_col not in df.columns:
		raise ValueError(f"Missing required column '{time_col}' in {path}")

	df = df.copy()
	df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
	df = df.dropna(subset=[time_col]).sort_values(time_col)
	return df


def _apply_latency(df: pd.DataFrame, time_col: str, latency_s: float) -> pd.DataFrame:
	if latency_s == 0:
		return df
	shifted = df.copy()
	shifted[time_col] = shifted[time_col] - latency_s
	return shifted.sort_values(time_col)


def interpolate_ground_truth_to_timestamps(
	df_gt: pd.DataFrame,
	target_times: pd.Series,
	time_col: str = "time",
	tolerance_s: Optional[float] = DEFAULT_ALIGNMENT_TOLERANCE_S,
) -> pd.DataFrame:
	"""Linearly interpolate GT onto target timestamps without extrapolation."""
	if tolerance_s is not None and tolerance_s < 0:
		raise ValueError("Alignment tolerance must be non-negative")

	gt = df_gt.copy()
	gt[time_col] = pd.to_numeric(gt[time_col], errors="coerce")
	gt = gt.dropna(subset=[time_col]).sort_values(time_col)
	gt = gt.drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)
	target = pd.to_numeric(target_times, errors="coerce").to_numpy(dtype=float)
	result = pd.DataFrame({time_col: target})
	if gt.empty:
		for col in gt.columns:
			if col != time_col:
				result[col] = np.nan
		return result

	source_times = gt[time_col].to_numpy(dtype=float)
	right_indices = np.searchsorted(source_times, target, side="left")
	exact = np.zeros(target.size, dtype=bool)
	in_bounds = right_indices < source_times.size
	exact[in_bounds] = source_times[right_indices[in_bounds]] == target[in_bounds]
	bracketed = (~exact) & (right_indices > 0) & (right_indices < source_times.size)
	left_indices = np.maximum(right_indices - 1, 0)
	safe_right_indices = np.minimum(right_indices, source_times.size - 1)

	if tolerance_s is not None:
		left_distance = target - source_times[left_indices]
		right_distance = source_times[safe_right_indices] - target
		bracketed &= (
			(left_distance <= tolerance_s + np.finfo(float).eps)
			& (right_distance <= tolerance_s + np.finfo(float).eps)
		)

	for col in gt.columns:
		if col == time_col:
			continue
		source_values = pd.to_numeric(gt[col], errors="coerce").to_numpy(dtype=float)
		aligned_values = np.full(target.size, np.nan, dtype=float)
		if exact.any():
			aligned_values[exact] = source_values[right_indices[exact]]

		valid = bracketed.copy()
		if valid.any():
			left_values = source_values[left_indices[valid]]
			right_values = source_values[safe_right_indices[valid]]
			left_times = source_times[left_indices[valid]]
			right_times = source_times[safe_right_indices[valid]]
			finite_endpoints = np.isfinite(left_values) & np.isfinite(right_values)
			fractions = (target[valid] - left_times) / (right_times - left_times)
			interpolated = left_values + fractions * (right_values - left_values)
			valid_positions = np.flatnonzero(valid)
			aligned_values[valid_positions[finite_endpoints]] = interpolated[finite_endpoints]
		result[col] = aligned_values

	return result[df_gt.columns]


def _derive_output_path(
	gt_path: Path,
	out_path: Optional[Path],
	out_dir: Optional[Path],
) -> Path:
	if out_path is not None:
		return out_path
	if out_dir is not None:
		return out_dir / f"{gt_path.stem}_aligned{gt_path.suffix}"
	return gt_path.with_name(f"{gt_path.stem}_aligned{gt_path.suffix}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Align GT velocity to estimate timestamps using linear interpolation.",
	)
	parser.add_argument("--est_csv", required=True, help="Estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="GT velocity CSV")
	parser.add_argument("--out", help="Aligned GT output CSV path")
	parser.add_argument("--out_dir", help="Output directory for aligned GT CSV")
	parser.add_argument("--time_col", default="time", help="Timestamp column name")
	parser.add_argument(
		"--gt_latency",
		type=float,
		default=0.025,
		help="Seconds to subtract from GT time to compensate transport lag",
	)
	parser.add_argument(
		"--tolerance",
		type=float,
		default=DEFAULT_ALIGNMENT_TOLERANCE_S,
		help="Maximum distance to each bracketing GT sample in seconds (default: 0.025)",
	)
	args = parser.parse_args()

	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	if not est_path.exists():
		raise SystemExit(f"Estimate CSV not found: {est_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	out_dir = Path(args.out_dir) if args.out_dir else None
	if out_dir is not None:
		out_dir.mkdir(parents=True, exist_ok=True)

	output_path = _derive_output_path(
		gt_path,
		Path(args.out) if args.out else None,
		out_dir,
	)

	df_est = _load_sorted_csv(est_path, args.time_col)
	df_gt = _load_sorted_csv(gt_path, args.time_col)

	# Step 3: deterministic latency compensation on the GT timeline.
	df_gt = _apply_latency(df_gt, args.time_col, args.gt_latency)

	# Step 4: linearly interpolate GT onto estimate timestamps.
	gt_aligned = interpolate_ground_truth_to_timestamps(
		df_gt,
		df_est[args.time_col],
		time_col=args.time_col,
		tolerance_s=args.tolerance,
	)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	gt_aligned.to_csv(output_path, index=False)

	print(f"Saved aligned GT CSV to {output_path}")
	print(f"Estimate rows: {len(df_est)}")
	print(f"GT rows (raw): {len(df_gt)}")
	print(f"Aligned rows: {len(gt_aligned)}")

# python3 4_align_vel_csv.py --est_csv flight_2026-06-03-08-27-15_vel_est_clean.csv --gt_csv flight_2026-06-03-08-27-15_vel_gt_clean.csv

if __name__ == "__main__":
	main()
