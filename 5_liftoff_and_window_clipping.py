#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def _load_sorted_csv(path: Path, time_col: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	if time_col not in df.columns:
		raise ValueError(f"Missing required column '{time_col}' in {path}")

	df = df.copy()
	df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
	df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
	return df


def _find_liftoff_time(
	df: pd.DataFrame,
	time_col: str,
	vz_col: str,
	threshold: float,
	min_consecutive: int,
) -> float:
	if vz_col not in df.columns:
		raise ValueError(f"Missing required column '{vz_col}' in GT CSV")

	mask = df[vz_col].abs() > threshold
	window = mask.rolling(min_consecutive, min_periods=min_consecutive).sum()
	hits = window == min_consecutive
	if not hits.any():
		raise ValueError(
			f"No liftoff found: |{vz_col}| > {threshold} for {min_consecutive} frames",
		)

	end_pos = int(hits.idxmax())
	start_pos = end_pos - min_consecutive + 1
	return float(df.at[start_pos, time_col])


def _clip_window(
	df: pd.DataFrame,
	time_col: str,
	start_time: float,
	end_time: float,
) -> pd.DataFrame:
	return df[(df[time_col] >= start_time) & (df[time_col] <= end_time)].copy()


def _validate_window(start_time: float, end_time: float) -> None:
	if end_time <= start_time:
		raise ValueError("Window end must be greater than window start")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Detect liftoff and clip estimate/GT velocity to a fixed window.",
	)
	parser.add_argument("--est_csv", required=True, help="Clean estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="Clean aligned GT velocity CSV")
	parser.add_argument(
		"--vz_col",
		default="vel_z",
		help="Vertical velocity column name in GT CSV",
	)
	parser.add_argument(
		"--lift_threshold",
		type=float,
		default=0.3,
		help="Liftoff threshold for |vz| (m/s)",
	)
	parser.add_argument(
		"--min_consecutive",
		type=int,
		default=5,
		help="Consecutive frames required for liftoff detection",
	)
	parser.add_argument(
		"--window_start",
		type=float,
		default=7.0,
		help="Seconds after liftoff to start window",
	)
	parser.add_argument(
		"--window_end",
		type=float,
		default=22.0,
		help="Seconds after liftoff to end window",
	)
	args = parser.parse_args()

	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	if not est_path.exists():
		raise SystemExit(f"Estimate CSV not found: {est_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	output_est = Path.cwd() / f"{est_path.stem}_window{est_path.suffix}"
	output_gt = Path.cwd() / f"{gt_path.stem}_window{gt_path.suffix}"

	df_est = _load_sorted_csv(est_path, "time")
	df_gt = _load_sorted_csv(gt_path, "time")

	t_lift = _find_liftoff_time(
		df_gt,
		"time",
		args.vz_col,
		args.lift_threshold,
		args.min_consecutive,
	)
	start_time = t_lift + args.window_start
	end_time = t_lift + args.window_end
	_validate_window(start_time, end_time)

	clipped_est = _clip_window(df_est, "time", start_time, end_time)
	clipped_gt = _clip_window(df_gt, "time", start_time, end_time)

	clipped_est.to_csv(output_est, index=False)
	clipped_gt.to_csv(output_gt, index=False)

	ref_time = df_gt["time"].iloc[0]
	requested_duration = end_time - start_time

	actual_est_start = clipped_est["time"].iloc[0]
	actual_est_end = clipped_est["time"].iloc[-1]
	actual_gt_start = clipped_gt["time"].iloc[0]
	actual_gt_end = clipped_gt["time"].iloc[-1]

	actual_est_duration = actual_est_end - actual_est_start
	actual_gt_duration = actual_gt_end - actual_gt_start

	print("Input time ranges:")
	print(
		f"  est absolute: {df_est['time'].iloc[0]:.6f} -> {df_est['time'].iloc[-1]:.6f}"
	)
	print(
		f"  gt  absolute: {df_gt['time'].iloc[0]:.6f} -> {df_gt['time'].iloc[-1]:.6f}"
	)

	print("Detected liftoff:")
	print(f"  absolute: {t_lift:.6f}")
	print(f"  relative: {t_lift - ref_time:.6f} s")

	print("Requested evaluation window:")
	print(f"  start after liftoff: {args.window_start:.3f} s")
	print(f"  end after liftoff  : {args.window_end:.3f} s")
	print(f"  requested duration : {requested_duration:.6f} s")

	print("Actual clipped window:")
	print(
		f"  est relative: {actual_est_start - ref_time:.6f} -> "
		f"{actual_est_end - ref_time:.6f} s"
	)
	print(
		f"  gt  relative: {actual_gt_start - ref_time:.6f} -> "
		f"{actual_gt_end - ref_time:.6f} s"
	)
	print(f"  est duration: {actual_est_duration:.6f} s")
	print(f"  gt  duration: {actual_gt_duration:.6f} s")

	if actual_est_duration < requested_duration:
		print(
			"Note: actual estimate window is shorter than requested by "
			f"{requested_duration - actual_est_duration:.6f} s because the log ended early."
		)

	if actual_gt_duration < requested_duration:
		print(
			"Note: actual GT window is shorter than requested by "
			f"{requested_duration - actual_gt_duration:.6f} s because the log ended early."
		)

	print(f"Saved estimate CSV to {output_est} (rows: {len(clipped_est)})")
	print(f"Saved GT CSV to {output_gt} (rows: {len(clipped_gt)})")


if __name__ == "__main__":
	main()

# python3 5_liftoff_and_window_clipping.py --est_csv flight_6_27_vel_est_clean.csv --gt_csv flight_6_27_vel_gt_clean_aligned.csv
