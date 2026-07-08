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


def _clip_window(
	df: pd.DataFrame,
	time_col: str,
	start_time: float,
	end_time: float,
) -> pd.DataFrame:
	return df[(df[time_col] >= start_time) & (df[time_col] <= end_time)].copy()


def _validate_window(window_start: float, window_end: float) -> None:
	if window_start < 0 or window_end < 0:
		raise ValueError("Window time values must be non-negative")
	if window_end <= window_start:
		raise ValueError("Window end must be greater than window start")


def _get_reference_time(df_gt: pd.DataFrame) -> float:
	if df_gt.empty:
		raise ValueError("GT CSV has no valid time rows")
	return float(df_gt["time"].iloc[0])


def _print_range(
	label: str,
	df: pd.DataFrame,
	ref_time: float,
	requested_duration: float,
) -> None:
	if df.empty:
		print(f"  {label}: no rows in clipped window")
		return

	actual_start = float(df["time"].iloc[0])
	actual_end = float(df["time"].iloc[-1])
	actual_duration = actual_end - actual_start
	print(
		f"  {label}: {actual_start - ref_time:.6f} -> "
		f"{actual_end - ref_time:.6f} s"
	)
	print(f"  {label} duration: {actual_duration:.6f} s")

	if actual_duration < requested_duration:
		print(
			f"  Note: {label} window is shorter than requested by "
			f"{requested_duration - actual_duration:.6f} s."
		)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Clip estimate/GT velocity CSVs using manually specified time values from aligned GT start.",
	)
	parser.add_argument("--est_csv", required=True, help="Estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="GT velocity CSV")
	parser.add_argument(
		"--window_start",
		type=float,
		required=True,
		help="Start time in seconds from the aligned GT CSV start",
	)
	parser.add_argument(
		"--window_end",
		type=float,
		required=True,
		help="End time in seconds from the aligned GT CSV start",
	)
	parser.add_argument(
		"--output_suffix",
		default="_window",
		help="Suffix added to output CSV stems",
	)
	args = parser.parse_args()

	_validate_window(args.window_start, args.window_end)

	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	if not est_path.exists():
		raise SystemExit(f"Estimate CSV not found: {est_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	output_est = Path.cwd() / f"{est_path.stem}{args.output_suffix}{est_path.suffix}"
	output_gt = Path.cwd() / f"{gt_path.stem}{args.output_suffix}{gt_path.suffix}"

	df_est = _load_sorted_csv(est_path, "time")
	df_gt = _load_sorted_csv(gt_path, "time")

	ref_time = _get_reference_time(df_gt)
	start_time = ref_time + args.window_start
	end_time = ref_time + args.window_end
	requested_duration = args.window_end - args.window_start

	clipped_est = _clip_window(df_est, "time", start_time, end_time)
	clipped_gt = _clip_window(df_gt, "time", start_time, end_time)

	clipped_est.to_csv(output_est, index=False)
	clipped_gt.to_csv(output_gt, index=False)

	print("Input time ranges:")
	print(
		f"  est absolute: {df_est['time'].iloc[0]:.6f} -> {df_est['time'].iloc[-1]:.6f}"
	)
	print(
		f"  gt  absolute: {df_gt['time'].iloc[0]:.6f} -> {df_gt['time'].iloc[-1]:.6f}"
	)

	print("Requested clipping window:")
	print("  reference: aligned GT start")
	print(f"  reference absolute time: {ref_time:.6f}")
	print(f"  requested time values: {args.window_start:.6f} -> {args.window_end:.6f} s")
	print(f"  absolute: {start_time:.6f} -> {end_time:.6f}")
	print(f"  requested duration: {requested_duration:.6f} s")

	print("Actual clipped window:")
	_print_range("est", clipped_est, ref_time, requested_duration)
	_print_range("gt ", clipped_gt, ref_time, requested_duration)

	print(f"Saved estimate CSV to {output_est} (rows: {len(clipped_est)})")
	print(f"Saved GT CSV to {output_gt} (rows: {len(clipped_gt)})")


if __name__ == "__main__":
	main()

# python3 5_manual_relative_window_clipping.py --est_csv flight_6_27_vel_est_clean.csv --gt_csv flight_6_27_vel_gt_clean_aligned.csv --window_start 20.0 --window_end 35.0
