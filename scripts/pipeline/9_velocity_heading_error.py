#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = ["time", "vel_x", "vel_y"]
HEADING_P90_LIMIT_DEG = 40.0
AVG_MAGNITUDE_ERROR_LIMIT_PCT = 40.0


def _load_velocity_csv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {path}")

	df = df.copy()
	for col in required_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.dropna(subset=required_cols).sort_values("time").reset_index(drop=True)
	if df.empty:
		raise ValueError(f"No valid velocity rows in {path}")
	return df


def _merge_velocity_data(est_df: pd.DataFrame, gt_df: pd.DataFrame) -> pd.DataFrame:
	merged = pd.merge(
		est_df[REQUIRED_COLS],
		gt_df[REQUIRED_COLS],
		on="time",
		how="inner",
		suffixes=("_est", "_gt"),
	)
	if merged.empty:
		raise ValueError("No overlapping timestamps between estimate and GT CSVs")
	return merged.sort_values("time").reset_index(drop=True)


def _compute_sample_metrics(merged: pd.DataFrame) -> pd.DataFrame:
	est_x = merged["vel_x_est"].to_numpy(dtype=float)
	est_y = merged["vel_y_est"].to_numpy(dtype=float)
	gt_x = merged["vel_x_gt"].to_numpy(dtype=float)
	gt_y = merged["vel_y_gt"].to_numpy(dtype=float)

	est_speed_xy = np.sqrt(est_x ** 2 + est_y ** 2)
	gt_speed_xy = np.sqrt(gt_x ** 2 + gt_y ** 2)
	dot_xy = est_x * gt_x + est_y * gt_y
	denom = est_speed_xy * gt_speed_xy

	heading_error_deg = np.full(len(merged), np.nan, dtype=float)
	valid_heading = denom > 0.0
	cos_theta = np.zeros(len(merged), dtype=float)
	cos_theta[valid_heading] = dot_xy[valid_heading] / denom[valid_heading]
	cos_theta = np.clip(cos_theta, -1.0, 1.0)
	heading_error_deg[valid_heading] = np.degrees(np.arccos(cos_theta[valid_heading]))

	magnitude_error_pct = np.full(len(merged), np.nan, dtype=float)
	valid_magnitude = gt_speed_xy > 0.0
	magnitude_error_pct[valid_magnitude] = (
		np.abs(est_speed_xy[valid_magnitude] - gt_speed_xy[valid_magnitude])
		/ gt_speed_xy[valid_magnitude]
		* 100.0
	)

	result = merged.copy()
	result["est_speed_xy"] = est_speed_xy
	result["gt_speed_xy"] = gt_speed_xy
	result["heading_error_deg"] = heading_error_deg
	result["magnitude_error_pct"] = magnitude_error_pct

	return result


def _nearest_rank_percentile(values: pd.Series, percentile: float) -> float:
	valid_values = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
	if len(valid_values) == 0:
		return float("nan")
	index = int(np.ceil(percentile / 100.0 * len(valid_values))) - 1
	index = min(max(index, 0), len(valid_values) - 1)
	return float(valid_values[index])


def _summarize_window(samples: pd.DataFrame) -> pd.DataFrame:
	p90_heading = _nearest_rank_percentile(samples["heading_error_deg"], 90.0)
	avg_magnitude_pct = pd.to_numeric(samples["magnitude_error_pct"], errors="coerce").mean()
	heading_pass = bool(p90_heading < HEADING_P90_LIMIT_DEG) if np.isfinite(p90_heading) else False
	magnitude_pass = (
		bool(avg_magnitude_pct < AVG_MAGNITUDE_ERROR_LIMIT_PCT)
		if np.isfinite(avg_magnitude_pct)
		else False
	)

	return pd.DataFrame(
		[
			{
				"samples": int(len(samples)),
				"valid_heading_samples": int(samples["heading_error_deg"].notna().sum()),
				"valid_magnitude_samples": int(samples["magnitude_error_pct"].notna().sum()),
				"heading_p90_limit_deg": HEADING_P90_LIMIT_DEG,
				"p90_heading_error_deg": p90_heading,
				"heading_pass": heading_pass,
				"avg_magnitude_error_limit_pct": AVG_MAGNITUDE_ERROR_LIMIT_PCT,
				"avg_magnitude_error_pct": avg_magnitude_pct,
				"magnitude_pass": magnitude_pass,
				"window_pass": heading_pass and magnitude_pass,
			}
		]
	)


def _print_summary(summary: pd.DataFrame, samples: pd.DataFrame) -> None:
	print("Velocity Heading Error Summary")
	print("Formula uses horizontal x/y velocity components.")
	print("")
	row = summary.iloc[0]
	print(f"Full window samples: {int(row['samples'])}")
	print(
		"Requirement: "
		f"P90 heading error < {row['heading_p90_limit_deg']:.1f} deg AND "
		f"average magnitude error < {row['avg_magnitude_error_limit_pct']:.1f}%"
	)
	print(
		"P90 heading error: "
		f"{row['p90_heading_error_deg']:.3f} deg, heading_pass={row['heading_pass']}"
		if np.isfinite(row["p90_heading_error_deg"])
		else f"P90 heading error: nan deg, heading_pass={row['heading_pass']}"
	)
	print(
		"Average magnitude error: "
		f"{row['avg_magnitude_error_pct']:.3f}%, magnitude_pass={row['magnitude_pass']}"
		if np.isfinite(row["avg_magnitude_error_pct"])
		else f"Average magnitude error: nan%, magnitude_pass={row['magnitude_pass']}"
	)
	print(f"Window pass: {row['window_pass']}")


def _derive_output_paths(est_path: Path) -> Tuple[Path, Path]:
	return (
		Path.cwd() / f"{est_path.stem}_vhe_samples{est_path.suffix}",
		Path.cwd() / f"{est_path.stem}_vhe_summary{est_path.suffix}",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Calculate horizontal velocity heading error and magnitude error, "
			"then evaluate the full window with fixed p90 heading and average magnitude limits."
		),
	)
	parser.add_argument("--est_csv", required=True, help="Windowed estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="Windowed aligned GT velocity CSV")
	parser.add_argument(
		"--out_samples_csv",
		help="Optional output CSV with per-sample heading and magnitude errors",
	)
	parser.add_argument(
		"--out_summary_csv",
		help="Optional output CSV with full-window VHE summary metrics",
	)
	args = parser.parse_args()

	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	if not est_path.exists():
		raise SystemExit(f"Estimate CSV not found: {est_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	default_samples_path, default_summary_path = _derive_output_paths(est_path)
	samples_path = Path(args.out_samples_csv) if args.out_samples_csv else default_samples_path
	summary_path = Path(args.out_summary_csv) if args.out_summary_csv else default_summary_path

	est_df = _load_velocity_csv(est_path, REQUIRED_COLS)
	gt_df = _load_velocity_csv(gt_path, REQUIRED_COLS)
	merged = _merge_velocity_data(est_df, gt_df)
	samples = _compute_sample_metrics(merged)
	summary = _summarize_window(samples)

	samples_path.parent.mkdir(parents=True, exist_ok=True)
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	samples.to_csv(samples_path, index=False)
	summary.to_csv(summary_path, index=False)

	_print_summary(summary, samples)
	print(f"Saved per-sample VHE results to {samples_path}")
	print(f"Saved VHE summary to {summary_path}")


if __name__ == "__main__":
	main()
