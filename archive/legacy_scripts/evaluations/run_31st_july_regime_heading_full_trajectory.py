#!/usr/bin/env python3

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[2]
PIPELINE_DIR = SCRIPT_DIR / "scripts" / "pipeline"
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]
MIN_EVALUATED_GT_SPEED_MPS = 0.2


AI_WINDOWS = {
	1: (36.67, 70.11),
	2: (37.12, 73.13),
	3: (37.55, 68.79),
	4: (34.63, 63.98),
	5: (33.57, 66.56),
	6: (32.35, 64.76),
	7: (33.98, 68.48),
	8: (36.93, 68.40),
}

RAW_WINDOWS = {
	1: (61.86, 87.66),
	3: (118.44, 161.29),
	4: (37.97, 75.38),
	5: (30.37, 63.64),
	6: (38.23, 67.16),
	7: (42.09, 71.53),
	8: (34.59, 59.24),
	9: (36.91, 68.46),
	10: (35.02, 65.28),
}


@dataclass(frozen=True)
class SpeedRegime:
	name: str
	label: str
	min_speed_mps: float
	max_speed_mps: float


SPEED_REGIMES = [
	SpeedRegime("low", "0.2 <= GT horizontal speed < 0.5 m/s", 0.2, 0.5),
	SpeedRegime("medium", "0.5 <= GT horizontal speed < 2 m/s", 0.5, 2.0),
	SpeedRegime("high", "2 <= GT horizontal speed < 5 m/s", 2.0, 5.0),
]


def load_script_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, PIPELINE_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


CLEANER = load_script_module("3_vel_csv_dataset_cleaner.py", "vel_csv_dataset_cleaner")
VHE = load_script_module("9_velocity_heading_error.py", "velocity_heading_error")


def clean_velocity_csv(
	input_path: Path,
	output_path: Path,
	window: int,
	median_tol: Optional[float],
	bound_overrides: List[Tuple[str, Optional[float], Optional[float]]],
	fill: str,
	savgol_window: Optional[int],
	savgol_polyorder: int,
) -> Path:
	df = pd.read_csv(input_path)
	cleaned, _, _ = CLEANER.clean_dataframe(
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
	return output_path


def align_ground_truth_to_estimate(
	est_clean_path: Path,
	gt_clean_path: Path,
	output_path: Path,
	gt_latency: float,
	tolerance: Optional[float],
) -> Path:
	df_est = pd.read_csv(est_clean_path)
	df_gt = pd.read_csv(gt_clean_path)
	for label, df in (("estimate", df_est), ("GT", df_gt)):
		if "time" not in df.columns:
			raise ValueError(f"Missing time column in {label} CSV")
		df["time"] = pd.to_numeric(df["time"], errors="coerce")

	df_est = df_est.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	df_gt = df_gt.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	if gt_latency:
		df_gt = df_gt.copy()
		df_gt["time"] = df_gt["time"] - gt_latency
		df_gt = df_gt.sort_values("time").reset_index(drop=True)

	merge_kwargs = {
		"on": "time",
		"direction": "nearest",
		"suffixes": ("_est", "_gt"),
	}
	if tolerance is not None:
		merge_kwargs["tolerance"] = tolerance

	df_sync = pd.merge_asof(df_est, df_gt, **merge_kwargs)
	gt_cols = []
	rename_map = {}
	for col in df_gt.columns:
		if col in df_sync.columns:
			gt_cols.append(col)
			continue
		gt_col = f"{col}_gt"
		if gt_col in df_sync.columns:
			gt_cols.append(gt_col)
			rename_map[gt_col] = col
	if not gt_cols:
		raise KeyError(f"No GT columns found after alignment for {gt_clean_path}")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	df_sync[gt_cols].rename(columns=rename_map).to_csv(output_path, index=False)
	return output_path


def clip_velocity_window(
	est_clean_path: Path,
	gt_aligned_path: Path,
	window_start: float,
	window_end: float,
) -> Tuple[Path, Path]:
	if window_end <= window_start:
		raise ValueError("Window end must be greater than window start")

	df_est = pd.read_csv(est_clean_path)
	df_gt = pd.read_csv(gt_aligned_path)
	for label, df in (("estimate", df_est), ("GT", df_gt)):
		if "time" not in df.columns:
			raise ValueError(f"Missing time column in {label} CSV")
		df["time"] = pd.to_numeric(df["time"], errors="coerce")

	df_est = df_est.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	df_gt = df_gt.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	if df_gt.empty:
		raise ValueError(f"GT CSV has no valid rows: {gt_aligned_path}")

	ref_time = float(df_gt["time"].iloc[0])
	start_abs = ref_time + window_start
	end_abs = ref_time + window_end
	est_window = df_est[(df_est["time"] >= start_abs) & (df_est["time"] <= end_abs)].copy()
	gt_window = df_gt[(df_gt["time"] >= start_abs) & (df_gt["time"] <= end_abs)].copy()
	if est_window.empty or gt_window.empty:
		raise ValueError(
			f"Empty clipped window for {est_clean_path.stem}: "
			f"est rows={len(est_window)}, gt rows={len(gt_window)}"
		)

	est_out = est_clean_path.with_name(f"{est_clean_path.stem}_full_window{est_clean_path.suffix}")
	gt_out = gt_aligned_path.with_name(f"{gt_aligned_path.stem}_full_window{gt_aligned_path.suffix}")
	est_window.to_csv(est_out, index=False)
	gt_window.to_csv(gt_out, index=False)
	return est_out, gt_out


def calculate_samples(est_window_path: Path, gt_window_path: Path) -> pd.DataFrame:
	est_df = VHE._load_velocity_csv(est_window_path, VHE.REQUIRED_COLS)
	gt_df = VHE._load_velocity_csv(gt_window_path, VHE.REQUIRED_COLS)
	merged = VHE._merge_velocity_data(est_df, gt_df)
	return VHE._compute_sample_metrics(merged)


def summarize_regime(samples: pd.DataFrame, regime: SpeedRegime) -> Dict[str, object]:
	total = int(len(samples))
	gt_speed = pd.to_numeric(samples["gt_speed_xy"], errors="coerce")
	heading = pd.to_numeric(samples["heading_error_deg"], errors="coerce")
	below_min = int((gt_speed < MIN_EVALUATED_GT_SPEED_MPS).sum())
	above_regimes = int((gt_speed >= SPEED_REGIMES[-1].max_speed_mps).sum())
	evaluated = samples[gt_speed >= MIN_EVALUATED_GT_SPEED_MPS]
	regime_mask = (gt_speed >= regime.min_speed_mps) & (gt_speed < regime.max_speed_mps)
	regime_heading = heading[regime_mask].dropna()

	return {
		"regime": regime.name,
		"regime_label": regime.label,
		"regime_min_gt_speed_mps": regime.min_speed_mps,
		"regime_max_gt_speed_mps": regime.max_speed_mps,
		"total_window_samples": total,
		"dropped_below_0p2_samples": below_min,
		"retained_gt_speed_ge_0p2_samples": int(len(evaluated)),
		"outside_regimes_gt_speed_ge_5_samples": above_regimes,
		"samples_in_regime": int(len(regime_heading)),
		"mean_heading_error_deg": regime_heading.mean(),
		"p90_heading_error_deg": VHE._nearest_rank_percentile(regime_heading, 90.0),
	}


def summarize_all_regimes(samples: pd.DataFrame) -> List[Dict[str, object]]:
	return [summarize_regime(samples, regime) for regime in SPEED_REGIMES]


def process_flight(
	group: str,
	flight: int,
	window: Tuple[float, float],
	source_dir: Path,
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
	flight_name = f"flight_{flight}"
	est_raw = source_dir / f"{flight_name}_vel_est.csv"
	gt_raw = source_dir / f"{flight_name}_vel_gt.csv"
	if not est_raw.exists():
		raise FileNotFoundError(f"Estimate source CSV not found: {est_raw}")
	if not gt_raw.exists():
		raise FileNotFoundError(f"GT source CSV not found: {gt_raw}")

	group_out = output_dir / group
	est_clean = group_out / f"{flight_name}_vel_est_clean.csv"
	gt_clean = group_out / f"{flight_name}_vel_gt_clean.csv"
	gt_aligned = group_out / f"{flight_name}_vel_gt_clean_aligned.csv"

	clean_velocity_csv(
		est_raw,
		est_clean,
		args.clean_window,
		args.median_tol,
		args.bound_overrides,
		args.fill,
		args.savgol_window,
		args.savgol_polyorder,
	)
	clean_velocity_csv(
		gt_raw,
		gt_clean,
		args.clean_window,
		args.median_tol,
		args.bound_overrides,
		args.fill,
		args.savgol_window,
		args.savgol_polyorder,
	)
	align_ground_truth_to_estimate(est_clean, gt_clean, gt_aligned, args.gt_latency, args.tolerance)
	est_window, gt_window = clip_velocity_window(est_clean, gt_aligned, window[0], window[1])

	samples = calculate_samples(est_window, gt_window)
	samples = samples.copy()
	samples["group"] = group
	samples["flight"] = flight_name
	samples["window_start_s"] = window[0]
	samples["window_end_s"] = window[1]
	samples["speed_regime"] = "dropped_below_0p2"
	for regime in SPEED_REGIMES:
		mask = (
			(samples["gt_speed_xy"] >= regime.min_speed_mps)
			& (samples["gt_speed_xy"] < regime.max_speed_mps)
		)
		samples.loc[mask, "speed_regime"] = regime.name
	samples.loc[samples["gt_speed_xy"] >= SPEED_REGIMES[-1].max_speed_mps, "speed_regime"] = "outside_ge_5"
	samples_path = est_window.with_name(f"{est_window.stem}_regime_heading_samples{est_window.suffix}")
	samples.to_csv(samples_path, index=False)

	rows = []
	for record in summarize_all_regimes(samples):
		record.update(
			{
				"group": group,
				"flight": flight_name,
				"window_start_s": window[0],
				"window_end_s": window[1],
				"window_duration_s": window[1] - window[0],
			}
		)
		rows.append(record)

	summary_path = est_window.with_name(f"{est_window.stem}_regime_heading_summary{est_window.suffix}")
	order_flight_summary_columns(pd.DataFrame(rows)).to_csv(summary_path, index=False)
	return rows, samples


def summarize_by_group(sample_frames: List[pd.DataFrame]) -> pd.DataFrame:
	if not sample_frames:
		return pd.DataFrame()

	all_samples = pd.concat(sample_frames, ignore_index=True)
	rows: List[Dict[str, object]] = []
	for group, group_df in all_samples.groupby("group", sort=False):
		for record in summarize_all_regimes(group_df):
			record["group"] = group
			rows.append(record)
	for record in summarize_all_regimes(all_samples):
		record["group"] = "ALL"
		rows.append(record)
	return order_group_summary_columns(pd.DataFrame(rows))


def order_flight_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"flight",
		"window_start_s",
		"window_end_s",
		"window_duration_s",
		"regime",
		"regime_label",
		"regime_min_gt_speed_mps",
		"regime_max_gt_speed_mps",
		"total_window_samples",
		"dropped_below_0p2_samples",
		"retained_gt_speed_ge_0p2_samples",
		"outside_regimes_gt_speed_ge_5_samples",
		"samples_in_regime",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def order_group_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"regime",
		"regime_label",
		"regime_min_gt_speed_mps",
		"regime_max_gt_speed_mps",
		"total_window_samples",
		"dropped_below_0p2_samples",
		"retained_gt_speed_ge_0p2_samples",
		"outside_regimes_gt_speed_ge_5_samples",
		"samples_in_regime",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run 31st July full-trajectory heading-error evaluation split by GT horizontal speed regime."
		),
	)
	parser.add_argument("--ai_dir", default="31st_July_AI", help="Source directory for AI CSVs")
	parser.add_argument("--raw_dir", default="31st_July_RAW", help="Source directory for RAW CSVs")
	parser.add_argument(
		"--out_dir",
		default="31st_July_regime_heading_full_trajectory",
		help="Output directory for cleaned, aligned, windowed, and regime result CSVs",
	)
	parser.add_argument("--group", choices=["all", "AI", "RAW"], default="all", help="Subset to process")
	parser.add_argument("--clean_window", type=int, default=9, help="Sliding median window size")
	parser.add_argument("--median_tol", type=float, default=None, help="Override median tolerance")
	parser.add_argument(
		"--bound",
		action="append",
		default=None,
		help="Override cleaning bounds pattern:min:max, same as 3_vel_csv_dataset_cleaner.py",
	)
	parser.add_argument("--fill", choices=["none", "linear"], default="linear", help="NaN fill mode")
	parser.add_argument("--savgol_window", type=int, default=21, help="Savitzky-Golay smoothing window")
	parser.add_argument("--savgol_polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
	parser.add_argument("--gt_latency", type=float, default=0.025, help="Seconds to subtract from GT time")
	parser.add_argument("--tolerance", type=float, default=None, help="Optional max time delta for alignment")
	args = parser.parse_args()
	args.clean_window = CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = CLEANER._parse_bound_overrides(args.bound or DEFAULT_VELOCITY_BOUNDS)
	return args


def select_groups(args: argparse.Namespace) -> List[Tuple[str, Path, Dict[int, Tuple[float, float]]]]:
	groups = []
	if args.group in {"all", "AI"}:
		groups.append(("AI", Path(args.ai_dir), AI_WINDOWS))
	if args.group in {"all", "RAW"}:
		groups.append(("RAW", Path(args.raw_dir), RAW_WINDOWS))
	return groups


def process_groups(
	groups: List[Tuple[str, Path, Dict[int, Tuple[float, float]]]],
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	all_rows: List[Dict[str, object]] = []
	sample_frames: List[pd.DataFrame] = []

	for group, source_dir, windows in groups:
		for flight, window in windows.items():
			print(f"Processing {group} flight_{flight}: {window[0]:.2f} -> {window[1]:.2f} s")
			rows, samples = process_flight(group, flight, window, source_dir, output_dir, args)
			all_rows.extend(rows)
			sample_frames.append(samples)

	if not all_rows:
		raise SystemExit("No flights processed")
	return order_flight_summary_columns(pd.DataFrame(all_rows)), summarize_by_group(sample_frames)


def save_outputs(output_dir: Path, flight_summary: pd.DataFrame, group_summary: pd.DataFrame) -> Tuple[Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	flight_summary_path = output_dir / "31st_july_regime_heading_summary_all_windows.csv"
	group_summary_path = output_dir / "31st_july_regime_heading_summary_by_group.csv"
	flight_summary.to_csv(flight_summary_path, index=False)
	group_summary.to_csv(group_summary_path, index=False)
	return flight_summary_path, group_summary_path


def print_summary(
	flight_summary: pd.DataFrame,
	group_summary: pd.DataFrame,
	flight_summary_path: Path,
	group_summary_path: Path,
) -> None:
	columns = [
		"group",
		"regime",
		"total_window_samples",
		"dropped_below_0p2_samples",
		"retained_gt_speed_ge_0p2_samples",
		"outside_regimes_gt_speed_ge_5_samples",
		"samples_in_regime",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
	]
	per_flight_columns = ["flight", "window_start_s", "window_end_s"] + columns

	print("")
	print(f"Saved per-flight regime heading summary to {flight_summary_path}")
	print(f"Saved AI/RAW regime heading summary to {group_summary_path}")
	print("")
	print("AI/RAW rolled-up summary:")
	print(group_summary[columns].to_string(index=False))
	print("")
	print("Per-flight summary:")
	print(flight_summary[per_flight_columns].to_string(index=False))


def run() -> None:
	args = parse_arguments()
	output_dir = Path(args.out_dir)
	flight_summary, group_summary = process_groups(select_groups(args), output_dir, args)
	flight_summary_path, group_summary_path = save_outputs(output_dir, flight_summary, group_summary)
	print_summary(flight_summary, group_summary, flight_summary_path, group_summary_path)


if __name__ == "__main__":
	run()
