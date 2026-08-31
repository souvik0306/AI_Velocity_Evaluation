#!/usr/bin/env python3

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]
METRIC_REQUIRED_COLS = ["time", "vel_x", "vel_y"]


AI_WINDOWS = {
	1: (36.67, 46.67),
	2: (37.12, 47.12),
	3: (37.55, 47.55),
	4: (34.63, 44.63),
	5: (33.57, 43.57),
	6: (32.35, 42.35),
	7: (33.98, 43.98),
	8: (36.93, 46.93),
}

RAW_WINDOWS = {
	1: (61.86, 71.86),
	3: (118.44, 128.44),
	4: (37.97, 47.97),
	5: (30.37, 40.37),
	6: (38.23, 48.23),
	7: (42.09, 52.09),
	8: (34.59, 44.59),
	9: (36.91, 46.91),
	10: (35.02, 45.02),
}


@dataclass(frozen=True)
class SpeedRegime:
	name: str
	label: str
	min_gt_speed_mps: float
	max_gt_speed_mps: float


SPEED_REGIMES = [
	SpeedRegime("low_0_to_0p5", "0.0 <= GT horizontal speed < 0.5 m/s", 0.0, 0.5),
	SpeedRegime("medium_0p5_to_2", "0.5 <= GT horizontal speed < 2 m/s", 0.5, 2.0),
	SpeedRegime("high_2_to_5", "2 <= GT horizontal speed < 5 m/s", 2.0, 5.0),
]


def load_script_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


CLEANER = load_script_module("3_vel_csv_dataset_cleaner.py", "vel_csv_dataset_cleaner")


def resolve_repo_path(value: str) -> Path:
	path = Path(value)
	if path.is_absolute():
		return path
	return SCRIPT_DIR / path


def nearest_rank_percentile(values: pd.Series, percentile: float) -> float:
	valid_values = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
	if len(valid_values) == 0:
		return float("nan")
	index = int(np.ceil(percentile / 100.0 * len(valid_values))) - 1
	index = min(max(index, 0), len(valid_values) - 1)
	return float(valid_values[index])


def load_velocity_csv(path: Path, required_cols: Iterable[str]) -> pd.DataFrame:
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


def clean_velocity_csv(
	input_path: Path,
	output_path: Path,
	window: int,
	median_tol: Optional[float],
	bound_overrides: List[Tuple[str, Optional[float], Optional[float]]],
	fill: str,
	savgol_window: Optional[int],
	savgol_polyorder: int,
) -> Tuple[Path, Dict[str, int], Dict[str, int]]:
	df = pd.read_csv(input_path)
	cleaned, pruned_counts, filtered_counts = CLEANER.clean_dataframe(
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
	return output_path, pruned_counts, filtered_counts


def align_ground_truth_to_estimate(
	est_clean_path: Path,
	gt_clean_path: Path,
	output_path: Path,
	gt_latency: float,
	tolerance: Optional[float],
) -> Path:
	df_est = load_velocity_csv(est_clean_path, ["time"])
	df_gt = load_velocity_csv(gt_clean_path, ["time"])
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

	df_est = load_velocity_csv(est_clean_path, ["time"])
	df_gt = load_velocity_csv(gt_aligned_path, ["time"])
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

	est_out = est_clean_path.with_name(f"{est_clean_path.stem}_10s_window{est_clean_path.suffix}")
	gt_out = gt_aligned_path.with_name(f"{gt_aligned_path.stem}_10s_window{gt_aligned_path.suffix}")
	est_window.to_csv(est_out, index=False)
	gt_window.to_csv(gt_out, index=False)
	return est_out, gt_out


def merge_window_data(est_window_path: Path, gt_window_path: Path) -> pd.DataFrame:
	est_df = load_velocity_csv(est_window_path, METRIC_REQUIRED_COLS)
	gt_df = load_velocity_csv(gt_window_path, METRIC_REQUIRED_COLS)
	merged = pd.merge(
		est_df[METRIC_REQUIRED_COLS],
		gt_df[METRIC_REQUIRED_COLS],
		on="time",
		how="inner",
		suffixes=("_est", "_gt"),
	)
	if merged.empty:
		raise ValueError(f"No overlapping timestamps between {est_window_path} and {gt_window_path}")
	return merged.sort_values("time").reset_index(drop=True)


def assign_speed_regime(gt_abs_magnitude: np.ndarray) -> np.ndarray:
	labels = np.full(len(gt_abs_magnitude), "outside_ge_5", dtype=object)
	for regime in SPEED_REGIMES:
		mask = (gt_abs_magnitude >= regime.min_gt_speed_mps) & (
			gt_abs_magnitude < regime.max_gt_speed_mps
		)
		labels[mask] = regime.name
	labels[~np.isfinite(gt_abs_magnitude)] = "invalid_gt_speed"
	return labels


def compute_sample_metrics(merged: pd.DataFrame) -> pd.DataFrame:
	est_x = merged["vel_x_est"].to_numpy(dtype=float)
	est_y = merged["vel_y_est"].to_numpy(dtype=float)
	gt_x = merged["vel_x_gt"].to_numpy(dtype=float)
	gt_y = merged["vel_y_gt"].to_numpy(dtype=float)

	est_abs_magnitude = np.sqrt(est_x**2 + est_y**2)
	gt_abs_magnitude = np.sqrt(gt_x**2 + gt_y**2)
	dot_xy = est_x * gt_x + est_y * gt_y
	denom = est_abs_magnitude * gt_abs_magnitude

	heading_error_deg = np.full(len(merged), np.nan, dtype=float)
	valid_heading = denom > 0.0
	cos_theta = np.zeros(len(merged), dtype=float)
	cos_theta[valid_heading] = dot_xy[valid_heading] / denom[valid_heading]
	cos_theta = np.clip(cos_theta, -1.0, 1.0)
	heading_error_deg[valid_heading] = np.degrees(np.arccos(cos_theta[valid_heading]))

	abs_magnitude_error = np.abs(est_abs_magnitude - gt_abs_magnitude)
	magnitude_error_pct = np.full(len(merged), np.nan, dtype=float)
	valid_magnitude = gt_abs_magnitude > 0.0
	magnitude_error_pct[valid_magnitude] = (
		abs_magnitude_error[valid_magnitude] / gt_abs_magnitude[valid_magnitude] * 100.0
	)

	samples = merged.copy()
	samples["rel_time_s"] = samples["time"] - float(samples["time"].iloc[0])
	samples["est_abs_magnitude_xy_mps"] = est_abs_magnitude
	samples["gt_abs_magnitude_xy_mps"] = gt_abs_magnitude
	samples["speed_regime"] = assign_speed_regime(gt_abs_magnitude)
	samples["abs_magnitude_error_mps"] = abs_magnitude_error
	samples["magnitude_error_pct"] = magnitude_error_pct
	samples["heading_error_deg"] = heading_error_deg
	return samples


def summarize_regime(
	samples: pd.DataFrame,
	group: str,
	flight: str,
	window: Tuple[float, float],
	regime: SpeedRegime,
) -> Dict[str, object]:
	regime_samples = samples[samples["speed_regime"] == regime.name]
	total_window_samples = int(len(samples))
	outside_ge_5_samples = int((samples["speed_regime"] == "outside_ge_5").sum())
	invalid_gt_speed_samples = int((samples["speed_regime"] == "invalid_gt_speed").sum())

	return {
		"group": group,
		"flight": flight,
		"window_start_s": window[0],
		"window_end_s": window[1],
		"window_duration_s": window[1] - window[0],
		"speed_regime": regime.name,
		"speed_regime_label": regime.label,
		"regime_min_gt_speed_mps": regime.min_gt_speed_mps,
		"regime_max_gt_speed_mps": regime.max_gt_speed_mps,
		"total_window_samples": total_window_samples,
		"outside_regimes_gt_speed_ge_5_samples": outside_ge_5_samples,
		"invalid_gt_speed_samples": invalid_gt_speed_samples,
		"samples_in_regime": int(len(regime_samples)),
		"valid_heading_samples": int(regime_samples["heading_error_deg"].notna().sum()),
		"valid_magnitude_samples": int(regime_samples["magnitude_error_pct"].notna().sum()),
		"mean_heading_error_deg": pd.to_numeric(
			regime_samples["heading_error_deg"], errors="coerce"
		).mean(),
		"p90_heading_error_deg": nearest_rank_percentile(regime_samples["heading_error_deg"], 90.0),
		"avg_est_abs_magnitude_xy_mps": pd.to_numeric(
			regime_samples["est_abs_magnitude_xy_mps"], errors="coerce"
		).mean(),
		"avg_gt_abs_magnitude_xy_mps": pd.to_numeric(
			regime_samples["gt_abs_magnitude_xy_mps"], errors="coerce"
		).mean(),
		"avg_abs_magnitude_error_mps": pd.to_numeric(
			regime_samples["abs_magnitude_error_mps"], errors="coerce"
		).mean(),
		"avg_magnitude_error_pct": pd.to_numeric(
			regime_samples["magnitude_error_pct"], errors="coerce"
		).mean(),
	}


def summarize_all_regimes(
	samples: pd.DataFrame,
	group: str,
	flight: str,
	window: Tuple[float, float],
) -> List[Dict[str, object]]:
	return [summarize_regime(samples, group, flight, window, regime) for regime in SPEED_REGIMES]


def summarize_rollup_regime(samples: pd.DataFrame, group: str, regime: SpeedRegime) -> Dict[str, object]:
	row = summarize_regime(
		samples,
		group,
		"ALL_TRAJECTORIES",
		(float("nan"), float("nan")),
		regime,
	)
	row["trajectories"] = int(samples[["group", "flight"]].drop_duplicates().shape[0])
	row.pop("window_start_s")
	row.pop("window_end_s")
	row.pop("window_duration_s")
	return row


def order_flight_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"flight",
		"window_start_s",
		"window_end_s",
		"window_duration_s",
		"speed_regime",
		"speed_regime_label",
		"regime_min_gt_speed_mps",
		"regime_max_gt_speed_mps",
		"total_window_samples",
		"outside_regimes_gt_speed_ge_5_samples",
		"invalid_gt_speed_samples",
		"samples_in_regime",
		"valid_heading_samples",
		"valid_magnitude_samples",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
		"avg_est_abs_magnitude_xy_mps",
		"avg_gt_abs_magnitude_xy_mps",
		"avg_abs_magnitude_error_mps",
		"avg_magnitude_error_pct",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def order_rollup_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"flight",
		"trajectories",
		"speed_regime",
		"speed_regime_label",
		"regime_min_gt_speed_mps",
		"regime_max_gt_speed_mps",
		"total_window_samples",
		"outside_regimes_gt_speed_ge_5_samples",
		"invalid_gt_speed_samples",
		"samples_in_regime",
		"valid_heading_samples",
		"valid_magnitude_samples",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
		"avg_est_abs_magnitude_xy_mps",
		"avg_gt_abs_magnitude_xy_mps",
		"avg_abs_magnitude_error_mps",
		"avg_magnitude_error_pct",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def process_flight(
	group: str,
	flight: int,
	window: Tuple[float, float],
	source_dir: Path,
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], pd.DataFrame, Dict[str, object]]:
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

	_, est_pruned, est_filtered = clean_velocity_csv(
		est_raw,
		est_clean,
		args.clean_window,
		args.median_tol,
		args.bound_overrides,
		args.fill,
		args.savgol_window,
		args.savgol_polyorder,
	)
	_, gt_pruned, gt_filtered = clean_velocity_csv(
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

	samples = compute_sample_metrics(merge_window_data(est_window, gt_window))
	samples.insert(0, "flight", flight_name)
	samples.insert(0, "group", group)
	samples["window_start_s"] = window[0]
	samples["window_end_s"] = window[1]

	samples_path = group_out / f"{flight_name}_10s_speed_regime_heading_magnitude_samples.csv"
	samples.to_csv(samples_path, index=False)
	rows = summarize_all_regimes(samples, group, flight_name, window)

	cleaning_record = {
		"group": group,
		"flight": flight_name,
		"est_rows_cleaned": int(len(pd.read_csv(est_clean))),
		"gt_rows_cleaned": int(len(pd.read_csv(gt_clean))),
	}
	for col in sorted(set(est_pruned) | set(est_filtered) | set(gt_pruned) | set(gt_filtered)):
		cleaning_record[f"est_pruned_{col}"] = est_pruned.get(col, 0)
		cleaning_record[f"est_median_filtered_{col}"] = est_filtered.get(col, 0)
		cleaning_record[f"gt_pruned_{col}"] = gt_pruned.get(col, 0)
		cleaning_record[f"gt_median_filtered_{col}"] = gt_filtered.get(col, 0)

	return rows, samples, cleaning_record


def select_groups(args: argparse.Namespace) -> List[Tuple[str, Path, Dict[int, Tuple[float, float]]]]:
	groups = []
	if args.group in {"all", "AI"}:
		groups.append(("AI", resolve_repo_path(args.ai_dir), AI_WINDOWS))
	if args.group in {"all", "RAW"}:
		groups.append(("RAW", resolve_repo_path(args.raw_dir), RAW_WINDOWS))
	return groups


def process_groups(
	groups: List[Tuple[str, Path, Dict[int, Tuple[float, float]]]],
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	flight_rows: List[Dict[str, object]] = []
	cleaning_rows: List[Dict[str, object]] = []
	sample_frames: List[pd.DataFrame] = []

	for group, source_dir, windows in groups:
		for flight, window in windows.items():
			print(f"Processing {group} flight_{flight}: {window[0]:.2f} -> {window[1]:.2f} s")
			rows, samples, cleaning_record = process_flight(
				group,
				flight,
				window,
				source_dir,
				output_dir,
				args,
			)
			flight_rows.extend(rows)
			sample_frames.append(samples)
			cleaning_rows.append(cleaning_record)

	if not flight_rows:
		raise SystemExit("No flights processed")

	all_samples = pd.concat(sample_frames, ignore_index=True)
	rollup_rows: List[Dict[str, object]] = []
	for group, group_samples in all_samples.groupby("group", sort=False):
		for regime in SPEED_REGIMES:
			rollup_rows.append(summarize_rollup_regime(group_samples, group, regime))
	for regime in SPEED_REGIMES:
		rollup_rows.append(summarize_rollup_regime(all_samples, "ALL", regime))

	return (
		order_flight_summary_columns(pd.DataFrame(flight_rows)),
		order_rollup_summary_columns(pd.DataFrame(rollup_rows)),
		pd.DataFrame(cleaning_rows),
	)


def save_outputs(
	output_dir: Path,
	flight_summary: pd.DataFrame,
	rollup_summary: pd.DataFrame,
	cleaning_summary: pd.DataFrame,
) -> Tuple[Path, Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	flight_summary_path = output_dir / "31st_july_10s_speed_regime_heading_magnitude_summary_by_flight.csv"
	rollup_summary_path = output_dir / "31st_july_10s_speed_regime_heading_magnitude_summary_by_group.csv"
	cleaning_summary_path = output_dir / "31st_july_10s_speed_regime_velocity_cleaning_summary.csv"
	flight_summary.to_csv(flight_summary_path, index=False)
	rollup_summary.to_csv(rollup_summary_path, index=False)
	cleaning_summary.to_csv(cleaning_summary_path, index=False)
	return flight_summary_path, rollup_summary_path, cleaning_summary_path


def write_readme(output_dir: Path) -> Path:
	readme_path = output_dir / "README.md"
	readme_path.write_text(
		"# 31st July 10s Speed-Regime Heading and Magnitude Metrics\n\n"
		"Each trajectory is cleaned with velocity bounds `-8 <= vel_* <= 8 m/s`, "
		"GT is latency-aligned to estimate time, then the provided 10s evaluation window is clipped. "
		"Regimes are assigned from GT horizontal speed: `sqrt(vel_x_gt^2 + vel_y_gt^2)`.\n\n"
		"Regimes: `low_0_to_0p5` is `0.0 <= GT speed < 0.5 m/s`; "
		"`medium_0p5_to_2` is `0.5 <= GT speed < 2 m/s`; "
		"`high_2_to_5` is `2 <= GT speed < 5 m/s`.\n\n"
		"`mean_heading_error_deg`: Average angle difference between estimate and GT horizontal velocity vectors within the regime.\n\n"
		"`p90_heading_error_deg`: Nearest-rank 90th percentile of per-sample heading error in degrees within the regime.\n\n"
		"`avg_est_abs_magnitude_xy_mps`: Average estimate horizontal speed within the regime, calculated per sample as `sqrt(vel_x_est^2 + vel_y_est^2)`.\n\n"
		"`avg_gt_abs_magnitude_xy_mps`: Average GT horizontal speed within the regime, calculated per sample as `sqrt(vel_x_gt^2 + vel_y_gt^2)`.\n\n"
		"`avg_abs_magnitude_error_mps`: Average absolute horizontal speed difference within the regime, calculated per sample as `abs(est_abs_magnitude_xy - gt_abs_magnitude_xy)`.\n\n"
		"`avg_magnitude_error_pct`: Average percentage horizontal speed error within the regime, calculated per sample as `abs(est_abs_magnitude_xy - gt_abs_magnitude_xy) / gt_abs_magnitude_xy * 100`.\n",
		encoding="utf-8",
	)
	return readme_path


def print_summary(
	flight_summary: pd.DataFrame,
	rollup_summary: pd.DataFrame,
	flight_summary_path: Path,
	rollup_summary_path: Path,
	cleaning_summary_path: Path,
	readme_path: Path,
) -> None:
	columns = [
		"group",
		"speed_regime",
		"trajectories",
		"samples_in_regime",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
		"avg_abs_magnitude_error_mps",
		"avg_magnitude_error_pct",
	]
	per_flight_columns = [
		"group",
		"flight",
		"speed_regime",
		"samples_in_regime",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
		"avg_abs_magnitude_error_mps",
		"avg_magnitude_error_pct",
	]

	print("")
	print(f"Saved per-flight regime summary to {flight_summary_path}")
	print(f"Saved AI/RAW/ALL regime summary to {rollup_summary_path}")
	print(f"Saved cleaning summary to {cleaning_summary_path}")
	print(f"Saved README to {readme_path}")
	print("")
	print("AI/RAW/ALL rolled-up summary by speed regime:")
	print(rollup_summary[columns].to_string(index=False))
	print("")
	print("Per-flight summary by speed regime:")
	print(flight_summary[per_flight_columns].to_string(index=False))


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run all 31st July AI/RAW trajectories through 10s evaluation windows, "
			"clean velocity to +/-8 m/s, split samples by GT horizontal speed regime, "
			"and calculate heading plus magnitude metrics."
		),
	)
	parser.add_argument("--ai_dir", default="31st_July_AI", help="Source directory for AI velocity CSVs")
	parser.add_argument("--raw_dir", default="31st_July_RAW", help="Source directory for RAW velocity CSVs")
	parser.add_argument(
		"--out_dir",
		default="31st_July_speed_regime_heading_magnitude_10s",
		help="Output directory for cleaned, aligned, windowed, and metric CSVs",
	)
	parser.add_argument("--group", choices=["all", "AI", "RAW"], default="all", help="Subset to process")
	parser.add_argument("--clean_window", type=int, default=9, help="Sliding median window size")
	parser.add_argument("--median_tol", type=float, default=None, help="Override median tolerance")
	parser.add_argument(
		"--bound",
		action="append",
		default=None,
		help="Cleaning bounds pattern:min:max. Defaults to vel_*:-8:8",
	)
	parser.add_argument("--fill", choices=["none", "linear"], default="linear", help="NaN fill mode")
	parser.add_argument(
		"--savgol_window",
		type=int,
		default=21,
		help="Savitzky-Golay smoothing window. Use 0 to disable.",
	)
	parser.add_argument("--savgol_polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
	parser.add_argument("--gt_latency", type=float, default=0.025, help="Seconds to subtract from GT time")
	parser.add_argument("--tolerance", type=float, default=None, help="Optional max time delta for alignment")
	args = parser.parse_args()

	args.clean_window = CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = CLEANER._parse_bound_overrides(args.bound or DEFAULT_VELOCITY_BOUNDS)
	if args.savgol_window is not None and args.savgol_window <= 0:
		args.savgol_window = None
	return args


def run() -> None:
	args = parse_arguments()
	output_dir = resolve_repo_path(args.out_dir)
	flight_summary, rollup_summary, cleaning_summary = process_groups(select_groups(args), output_dir, args)
	flight_summary_path, rollup_summary_path, cleaning_summary_path = save_outputs(
		output_dir,
		flight_summary,
		rollup_summary,
		cleaning_summary,
	)
	readme_path = write_readme(output_dir)
	print_summary(
		flight_summary,
		rollup_summary,
		flight_summary_path,
		rollup_summary_path,
		cleaning_summary_path,
		readme_path,
	)


if __name__ == "__main__":
	run()
