#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-4:4"]
METRIC_REQUIRED_COLS = ["time", "vel_x", "vel_y"]


AI_WINDOWS = {
	"ai_10_20_35": (66.20, 76.20),
	"ai_10_31_13": (59.65, 69.65),
	"ai_10_34_48": (48.81, 58.81),
	"ai_10_37_53": (45.39, 55.39),
	"ai_10_40_49": (47.90, 57.90),
	"ai_10_43_37": (41.65, 51.65),
	"ai_10_46_31": (42.18, 52.18),
}

RAW_FLIGHT_LABELS_ALL = {
	1: "raw_10_18_50",
	2: "raw_10_21_43",
	3: "raw_10_24_22",
	4: "raw_10_29_23",
	5: "raw_10_32_09",
	6: "raw_10_35_00",
	7: "raw_10_37_51",
}

RAW_WINDOWS_ALL = {
	"raw_10_18_50": (36.89, 46.89),
	"raw_10_21_43": (38.09, 48.09),
	"raw_10_24_22": (38.54, 48.54),
	"raw_10_29_23": (42.35, 52.35),
	"raw_10_32_09": (41.89, 51.89),
	"raw_10_35_00": (37.46, 47.46),
	"raw_10_37_51": (48.27, 58.27),
}


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


def flight_stem_from_label(label: str) -> str:
	_, hour, minute, second = label.split("_")
	return f"flight_2026-07-01-{hour}-{minute}-{second}"


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
	samples["abs_magnitude_error_mps"] = abs_magnitude_error
	samples["magnitude_error_pct"] = magnitude_error_pct
	samples["heading_error_deg"] = heading_error_deg
	return samples


def summarize_samples(
	samples: pd.DataFrame,
	group: str,
	eval_id: str,
	flight_stem: str,
	window: Tuple[float, float],
) -> Dict[str, object]:
	return {
		"group": group,
		"eval_id": eval_id,
		"flight_stem": flight_stem,
		"window_start_s": window[0],
		"window_end_s": window[1],
		"window_duration_s": window[1] - window[0],
		"samples": int(len(samples)),
		"valid_heading_samples": int(samples["heading_error_deg"].notna().sum()),
		"valid_magnitude_samples": int(samples["magnitude_error_pct"].notna().sum()),
		"mean_heading_error_deg": pd.to_numeric(samples["heading_error_deg"], errors="coerce").mean(),
		"p90_heading_error_deg": nearest_rank_percentile(samples["heading_error_deg"], 90.0),
		"avg_est_abs_magnitude_xy_mps": pd.to_numeric(
			samples["est_abs_magnitude_xy_mps"], errors="coerce"
		).mean(),
		"avg_gt_abs_magnitude_xy_mps": pd.to_numeric(
			samples["gt_abs_magnitude_xy_mps"], errors="coerce"
		).mean(),
		"avg_abs_magnitude_error_mps": pd.to_numeric(
			samples["abs_magnitude_error_mps"], errors="coerce"
		).mean(),
		"avg_magnitude_error_pct": pd.to_numeric(samples["magnitude_error_pct"], errors="coerce").mean(),
	}


def summarize_rollup(samples: pd.DataFrame, group: str) -> Dict[str, object]:
	row = summarize_samples(samples, group, "ALL_TRAJECTORIES", "ALL_TRAJECTORIES", (float("nan"), float("nan")))
	row["trajectories"] = int(samples[["group", "eval_id"]].drop_duplicates().shape[0])
	row.pop("window_start_s")
	row.pop("window_end_s")
	row.pop("window_duration_s")
	return row


def order_flight_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"eval_id",
		"flight_stem",
		"window_start_s",
		"window_end_s",
		"window_duration_s",
		"samples",
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
		"eval_id",
		"trajectories",
		"samples",
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
	eval_id: str,
	window: Tuple[float, float],
	source_dir: Path,
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, object]]:
	flight_stem = flight_stem_from_label(eval_id)
	est_raw = source_dir / f"{flight_stem}_vel_est.csv"
	gt_raw = source_dir / f"{flight_stem}_vel_gt.csv"
	if not est_raw.exists():
		raise FileNotFoundError(f"Estimate source CSV not found: {est_raw}")
	if not gt_raw.exists():
		raise FileNotFoundError(f"GT source CSV not found: {gt_raw}")

	group_out = output_dir / group
	est_clean = group_out / f"{eval_id}_vel_est_clean.csv"
	gt_clean = group_out / f"{eval_id}_vel_gt_clean.csv"
	gt_aligned = group_out / f"{eval_id}_vel_gt_clean_aligned.csv"

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
	samples.insert(0, "flight_stem", flight_stem)
	samples.insert(0, "eval_id", eval_id)
	samples.insert(0, "group", group)
	samples["window_start_s"] = window[0]
	samples["window_end_s"] = window[1]

	samples_path = group_out / f"{eval_id}_10s_heading_magnitude_samples.csv"
	samples.to_csv(samples_path, index=False)
	summary = summarize_samples(samples, group, eval_id, flight_stem, window)

	cleaning_record = {
		"group": group,
		"eval_id": eval_id,
		"flight_stem": flight_stem,
		"est_rows_cleaned": int(len(pd.read_csv(est_clean))),
		"gt_rows_cleaned": int(len(pd.read_csv(gt_clean))),
	}
	for col in sorted(set(est_pruned) | set(est_filtered) | set(gt_pruned) | set(gt_filtered)):
		cleaning_record[f"est_pruned_{col}"] = est_pruned.get(col, 0)
		cleaning_record[f"est_median_filtered_{col}"] = est_filtered.get(col, 0)
		cleaning_record[f"gt_pruned_{col}"] = gt_pruned.get(col, 0)
		cleaning_record[f"gt_median_filtered_{col}"] = gt_filtered.get(col, 0)

	return summary, samples, cleaning_record


def parse_raw_flights(value: str) -> List[int]:
	flights: List[int] = []
	for item in value.split(","):
		item = item.strip()
		if not item:
			continue
		try:
			flight = int(item)
		except ValueError as exc:
			raise ValueError(f"Invalid RAW flight number '{item}' in --raw_flights") from exc
		if flight not in RAW_FLIGHT_LABELS_ALL:
			valid = ", ".join(str(key) for key in sorted(RAW_FLIGHT_LABELS_ALL))
			raise ValueError(f"RAW flight {flight} is not valid. Use one or more of: {valid}")
		if flight not in flights:
			flights.append(flight)
	if not flights:
		raise ValueError("--raw_flights must include at least one RAW flight number")
	return flights


def select_raw_windows(raw_flights: List[int]) -> Dict[str, Tuple[float, float]]:
	return {
		RAW_FLIGHT_LABELS_ALL[flight]: RAW_WINDOWS_ALL[RAW_FLIGHT_LABELS_ALL[flight]]
		for flight in raw_flights
	}


def select_groups(args: argparse.Namespace) -> List[Tuple[str, Path, Dict[str, Tuple[float, float]]]]:
	groups = []
	if args.group in {"all", "AI"}:
		groups.append(("AI", resolve_repo_path(args.ai_dir), AI_WINDOWS))
	if args.group in {"all", "RAW"}:
		groups.append(("RAW", resolve_repo_path(args.raw_dir), select_raw_windows(args.raw_flights)))
	return groups


def describe_groups(groups: List[Tuple[str, Path, Dict[str, Tuple[float, float]]]]) -> str:
	parts = []
	for group, _, windows in groups:
		parts.append(f"{group}: {', '.join(windows.keys())}")
	return "; ".join(parts)


def process_groups(
	groups: List[Tuple[str, Path, Dict[str, Tuple[float, float]]]],
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	flight_rows: List[Dict[str, object]] = []
	cleaning_rows: List[Dict[str, object]] = []
	sample_frames: List[pd.DataFrame] = []

	for group, source_dir, windows in groups:
		for eval_id, window in windows.items():
			print(f"Processing {group} {eval_id}: {window[0]:.2f} -> {window[1]:.2f} s")
			summary, samples, cleaning_record = process_flight(
				group,
				eval_id,
				window,
				source_dir,
				output_dir,
				args,
			)
			flight_rows.append(summary)
			sample_frames.append(samples)
			cleaning_rows.append(cleaning_record)

	if not flight_rows:
		raise SystemExit("No flights processed")

	all_samples = pd.concat(sample_frames, ignore_index=True)
	rollup_rows = []
	for group, group_samples in all_samples.groupby("group", sort=False):
		rollup_rows.append(summarize_rollup(group_samples, group))
	rollup_rows.append(summarize_rollup(all_samples, "ALL"))

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
	flight_summary_path = output_dir / "28th_july_10s_heading_magnitude_summary_by_flight.csv"
	rollup_summary_path = output_dir / "28th_july_10s_heading_magnitude_summary_by_group.csv"
	cleaning_summary_path = output_dir / "28th_july_10s_velocity_cleaning_summary.csv"
	flight_summary.to_csv(flight_summary_path, index=False)
	rollup_summary.to_csv(rollup_summary_path, index=False)
	cleaning_summary.to_csv(cleaning_summary_path, index=False)
	return flight_summary_path, rollup_summary_path, cleaning_summary_path


def write_readme(output_dir: Path, selection_description: str) -> Path:
	readme_path = output_dir / "README.md"
	readme_path.write_text(
		"# 28th July 10s Heading and Magnitude Metrics\n\n"
		f"This run includes {selection_description}. "
		"Each trajectory is cleaned with velocity bounds `-4 <= vel_* <= 4 m/s`, "
		"GT is latency-aligned to estimate time, then the provided 10s evaluation window is clipped. "
		"Heading and magnitude metrics use horizontal velocity only: `vel_x` and `vel_y`.\n\n"
		"`mean_heading_error_deg`: Average angle difference between estimate and GT horizontal velocity vectors.\n\n"
		"`p90_heading_error_deg`: Nearest-rank 90th percentile of per-sample heading error in degrees.\n\n"
		"`avg_est_abs_magnitude_xy_mps`: Average estimate horizontal speed, calculated per sample as "
		"`sqrt(vel_x_est^2 + vel_y_est^2)`.\n\n"
		"`avg_gt_abs_magnitude_xy_mps`: Average GT horizontal speed, calculated per sample as "
		"`sqrt(vel_x_gt^2 + vel_y_gt^2)`.\n\n"
		"`avg_abs_magnitude_error_mps`: Average absolute horizontal speed difference, calculated per sample as "
		"`abs(est_abs_magnitude_xy - gt_abs_magnitude_xy)`.\n\n"
		"`avg_magnitude_error_pct`: Average percentage horizontal speed error, calculated per sample as "
		"`abs(est_abs_magnitude_xy - gt_abs_magnitude_xy) / gt_abs_magnitude_xy * 100`.\n",
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
		"eval_id",
		"samples",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
		"avg_abs_magnitude_error_mps",
		"avg_magnitude_error_pct",
	]
	rollup_columns = [
		"group",
		"trajectories",
		"samples",
		"mean_heading_error_deg",
		"p90_heading_error_deg",
		"avg_abs_magnitude_error_mps",
		"avg_magnitude_error_pct",
	]

	print("")
	print(f"Saved per-flight summary to {flight_summary_path}")
	print(f"Saved AI/RAW/ALL summary to {rollup_summary_path}")
	print(f"Saved cleaning summary to {cleaning_summary_path}")
	print(f"Saved README to {readme_path}")
	print("")
	print("AI/RAW/ALL rolled-up summary:")
	print(rollup_summary[rollup_columns].to_string(index=False))
	print("")
	print("Per-flight summary:")
	print(flight_summary[columns].to_string(index=False))


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run all 28th July AI/RAW trajectories through the 10s evaluation windows, "
			"clean velocity to +/-4 m/s, and calculate heading plus magnitude metrics."
		),
	)
	parser.add_argument(
		"--ai_dir",
		default="28th July/28th_July_AI_rosbags",
		help="Source directory for AI velocity CSVs",
	)
	parser.add_argument(
		"--raw_dir",
		default="28th July/28th_July_RAW_rosbags",
		help="Source directory for RAW velocity CSVs",
	)
	parser.add_argument(
		"--out_dir",
		default="28th July/28th_July_heading_magnitude_10s_ai_all_raw7",
		help="Output directory for cleaned, aligned, windowed, and metric CSVs",
	)
	parser.add_argument("--group", choices=["all", "AI", "RAW"], default="all", help="Subset to process")
	parser.add_argument(
		"--raw_flights",
		default="7",
		help=(
			"Comma-separated RAW flight ordinals to process, using the provided order "
			"1=raw_10_18_50 through 7=raw_10_37_51. Default: 7"
		),
	)
	parser.add_argument("--clean_window", type=int, default=9, help="Sliding median window size")
	parser.add_argument("--median_tol", type=float, default=None, help="Override median tolerance")
	parser.add_argument(
		"--bound",
		action="append",
		default=None,
		help="Cleaning bounds pattern:min:max. Defaults to vel_*:-4:4",
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
	args.raw_flights = parse_raw_flights(args.raw_flights)
	if args.savgol_window is not None and args.savgol_window <= 0:
		args.savgol_window = None
	return args


def run() -> None:
	args = parse_arguments()
	output_dir = resolve_repo_path(args.out_dir)
	groups = select_groups(args)
	flight_summary, rollup_summary, cleaning_summary = process_groups(groups, output_dir, args)
	flight_summary_path, rollup_summary_path, cleaning_summary_path = save_outputs(
		output_dir,
		flight_summary,
		rollup_summary,
		cleaning_summary,
	)
	readme_path = write_readme(output_dir, describe_groups(groups))
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
