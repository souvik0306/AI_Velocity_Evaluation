#!/usr/bin/env python3

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = SCRIPT_DIR / "28th July"
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]


AI_WINDOWS_10S = {
	"ai_10_20_35": (66.20, 76.20),
	"ai_10_31_13": (59.65, 69.65),
	"ai_10_34_48": (48.81, 58.81),
	"ai_10_37_53": (45.39, 55.39),
	"ai_10_40_49": (47.90, 57.90),
	"ai_10_43_37": (41.65, 51.65),
	"ai_10_46_31": (42.18, 52.18),
}

RAW_WINDOWS_10S = {
	"raw_10_18_50": (36.89, 46.89),
	"raw_10_21_43": (38.09, 48.09),
	"raw_10_24_22": (38.54, 48.54),
	"raw_10_29_23": (42.35, 52.35),
	"raw_10_32_09": (41.89, 51.89),
	"raw_10_35_00": (37.46, 47.46),
	"raw_10_37_51": (48.27, 58.27),
}


@dataclass(frozen=True)
class DynamicCriterion:
	name: str
	heading_p90_limit_deg: float
	magnitude_metric: str
	magnitude_limit: float
	magnitude_unit: str


DYNAMIC_CRITERIA = [
	DynamicCriterion(
		name="low_dynamic",
		heading_p90_limit_deg=70.0,
		magnitude_metric="abs",
		magnitude_limit=0.15,
		magnitude_unit="m/s",
	),
	DynamicCriterion(
		name="medium_dynamic",
		heading_p90_limit_deg=50.0,
		magnitude_metric="pct",
		magnitude_limit=60.0,
		magnitude_unit="%",
	),
]


def load_script_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


CLEANER = load_script_module("3_vel_csv_dataset_cleaner.py", "vel_csv_dataset_cleaner")
VHE = load_script_module("9_velocity_heading_error.py", "velocity_heading_error")


def flight_stem_from_label(label: str) -> str:
	_, hour, minute, second = label.split("_")
	return f"flight_2026-07-01-{hour}-{minute}-{second}"


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
	output_suffix: str,
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

	est_out = est_clean_path.with_name(f"{est_clean_path.stem}{output_suffix}{est_clean_path.suffix}")
	gt_out = gt_aligned_path.with_name(f"{gt_aligned_path.stem}{output_suffix}{gt_aligned_path.suffix}")
	est_window.to_csv(est_out, index=False)
	gt_window.to_csv(gt_out, index=False)
	return est_out, gt_out


def calculate_window_samples(est_window_path: Path, gt_window_path: Path) -> pd.DataFrame:
	est_df = VHE._load_velocity_csv(est_window_path, VHE.REQUIRED_COLS)
	gt_df = VHE._load_velocity_csv(gt_window_path, VHE.REQUIRED_COLS)
	merged = VHE._merge_velocity_data(est_df, gt_df)
	samples = VHE._compute_sample_metrics(merged)
	samples["magnitude_error_abs_mps"] = (
		pd.to_numeric(samples["est_speed_xy"], errors="coerce")
		- pd.to_numeric(samples["gt_speed_xy"], errors="coerce")
	).abs()
	return samples


def filter_samples_by_min_horizontal_speed(
	samples: pd.DataFrame,
	min_speed_mps: Optional[float],
	filter_side: str,
) -> pd.DataFrame:
	if min_speed_mps is None:
		result = samples.copy()
		result["pre_speed_filter_samples"] = len(samples)
		result["min_horizontal_speed_mps"] = float("nan")
		result["min_speed_filter_side"] = "none"
		return result

	if min_speed_mps < 0:
		raise ValueError("--min_horizontal_speed_mps must be non-negative")
	if filter_side not in {"both", "est", "gt"}:
		raise ValueError("--min_speed_filter_side must be one of: both, est, gt")

	est_speed = pd.to_numeric(samples["est_speed_xy"], errors="coerce")
	gt_speed = pd.to_numeric(samples["gt_speed_xy"], errors="coerce")
	if filter_side == "est":
		mask = est_speed >= min_speed_mps
	elif filter_side == "gt":
		mask = gt_speed >= min_speed_mps
	else:
		mask = (est_speed >= min_speed_mps) & (gt_speed >= min_speed_mps)
	result = samples[mask].copy()
	result["pre_speed_filter_samples"] = len(samples)
	result["min_horizontal_speed_mps"] = min_speed_mps
	result["min_speed_filter_side"] = filter_side
	return result


def pre_speed_filter_count(samples: pd.DataFrame) -> int:
	if "pre_speed_filter_samples" not in samples.columns:
		return int(len(samples))
	if "window_label" in samples.columns:
		id_cols = [col for col in ("group", "window_label", "flight") if col in samples.columns]
		return int(samples[id_cols + ["pre_speed_filter_samples"]].drop_duplicates()["pre_speed_filter_samples"].sum())
	return int(samples["pre_speed_filter_samples"].iloc[0]) if not samples.empty else 0


def summary_constant(samples: pd.DataFrame, column: str, default: object) -> object:
	if column not in samples.columns or samples.empty:
		return default
	values = samples[column].dropna().unique()
	if len(values) == 1:
		return values[0]
	if len(values) == 0:
		return default
	return "mixed"


def summarize_samples(samples: pd.DataFrame, criterion: DynamicCriterion) -> Dict[str, object]:
	heading_values = pd.to_numeric(samples["heading_error_deg"], errors="coerce")
	p90_heading = VHE._nearest_rank_percentile(heading_values, 90.0)
	mean_heading = heading_values.mean()
	if criterion.magnitude_metric == "abs":
		mag_values = pd.to_numeric(samples["magnitude_error_abs_mps"], errors="coerce")
	else:
		mag_values = pd.to_numeric(samples["magnitude_error_pct"], errors="coerce")
	avg_magnitude = mag_values.mean()

	heading_pass = (
		bool(p90_heading < criterion.heading_p90_limit_deg)
		if np.isfinite(p90_heading)
		else False
	)
	magnitude_pass = (
		bool(avg_magnitude < criterion.magnitude_limit)
		if np.isfinite(avg_magnitude)
		else False
	)

	return {
		"dynamic_case": criterion.name,
		"pre_speed_filter_samples": pre_speed_filter_count(samples),
		"min_horizontal_speed_mps": summary_constant(samples, "min_horizontal_speed_mps", float("nan")),
		"min_speed_filter_side": summary_constant(samples, "min_speed_filter_side", "none"),
		"samples": int(len(samples)),
		"valid_heading_samples": int(samples["heading_error_deg"].notna().sum()),
		"valid_magnitude_samples": int(mag_values.notna().sum()),
		"heading_p90_limit_deg": criterion.heading_p90_limit_deg,
		"p90_heading_error_deg": p90_heading,
		"mean_heading_error_deg": mean_heading,
		"heading_pass": heading_pass,
		"avg_magnitude_error": avg_magnitude,
		"magnitude_metric": criterion.magnitude_metric,
		"magnitude_limit": criterion.magnitude_limit,
		"magnitude_unit": criterion.magnitude_unit,
		"magnitude_pass": magnitude_pass,
		"window_pass": heading_pass and magnitude_pass,
	}


def process_window(
	group: str,
	label: str,
	window: Tuple[float, float],
	source_dir: Path,
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], pd.DataFrame]:
	flight_stem = flight_stem_from_label(label)
	est_raw = source_dir / f"{flight_stem}_vel_est.csv"
	gt_raw = source_dir / f"{flight_stem}_vel_gt.csv"
	if not est_raw.exists():
		raise FileNotFoundError(f"Estimate source CSV not found: {est_raw}")
	if not gt_raw.exists():
		raise FileNotFoundError(f"GT source CSV not found: {gt_raw}")

	group_out = output_dir / group
	est_clean = group_out / f"{flight_stem}_vel_est_clean.csv"
	gt_clean = group_out / f"{flight_stem}_vel_gt_clean.csv"
	gt_aligned = group_out / f"{flight_stem}_vel_gt_clean_aligned.csv"

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
	est_window, gt_window = clip_velocity_window(
		est_clean,
		gt_aligned,
		window[0],
		window[1],
		"_window",
	)

	samples = filter_samples_by_min_horizontal_speed(
		calculate_window_samples(est_window, gt_window),
		args.min_horizontal_speed_mps,
		args.min_speed_filter_side,
	)
	if samples.empty:
		raise ValueError(
			f"No samples left for {label} after min horizontal speed filter "
			f"{args.min_horizontal_speed_mps} m/s"
		)
	samples = samples.copy()
	samples["group"] = group
	samples["window_label"] = label
	samples["flight"] = flight_stem
	samples["window_start_s"] = window[0]
	samples["window_end_s"] = window[1]

	samples_path = est_window.with_name(f"{est_window.stem}_low_medium_vhe_samples{est_window.suffix}")
	samples.to_csv(samples_path, index=False)

	rows = []
	for criterion in DYNAMIC_CRITERIA:
		record = summarize_samples(samples, criterion)
		record.update(
			{
				"group": group,
				"window_label": label,
				"flight": flight_stem,
				"window_start_s": window[0],
				"window_end_s": window[1],
				"window_duration_s": window[1] - window[0],
			}
		)
		rows.append(record)

	summary_path = est_window.with_name(f"{est_window.stem}_low_medium_vhe_summary{est_window.suffix}")
	order_summary_columns(pd.DataFrame(rows)).to_csv(summary_path, index=False)
	return rows, samples


def summarize_samples_by_group(samples_df: pd.DataFrame, group: str) -> List[Dict[str, object]]:
	rows = []
	for criterion in DYNAMIC_CRITERIA:
		record = summarize_samples(samples_df, criterion)
		record["group"] = group
		rows.append(record)
	return rows


def summarize_groups(sample_frames: List[pd.DataFrame]) -> pd.DataFrame:
	if not sample_frames:
		return pd.DataFrame()
	all_samples = pd.concat(sample_frames, ignore_index=True)
	rows: List[Dict[str, object]] = []
	for group, group_df in all_samples.groupby("group", sort=False):
		rows.extend(summarize_samples_by_group(group_df, group))
	rows.extend(summarize_samples_by_group(all_samples, "ALL"))
	return order_group_summary_columns(pd.DataFrame(rows))


def order_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"window_label",
		"flight",
		"dynamic_case",
		"window_start_s",
		"window_end_s",
		"window_duration_s",
		"pre_speed_filter_samples",
		"min_horizontal_speed_mps",
		"min_speed_filter_side",
		"samples",
		"valid_heading_samples",
		"valid_magnitude_samples",
		"heading_p90_limit_deg",
		"p90_heading_error_deg",
		"mean_heading_error_deg",
		"heading_pass",
		"avg_magnitude_error",
		"magnitude_metric",
		"magnitude_limit",
		"magnitude_unit",
		"magnitude_pass",
		"window_pass",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def order_group_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"dynamic_case",
		"pre_speed_filter_samples",
		"min_horizontal_speed_mps",
		"min_speed_filter_side",
		"samples",
		"valid_heading_samples",
		"valid_magnitude_samples",
		"heading_p90_limit_deg",
		"p90_heading_error_deg",
		"mean_heading_error_deg",
		"heading_pass",
		"avg_magnitude_error",
		"magnitude_metric",
		"magnitude_limit",
		"magnitude_unit",
		"magnitude_pass",
		"window_pass",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run 28th July 10s full-window velocity direction/magnitude evaluation "
			"for low- and medium-dynamic criteria."
		),
	)
	parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR), help="28th July base directory")
	parser.add_argument("--ai_dir", default=None, help="Source directory for AI CSVs")
	parser.add_argument("--raw_dir", default=None, help="Source directory for RAW CSVs")
	parser.add_argument(
		"--out_dir",
		default=None,
		help="Output directory for cleaned, aligned, windowed, and VHE result CSVs",
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
	parser.add_argument(
		"--min_horizontal_speed_mps",
		type=float,
		default=None,
		help="Drop samples below this horizontal speed according to --min_speed_filter_side",
	)
	parser.add_argument(
		"--min_speed_filter_side",
		choices=["both", "est", "gt"],
		default="both",
		help="Which horizontal speed to apply --min_horizontal_speed_mps to",
	)
	parser.add_argument(
		"--ai_windows",
		default=None,
		help="Comma-separated AI windows to process by 1-based index or label, e.g. 1,3,5",
	)
	parser.add_argument(
		"--raw_windows",
		default=None,
		help="Comma-separated RAW windows to process by 1-based index or label, e.g. 7 or raw_10_37_51",
	)
	args = parser.parse_args()

	base_dir = Path(args.base_dir)
	args.ai_dir = Path(args.ai_dir) if args.ai_dir else base_dir / "28th_July_AI_rosbags"
	args.raw_dir = Path(args.raw_dir) if args.raw_dir else base_dir / "28th_July_RAW_rosbags"
	args.out_dir = Path(args.out_dir) if args.out_dir else base_dir / "28th_July_low_medium_vhe_10s"
	args.clean_window = CLEANER._normalize_window(args.clean_window)
	args.bound_overrides = CLEANER._parse_bound_overrides(args.bound or DEFAULT_VELOCITY_BOUNDS)
	return args


def select_windows(
	windows: Dict[str, Tuple[float, float]],
	selection: Optional[str],
	group: str,
) -> Dict[str, Tuple[float, float]]:
	if not selection:
		return windows

	items = list(windows.items())
	selected: Dict[str, Tuple[float, float]] = {}
	for raw_token in selection.split(","):
		token = raw_token.strip()
		if not token:
			continue
		if token.isdigit():
			index = int(token)
			if index < 1 or index > len(items):
				raise ValueError(f"{group} window index {index} is outside 1..{len(items)}")
			label, window = items[index - 1]
		else:
			label = token
			if label not in windows:
				valid = ", ".join(windows.keys())
				raise ValueError(f"Unknown {group} window '{label}'. Valid labels: {valid}")
			window = windows[label]
		selected[label] = window

	if not selected:
		raise ValueError(f"No {group} windows selected from '{selection}'")
	return selected


def select_groups(args: argparse.Namespace) -> List[Tuple[str, Path, Dict[str, Tuple[float, float]]]]:
	groups = []
	if args.group in {"all", "AI"}:
		groups.append(("AI", args.ai_dir, select_windows(AI_WINDOWS_10S, args.ai_windows, "AI")))
	if args.group in {"all", "RAW"}:
		groups.append(("RAW", args.raw_dir, select_windows(RAW_WINDOWS_10S, args.raw_windows, "RAW")))
	return groups


def process_groups(
	groups: List[Tuple[str, Path, Dict[str, Tuple[float, float]]]],
	output_dir: Path,
	args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	all_rows: List[Dict[str, object]] = []
	sample_frames: List[pd.DataFrame] = []

	for group, source_dir, windows in groups:
		for label, window in windows.items():
			print(f"Processing {group} {label}: {window[0]:.2f} -> {window[1]:.2f} s")
			rows, samples = process_window(group, label, window, source_dir, output_dir, args)
			all_rows.extend(rows)
			sample_frames.append(samples)

	if not all_rows:
		raise SystemExit("No windows processed")

	return order_summary_columns(pd.DataFrame(all_rows)), summarize_groups(sample_frames)


def save_outputs(output_dir: Path, flight_summary: pd.DataFrame, group_summary: pd.DataFrame) -> Tuple[Path, Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	summary_path = output_dir / "28th_july_low_medium_vhe_summary_all_windows.csv"
	group_summary_path = output_dir / "28th_july_low_medium_vhe_summary_by_group.csv"
	flight_summary.to_csv(summary_path, index=False)
	group_summary.to_csv(group_summary_path, index=False)
	return summary_path, group_summary_path


def print_summary_tables(
	flight_summary: pd.DataFrame,
	group_summary: pd.DataFrame,
	summary_path: Path,
	group_summary_path: Path,
) -> None:
	columns = [
		"group",
		"dynamic_case",
		"pre_speed_filter_samples",
		"min_horizontal_speed_mps",
		"min_speed_filter_side",
		"samples",
		"heading_p90_limit_deg",
		"p90_heading_error_deg",
		"mean_heading_error_deg",
		"heading_pass",
		"avg_magnitude_error",
		"magnitude_metric",
		"magnitude_limit",
		"magnitude_unit",
		"magnitude_pass",
		"window_pass",
	]
	per_window_columns = ["window_label", "flight"] + columns

	print("")
	print(f"Saved per-window VHE summary to {summary_path}")
	print(f"Saved AI/RAW rolled-up VHE summary to {group_summary_path}")
	print("")
	print("AI/RAW rolled-up summary:")
	print(group_summary[columns].to_string(index=False))
	print("")
	print("Per-window summary:")
	print(flight_summary[per_window_columns].to_string(index=False))


def run() -> None:
	args = parse_arguments()
	flight_summary, group_summary = process_groups(select_groups(args), args.out_dir, args)
	summary_path, group_summary_path = save_outputs(args.out_dir, flight_summary, group_summary)
	print_summary_tables(flight_summary, group_summary, summary_path, group_summary_path)


if __name__ == "__main__":
	run()
