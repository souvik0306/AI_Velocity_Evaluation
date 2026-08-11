#!/usr/bin/env python3

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VELOCITY_BOUNDS = ["vel_*:-8:8"]


def _load_script_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


CLEANER = _load_script_module("3_vel_csv_dataset_cleaner.py", "vel_csv_dataset_cleaner")
VHE = _load_script_module("9_velocity_heading_error.py", "velocity_heading_error")


def _clean_csv(
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


def _align_gt_to_est(
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

	gt_aligned = df_sync[gt_cols].rename(columns=rename_map)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	gt_aligned.to_csv(output_path, index=False)
	return output_path


def _clip_window_pair(
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


def _write_vhe_outputs(est_window_path: Path, gt_window_path: Path) -> Tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
	est_df = VHE._load_velocity_csv(est_window_path, VHE.REQUIRED_COLS)
	gt_df = VHE._load_velocity_csv(gt_window_path, VHE.REQUIRED_COLS)
	merged = VHE._merge_velocity_data(est_df, gt_df)
	samples = VHE._compute_sample_metrics(merged)
	summary = VHE._summarize_profiles(samples)

	samples_path = est_window_path.with_name(f"{est_window_path.stem}_vhe_samples{est_window_path.suffix}")
	summary_path = est_window_path.with_name(f"{est_window_path.stem}_vhe_summary{est_window_path.suffix}")
	samples.to_csv(samples_path, index=False)
	summary.to_csv(summary_path, index=False)
	return samples_path, summary_path, summary, samples


def _plot_velocity_components(est_window_path: Path, gt_window_path: Path, group: str, flight_name: str) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.ticker import AutoMinorLocator, MaxNLocator

	est_df = pd.read_csv(est_window_path)
	gt_df = pd.read_csv(gt_window_path)
	merged = pd.merge(
		est_df,
		gt_df,
		on="time",
		how="inner",
		suffixes=("_est", "_gt"),
	).sort_values("time")
	if merged.empty:
		raise ValueError(f"No overlapping timestamps for component plot: {est_window_path}")

	time = merged["time"].to_numpy(dtype=float)
	rel_time = time - time[0]
	out_path = est_window_path.with_name(f"{est_window_path.stem}_velocity_components_est_gt_subplot.png")

	fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
	components = [("x", "v_x"), ("y", "v_y"), ("z", "v_z")]
	for ax, (axis, label) in zip(axes, components):
		ax.plot(
			rel_time,
			merged[f"vel_{axis}_est"].to_numpy(dtype=float),
			label=f"Estimate {label}",
			color="#1f77b4",
			linewidth=2.2,
		)
		ax.plot(
			rel_time,
			merged[f"vel_{axis}_gt"].to_numpy(dtype=float),
			label=f"GT {label}",
			color="#d62728",
			linewidth=2.2,
			alpha=0.9,
		)
		ax.set_ylabel(f"{label} (m/s)", fontsize=13)
		ax.grid(True, which="major", alpha=0.35)
		ax.grid(True, which="minor", alpha=0.15)
		ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
		ax.xaxis.set_minor_locator(AutoMinorLocator(2))
		ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
		ax.yaxis.set_minor_locator(AutoMinorLocator(2))
		ax.legend(loc="best")

	axes[0].set_title(
		f"{group} {flight_name} Cleaned + Time-Aligned Velocity Components Before VHE Calculation",
		fontsize=15,
	)
	axes[-1].set_xlabel("time from evaluation window start (s)", fontsize=13)
	fig.tight_layout()
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return out_path


def _plot_horizontal_velocity(est_window_path: Path, gt_window_path: Path, group: str, flight_name: str) -> Path:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib.ticker import AutoMinorLocator, MaxNLocator

	est_df = pd.read_csv(est_window_path)
	gt_df = pd.read_csv(gt_window_path)
	merged = pd.merge(
		est_df,
		gt_df,
		on="time",
		how="inner",
		suffixes=("_est", "_gt"),
	).sort_values("time")
	if merged.empty:
		raise ValueError(f"No overlapping timestamps for horizontal velocity plot: {est_window_path}")

	time = merged["time"].to_numpy(dtype=float)
	rel_time = time - time[0]
	est_vxy = (
		merged["vel_x_est"].to_numpy(dtype=float) ** 2
		+ merged["vel_y_est"].to_numpy(dtype=float) ** 2
	) ** 0.5
	gt_vxy = (
		merged["vel_x_gt"].to_numpy(dtype=float) ** 2
		+ merged["vel_y_gt"].to_numpy(dtype=float) ** 2
	) ** 0.5
	out_path = est_window_path.with_name(f"{est_window_path.stem}_horizontal_velocity_est_gt.png")

	fig, ax = plt.subplots(figsize=(15, 6.5))
	ax.plot(rel_time, est_vxy, label="Estimate horizontal speed", color="#1f77b4", linewidth=2.3)
	ax.plot(rel_time, gt_vxy, label="GT horizontal speed", color="#d62728", linewidth=2.3, alpha=0.9)

	for speed, color, label in (
		(0.2, "#ff0000", "low start 0.2 m/s"),
		(0.5, "#00a000", "medium start 0.5 m/s"),
		(2.0, "#0040ff", "high start 2.0 m/s"),
	):
		ax.axhline(speed, color=color, linestyle="--", linewidth=1.2, alpha=0.75, label=label)

	ax.set_title(f"{group} {flight_name} Horizontal Velocity Before VHE Calculation", fontsize=15)
	ax.set_xlabel("time from evaluation window start (s)", fontsize=13)
	ax.set_ylabel("horizontal speed sqrt(v_x^2 + v_y^2) (m/s)", fontsize=13)
	ax.grid(True, which="major", alpha=0.35)
	ax.grid(True, which="minor", alpha=0.15)
	ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.legend(loc="best", ncol=2, fontsize=9)
	fig.tight_layout()
	fig.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	return out_path


def _summarize_samples_for_group(
	samples_df: pd.DataFrame,
	group: str,
) -> List[Dict[str, object]]:
	rows = []
	for profile, spec in VHE.PROFILE_SPECS.items():
		profile_df = samples_df[samples_df["speed_profile"] == profile]
		total = int(len(profile_df))
		passing = int(profile_df["sample_pass"].sum()) if total else 0
		rows.append(
			{
				"group": group,
				"profile": profile,
				"profile_label": spec["label"],
				"gt_speed_range_mps": VHE._format_speed_range(spec),
				"heading_limit_deg": spec["heading_limit_deg"],
				"magnitude_limit_pct": spec["mag_limit_pct"],
				"samples": total,
				"passing_samples": passing,
				"pass_pct": passing / total * 100.0 if total else float("nan"),
				"avg_heading_error_deg": profile_df["heading_error_deg"].mean(),
				"avg_magnitude_error_pct": profile_df["magnitude_error_pct"].mean(),
			}
		)
	return rows


def _build_group_profile_summary(sample_frames: List[pd.DataFrame]) -> pd.DataFrame:
	if not sample_frames:
		return pd.DataFrame()

	all_samples = pd.concat(sample_frames, ignore_index=True)
	rows: List[Dict[str, object]] = []
	for group, group_df in all_samples.groupby("group", sort=False):
		rows.extend(_summarize_samples_for_group(group_df, group))
	rows.extend(_summarize_samples_for_group(all_samples, "ALL"))
	return _ordered_group_summary_columns(pd.DataFrame(rows))


def _process_flight(
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

	_clean_csv(
		est_raw,
		est_clean,
		args.clean_window,
		args.median_tol,
		args.bound_overrides,
		args.fill,
		args.savgol_window,
		args.savgol_polyorder,
	)
	_clean_csv(
		gt_raw,
		gt_clean,
		args.clean_window,
		args.median_tol,
		args.bound_overrides,
		args.fill,
		args.savgol_window,
		args.savgol_polyorder,
	)
	_align_gt_to_est(gt_latency=args.gt_latency, tolerance=args.tolerance, est_clean_path=est_clean, gt_clean_path=gt_clean, output_path=gt_aligned)
	est_window, gt_window = _clip_window_pair(
		est_clean,
		gt_aligned,
		window[0],
		window[1],
		"_window",
	)
	_plot_velocity_components(est_window, gt_window, group, flight_name)
	_plot_horizontal_velocity(est_window, gt_window, group, flight_name)
	_, _, vhe_summary, vhe_samples = _write_vhe_outputs(est_window, gt_window)
	vhe_samples = vhe_samples.copy()
	vhe_samples["group"] = group
	vhe_samples["flight"] = flight_name

	rows = []
	for _, row in vhe_summary.iterrows():
		record = row.to_dict()
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

	return rows, vhe_samples


def _ordered_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"flight",
		"profile",
		"profile_label",
		"window_start_s",
		"window_end_s",
		"window_duration_s",
		"gt_speed_range_mps",
		"heading_limit_deg",
		"magnitude_limit_pct",
		"samples",
		"passing_samples",
		"pass_pct",
		"avg_heading_error_deg",
		"avg_magnitude_error_pct",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def _ordered_group_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	front = [
		"group",
		"profile",
		"profile_label",
		"gt_speed_range_mps",
		"heading_limit_deg",
		"magnitude_limit_pct",
		"samples",
		"passing_samples",
		"pass_pct",
		"avg_heading_error_deg",
		"avg_magnitude_error_pct",
	]
	return df[[col for col in front if col in df.columns] + [col for col in df.columns if col not in front]]


def _drop_clean_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
	drop_cols = [
		"profile",
		"profile_label",
	]
	return df.drop(columns=[col for col in drop_cols if col in df.columns])


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run the clean, align, window, plot, and VHE evaluation for the 31st July AI/RAW windows.",
	)
	parser.add_argument("--ai_dir", default="31st_July_AI", help="Source directory for AI CSVs")
	parser.add_argument("--raw_dir", default="31st_July_RAW", help="Source directory for RAW CSVs")
	parser.add_argument(
		"--out_dir",
		default="31st_July_vhe_eval_windows",
		help="Output directory for cleaned, aligned, windowed, and VHE result CSVs",
	)
	parser.add_argument(
		"--group",
		choices=["all", "AI", "RAW"],
		default="all",
		help="Subset to process",
	)
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

	output_dir = Path(args.out_dir)
	all_rows: List[Dict[str, object]] = []
	sample_frames: List[pd.DataFrame] = []

	groups = []
	if args.group in {"all", "AI"}:
		groups.append(("AI", Path(args.ai_dir), AI_WINDOWS))
	if args.group in {"all", "RAW"}:
		groups.append(("RAW", Path(args.raw_dir), RAW_WINDOWS))

	for group, source_dir, windows in groups:
		for flight, window in windows.items():
			print(f"Processing {group} flight_{flight}: {window[0]:.2f} -> {window[1]:.2f} s")
			rows, samples = _process_flight(group, flight, window, source_dir, output_dir, args)
			all_rows.extend(rows)
			sample_frames.append(samples)

	if not all_rows:
		raise SystemExit("No flights processed")

	summary_df_full = _ordered_summary_columns(pd.DataFrame(all_rows))
	group_summary_df_full = _build_group_profile_summary(sample_frames)
	summary_df = _drop_clean_summary_columns(summary_df_full)
	group_summary_df = _drop_clean_summary_columns(group_summary_df_full)
	summary_path = output_dir / "vhe_profile_summary_all_windows.csv"
	group_summary_path = output_dir / "vhe_profile_summary_by_group.csv"
	output_dir.mkdir(parents=True, exist_ok=True)
	summary_df.to_csv(summary_path, index=False)
	group_summary_df.to_csv(group_summary_path, index=False)

	print("")
	print(f"Saved combined VHE profile summary to {summary_path}")
	print(f"Saved AI/RAW profile summary to {group_summary_path}")
	print("")
	print("AI/RAW rolled-up summary:")
	print(
		group_summary_df_full[
			[
				"group",
				"profile_label",
				"samples",
				"passing_samples",
				"pass_pct",
				"avg_heading_error_deg",
				"avg_magnitude_error_pct",
			]
		].to_string(index=False)
	)
	print("")
	print("Per-flight summary:")
	print(
		summary_df_full[
			[
				"group",
				"flight",
				"profile_label",
				"samples",
				"passing_samples",
				"pass_pct",
				"avg_heading_error_deg",
				"avg_magnitude_error_pct",
			]
		].to_string(index=False)
	)


if __name__ == "__main__":
	main()
