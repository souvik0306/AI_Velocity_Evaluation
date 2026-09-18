#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = ROOT / "scripts" / "pipeline"
VELOCITY_BOUND = "vel_*:-8:8"
MIN_VELOCITY = -8.0
MAX_VELOCITY = 8.0

AI_WINDOWS_10S = {
	1: (36.67, 46.67),
	2: (37.12, 47.12),
	3: (37.55, 47.55),
	4: (34.63, 44.63),
	5: (33.57, 43.57),
	6: (32.35, 42.35),
	7: (33.98, 43.98),
	8: (36.93, 46.93),
}

RAW_WINDOWS_10S = {
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


def extend_windows(windows: Dict[int, Tuple[float, float]], duration_s: float) -> Dict[int, Tuple[float, float]]:
	return {flight: (start, start + duration_s) for flight, (start, _) in windows.items()}


def run_command(command: List[str], log_path: Path, cwd: Path) -> None:
	with log_path.open("a") as log:
		log.write("Running: " + " ".join(command) + "\n")
		result = subprocess.run(command, cwd=cwd, text=True, stdout=log, stderr=subprocess.STDOUT)
	if result.returncode != 0:
		raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def enforce_velocity_bounds(csv_path: Path) -> None:
	df = pd.read_csv(csv_path)
	velocity_cols = [col for col in df.columns if col.startswith("vel_")]
	if not velocity_cols:
		return
	for col in velocity_cols:
		df[col] = pd.to_numeric(df[col], errors="coerce").clip(MIN_VELOCITY, MAX_VELOCITY)
	df.to_csv(csv_path, index=False)


def clean_and_align_source(source_dir: Path, flights: Iterable[int], log_path: Path, cwd: Path) -> None:
	for flight in flights:
		base = f"flight_{flight}"
		est_csv = source_dir / f"{base}_vel_est.csv"
		gt_csv = source_dir / f"{base}_vel_gt.csv"
		est_clean = source_dir / f"{base}_vel_est_clean.csv"
		gt_clean = source_dir / f"{base}_vel_gt_clean.csv"

		run_command(
			[
				"python3",
				str(PIPELINE_DIR / "3_vel_csv_dataset_cleaner.py"),
				"--est_csv",
				str(est_csv),
				"--gt_csv",
				str(gt_csv),
				"--bound",
				VELOCITY_BOUND,
			],
			log_path,
			cwd,
		)
		enforce_velocity_bounds(est_clean)
		enforce_velocity_bounds(gt_clean)
		run_command(
			[
				"python3",
				str(PIPELINE_DIR / "4_align_vel_csv.py"),
				"--est_csv",
				str(est_clean),
				"--gt_csv",
				str(gt_clean),
			],
			log_path,
			cwd,
		)


def clip_and_score_window(
	source_dir: Path,
	out_dir: Path,
	group: str,
	duration_label: str,
	flight: int,
	window: Tuple[float, float],
	log_path: Path,
) -> Tuple[str, bool]:
	base = f"flight_{flight}"
	est_clean = source_dir / f"{base}_vel_est_clean.csv"
	gt_aligned = source_dir / f"{base}_vel_gt_clean_aligned.csv"
	est_window = out_dir / f"{base}_vel_est_clean_window.csv"
	gt_window = out_dir / f"{base}_vel_gt_clean_aligned_window.csv"
	start, end = window

	try:
		run_command(
			[
				"python3",
				str(PIPELINE_DIR / "5_manual_relative_window_clipping.py"),
				"--est_csv",
				str(est_clean),
				"--gt_csv",
				str(gt_aligned),
				"--window_start",
				f"{start:.2f}",
				"--window_end",
				f"{end:.2f}",
				"--output_suffix",
				"_window",
			],
			log_path,
			out_dir,
		)
		run_command(
			[
				"python3",
				str(PIPELINE_DIR / "6_bias_zeroing_and_rmse.py"),
				"--est_csv",
				str(est_window),
				"--gt_csv",
				str(gt_window),
			],
			log_path,
			out_dir,
		)
	except Exception as exc:
		return f"FAIL {group} {duration_label} {base}: {exc}", False

	return f"OK {group} {duration_label} {base}", True


def parse_rmse(path: Path) -> Tuple[float, float, float, float]:
	text = path.read_text()
	values = [float(value) for _, value in re.findall(r"RMSE_(x|y|z|xy)=([0-9eE+\-.]+)", text)]
	if len(values) < 8:
		raise ValueError(f"Unexpected RMSE format in {path}")
	return values[4], values[5], values[6], values[7]


def calculate_drift_rate(path: Path) -> float:
	df = pd.read_csv(path)
	t = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
	ex = pd.to_numeric(df["err_x"], errors="coerce").to_numpy(dtype=float)
	ey = pd.to_numeric(df["err_y"], errors="coerce").to_numpy(dtype=float)
	mask = np.isfinite(t) & np.isfinite(ex) & np.isfinite(ey)
	t = t[mask]
	ex = ex[mask]
	ey = ey[mask]
	if t.size < 2:
		return float("nan")
	error_xy = np.sqrt(ex**2 + ey**2)
	slope, _ = np.polyfit(t - t[0], error_xy, 1)
	return float(slope)


def write_metrics_summary(
	out_dir: Path,
	group: str,
	duration_label: str,
	windows: Dict[int, Tuple[float, float]],
) -> Path:
	summary_path = out_dir / f"31st_July_{group}_metrics_summary_{duration_label}.txt"
	lines = [
		f"31st July {group} Metrics Summary ({duration_label} window, velocity bounds -8..8 m/s)",
		"Columns: file_name | window_start_s | window_end_s | RMSE_x_bias_corrected | RMSE_y_bias_corrected | RMSE_z_bias_corrected | RMSE_xy_bias_corrected | drift_rate_mps2",
		"",
	]

	for flight, (start, end) in windows.items():
		base = f"flight_{flight}"
		rmse_path = out_dir / f"{base}_vel_est_clean_window_rmse_summary.txt"
		bias_path = out_dir / f"{base}_vel_est_clean_window_bias_errors.csv"
		if not rmse_path.exists() or not bias_path.exists():
			lines.append(f"{base}.bag | {start:.2f} | {end:.2f} | MISSING_OUTPUTS")
			continue
		rx, ry, rz, rxy = parse_rmse(rmse_path)
		drift = calculate_drift_rate(bias_path)
		lines.append(
			f"{base}.bag | {start:.2f} | {end:.2f} | "
			f"{rx:.6f} | {ry:.6f} | {rz:.6f} | {rxy:.6f} | {drift:.6e}"
		)

	summary_path.write_text("\n".join(lines) + "\n")
	return summary_path


def process_duration(
	group: str,
	source_dir: Path,
	out_dir: Path,
	duration_label: str,
	windows: Dict[int, Tuple[float, float]],
) -> List[str]:
	out_dir.mkdir(parents=True, exist_ok=True)
	log_path = out_dir / f"31st_July_{group.lower()}_{duration_label}_wide_bounds.log"
	failure_path = out_dir / f"31st_July_{group.lower()}_{duration_label}_wide_bounds_failures.log"
	log_path.write_text("")
	failure_path.write_text("")

	clean_and_align_source(source_dir, windows.keys(), log_path, out_dir)

	statuses = []
	for flight, window in windows.items():
		print(f"Processing {group} {duration_label} flight_{flight}: {window[0]:.2f} -> {window[1]:.2f}")
		status, ok = clip_and_score_window(source_dir, out_dir, group, duration_label, flight, window, log_path)
		statuses.append(status)
		if not ok:
			with failure_path.open("a") as failures:
				failures.write(status + "\n")

	summary_path = write_metrics_summary(out_dir, group, duration_label, windows)
	statuses.append(f"Wrote {summary_path}")
	return statuses


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Rerun 31st July AI/RAW 10s RMSE/drift evaluations with -8..8 m/s velocity cleaning bounds.",
	)
	parser.add_argument("--group", choices=["all", "AI", "RAW"], default="all")
	parser.add_argument("--duration", choices=["all", "10s", "25s"], default="10s")
	return parser.parse_args()


def run() -> None:
	args = parse_arguments()
	configs = []
	if args.group in {"all", "AI"}:
		configs.append(("AI", ROOT / "31st_July_AI", AI_WINDOWS_10S))
	if args.group in {"all", "RAW"}:
		configs.append(("RAW", ROOT / "31st_July_RAW", RAW_WINDOWS_10S))

	for group, source_dir, windows_10s in configs:
		if args.duration in {"all", "10s"}:
			out_dir = ROOT / "31st_July_reeval_10s" / group
			for status in process_duration(group, source_dir, out_dir, "10s", windows_10s):
				print(status)
		if args.duration in {"all", "25s"}:
			out_dir = ROOT / "31st_July_reeval_25s" / group
			windows_25s = extend_windows(windows_10s, 25.0)
			for status in process_duration(group, source_dir, out_dir, "25s", windows_25s):
				print(status)


if __name__ == "__main__":
	run()
