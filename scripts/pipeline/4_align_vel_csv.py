#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


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
		description="Compensate GT latency and align GT velocity to estimate timeline.",
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
		default=None,
		help="Optional max time delta (seconds) for merge_asof",
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

	# Step 4: align to the estimate timeline using nearest-neighbor matching.
	merge_kwargs = {
		"on": args.time_col,
		"direction": "nearest",
		"suffixes": ("_est", "_gt"),
	}
	if args.tolerance is not None:
		merge_kwargs["tolerance"] = args.tolerance

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
		raise KeyError(
			"No GT columns found after merge. "
			f"GT columns: {list(df_gt.columns)}; merged columns: {list(df_sync.columns)}"
		)
	gt_aligned = df_sync[gt_cols].rename(columns=rename_map)
	gt_aligned.to_csv(output_path, index=False)

	print(f"Saved aligned GT CSV to {output_path}")
	print(f"Estimate rows: {len(df_est)}")
	print(f"GT rows (raw): {len(df_gt)}")
	print(f"Aligned rows: {len(gt_aligned)}")

# python3 4_align_vel_csv.py --est_csv flight_2026-06-03-08-27-15_vel_est_clean.csv --gt_csv flight_2026-06-03-08-27-15_vel_gt_clean.csv

if __name__ == "__main__":
	main()
