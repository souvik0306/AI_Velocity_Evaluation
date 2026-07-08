#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator


# Edit these arrays to control which velocity columns are plotted.
EST_PLOT_COLS = [
	"vel_x",
	"vel_y",
	"vel_z",
]

GT_PLOT_COLS = [
	"vel_x",
	"vel_y",
	"vel_z",
]

COMPARE_AXES = [
	"x",
	"y",
]


def _load_csv(path: str, required_time_col: str = "time") -> pd.DataFrame:
	df = pd.read_csv(path)
	if required_time_col not in df.columns:
		raise ValueError(f"Missing required column '{required_time_col}' in {path}")
	return df.sort_values(required_time_col)


def _filter_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
	return [col for col in columns if col in df.columns]


def _relative_time(df: pd.DataFrame) -> pd.Series:
	time_values = df["time"]
	if len(time_values) > 0:
		return time_values - time_values.iloc[0]
	return time_values


def _style_velocity_axis() -> None:
	ax = plt.gca()
	ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.grid(True, which="major", alpha=0.8)
	ax.grid(True, which="minor", alpha=0.25)


def _plot_series(
	df: pd.DataFrame,
	columns: List[str],
	label_prefix: str,
	out_dir: Path,
	show: bool,
) -> None:
	available = _filter_columns(df, columns)
	if not available:
		print(f"No matching columns found for: {label_prefix}")
		return

	time_values = _relative_time(df).to_numpy()

	for col in available:
		plt.figure(figsize=(12, 6))
		plt.plot(time_values, df[col].to_numpy(), label=col)
		plt.title(f"{label_prefix}: {col}")
		plt.xlabel("time (from start)")
		plt.ylabel("velocity")
		plt.legend()
		_style_velocity_axis()
		plt.tight_layout()
		output_path = out_dir / f"{label_prefix.lower()}_{col}.png"
		plt.savefig(output_path, dpi=150)
		if not show:
			plt.close()


def _plot_comparisons(
	df_est: pd.DataFrame,
	df_gt: pd.DataFrame,
	axes: List[str],
	out_dir: Path,
	show: bool,
) -> None:
	est_time = _relative_time(df_est).to_numpy()
	gt_time = _relative_time(df_gt).to_numpy()

	for axis in axes:
		col = f"vel_{axis}"
		if col not in df_est.columns or col not in df_gt.columns:
			print(f"Skipping comparison for {col}: missing in estimate or GT CSV")
			continue

		plt.figure(figsize=(12, 6))
		plt.plot(gt_time, df_gt[col].to_numpy(), label=f"gt_{col}")
		plt.plot(est_time, df_est[col].to_numpy(), label=f"est_{col}")
		plt.title(f"GT vs EST: {col}")
		plt.xlabel("time (from start)")
		plt.ylabel("velocity")
		plt.legend()
		_style_velocity_axis()
		plt.tight_layout()
		output_path = out_dir / f"compare_gt_est_{col}.png"
		plt.savefig(output_path, dpi=150)
		if not show:
			plt.close()


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--est_csv", default="vel_est.csv")
	parser.add_argument("--gt_csv", default="vel_gt.csv")
	parser.add_argument("--show", action="store_true", help="Show plots interactively")
	parser.add_argument("--out_dir", default="plots_velocity_9_41", help="Directory to save plots")
	args = parser.parse_args()

	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	out_dir = Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	if est_path.exists():
		df_est = _load_csv(str(est_path))
		_plot_series(df_est, EST_PLOT_COLS, "EST", out_dir, args.show)
	else:
		df_est = None
		print(f"Estimate CSV not found: {est_path}")

	if gt_path.exists():
		df_gt = _load_csv(str(gt_path))
		_plot_series(df_gt, GT_PLOT_COLS, "GT", out_dir, args.show)
	else:
		df_gt = None
		print(f"GT CSV not found: {gt_path}")

	if df_est is not None and df_gt is not None:
		_plot_comparisons(df_est, df_gt, COMPARE_AXES, out_dir, args.show)

	if args.show:
		plt.show()
	else:
		plt.close("all")


if __name__ == "__main__":
	main()

# python3 2_plot_vel_csv_data.py --est_csv flight_6_22_vel_est.csv --gt_csv flight_6_22_vel_gt.csv --show
