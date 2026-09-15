#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


REQUIRED_COLS = ["time", "vel_x", "vel_y", "vel_z"]


def _load_velocity_csv(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [col for col in REQUIRED_COLS if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {path}")

	df = df.copy()
	df["time"] = pd.to_numeric(df["time"], errors="coerce")
	df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
	return df


def _rmse(values: pd.Series) -> float:
	vals = values.to_numpy(dtype=float)
	vals = vals[np.isfinite(vals)]
	if vals.size == 0:
		return float("nan")
	return float(np.sqrt(np.mean(vals ** 2)))


def _rmse_xy(err_x: pd.Series, err_y: pd.Series) -> float:
	vals_x = err_x.to_numpy(dtype=float)
	vals_y = err_y.to_numpy(dtype=float)
	mask = np.isfinite(vals_x) & np.isfinite(vals_y)
	if not mask.any():
		return float("nan")
	return float(np.sqrt(np.mean(vals_x[mask] ** 2 + vals_y[mask] ** 2)))


def _compute_bias_errors(merged: pd.DataFrame) -> pd.DataFrame:
	for axis in ("x", "y", "z"):
		est_col = f"vel_{axis}_est"
		gt_col = f"vel_{axis}_gt"
		if est_col not in merged.columns or gt_col not in merged.columns:
			raise ValueError(f"Missing columns after merge: {est_col} or {gt_col}")

	merged = merged.copy()
	merged["err_x"] = merged["vel_x_est"] - merged["vel_x_gt"]
	merged["err_y"] = merged["vel_y_est"] - merged["vel_y_gt"]
	merged["err_z"] = merged["vel_z_est"] - merged["vel_z_gt"]

	bias_x = float(merged.at[0, "err_x"])
	bias_y = float(merged.at[0, "err_y"])
	bias_z = float(merged.at[0, "err_z"])

	merged["err_x_bias"] = merged["err_x"] - bias_x
	merged["err_y_bias"] = merged["err_y"] - bias_y
	merged["err_z_bias"] = merged["err_z"] - bias_z
	return merged


def _format_rmse(metrics: Dict[str, float]) -> str:
	return ", ".join(f"{key}={value:.6f}" for key, value in metrics.items())


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Bias-normalize velocity errors per axis and report RMSE.",
	)
	parser.add_argument("--est_csv", required=True, help="Windowed estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="Windowed aligned GT velocity CSV")
	args = parser.parse_args()

	est_path = Path(args.est_csv)
	gt_path = Path(args.gt_csv)
	if not est_path.exists():
		raise SystemExit(f"Estimate CSV not found: {est_path}")
	if not gt_path.exists():
		raise SystemExit(f"GT CSV not found: {gt_path}")

	output_path = Path.cwd() / f"{est_path.stem}_bias_errors{est_path.suffix}"
	output_txt = Path.cwd() / f"{est_path.stem}_rmse_summary.txt"

	df_est = _load_velocity_csv(est_path)
	df_gt = _load_velocity_csv(gt_path)

	merged = pd.merge(
		df_est,
		df_gt,
		on="time",
		how="inner",
		suffixes=("_est", "_gt"),
	)
	if merged.empty:
		raise ValueError("No overlapping timestamps between estimate and GT CSVs")

	merged = _compute_bias_errors(merged)

	rmse_x = _rmse(merged["err_x"])
	rmse_y = _rmse(merged["err_y"])
	rmse_z = _rmse(merged["err_z"])
	rmse_xy = _rmse_xy(merged["err_x"], merged["err_y"])

	rmse_x_bias = _rmse(merged["err_x_bias"])
	rmse_y_bias = _rmse(merged["err_y_bias"])
	rmse_z_bias = _rmse(merged["err_z_bias"])
	rmse_xy_bias = _rmse_xy(merged["err_x_bias"], merged["err_y_bias"])

	output_cols = [
		"time",
		"err_x",
		"err_y",
		"err_z",
		"err_x_bias",
		"err_y_bias",
		"err_z_bias",
	]
	merged[output_cols].to_csv(output_path, index=False)

	print(f"Saved bias-normalized errors to {output_path} (rows: {len(merged)})")
	print(
		"RMSE (Not including bias): "
		f"RMSE_x={rmse_x:.6f}, "
		f"RMSE_y={rmse_y:.6f}, "
		f"RMSE_z={rmse_z:.6f}, "
		f"RMSE_xy={rmse_xy:.6f}"
	)
	print(
		"RMSE (bias corrected): "
		f"RMSE_x={rmse_x_bias:.6f}, "
		f"RMSE_y={rmse_y_bias:.6f}, "
		f"RMSE_z={rmse_z_bias:.6f}, "
		f"RMSE_xy={rmse_xy_bias:.6f}"
	)

	lines = [
     	"RMSE (Not including bias): ",
        "",
		f"RMSE_x={rmse_x:.6f}",
		f"RMSE_y={rmse_y:.6f}",
		f"RMSE_z={rmse_z:.6f}",
		f"RMSE_xy={rmse_xy:.6f}",
        "",
        "RMSE (bias corrected):", 
        "",
		f"RMSE_x={rmse_x_bias:.6f}",
		f"RMSE_y={rmse_y_bias:.6f}",
		f"RMSE_z={rmse_z_bias:.6f}",
		f"RMSE_xy={rmse_xy_bias:.6f}",
	]
	output_txt.write_text("\n".join(lines) + "\n")
	print(f"Saved RMSE summary to {output_txt}")

if __name__ == "__main__":
	main()

# python3 6_bias_zeroing_and_rmse.py --est_csv flight_6_27_vel_est_clean_window.csv --gt_csv flight_6_27_vel_gt_clean_aligned_window.csv
