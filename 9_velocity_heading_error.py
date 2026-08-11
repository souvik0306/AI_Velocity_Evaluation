#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLS = ["time", "vel_x", "vel_y"]
PROFILE_SPECS = {
	"low": {
		"label": "Low speed",
		"min_speed": 0.2,
		"max_speed": 0.5,
		"include_min": True,
		"heading_limit_deg": 60.0,
		"mag_limit_pct": 100.0,
	},
	"medium": {
		"label": "Medium speed",
		"min_speed": 0.5,
		"max_speed": 2.0,
		"include_min": True,
		"heading_limit_deg": 30.0,
		"mag_limit_pct": 50.0,
	},
	"high": {
		"label": "High speed",
		"min_speed": 2.0,
		"max_speed": 5.0,
		"include_min": True,
		"heading_limit_deg": 5.0,
		"mag_limit_pct": 30.0,
	},
}


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


def _profile_mask(gt_speed_xy: np.ndarray, spec: Dict[str, float]) -> np.ndarray:
	if spec["include_min"]:
		min_mask = gt_speed_xy >= spec["min_speed"]
	else:
		min_mask = gt_speed_xy > spec["min_speed"]
	return min_mask & (gt_speed_xy < spec["max_speed"])


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
	result["speed_profile"] = "out_of_range"
	result["heading_pass"] = False
	result["magnitude_pass"] = False
	result["sample_pass"] = False

	for profile, spec in PROFILE_SPECS.items():
		mask = _profile_mask(gt_speed_xy, spec)
		result.loc[mask, "speed_profile"] = profile
		result.loc[mask, "heading_pass"] = (
			result.loc[mask, "heading_error_deg"] < spec["heading_limit_deg"]
		).fillna(False)
		result.loc[mask, "magnitude_pass"] = (
			result.loc[mask, "magnitude_error_pct"] < spec["mag_limit_pct"]
		).fillna(False)
		result.loc[mask, "sample_pass"] = (
			result.loc[mask, "heading_pass"] & result.loc[mask, "magnitude_pass"]
		)

	return result


def _summarize_profiles(samples: pd.DataFrame) -> pd.DataFrame:
	rows = []
	for profile, spec in PROFILE_SPECS.items():
		profile_df = samples[samples["speed_profile"] == profile]
		total = int(len(profile_df))
		passing = int(profile_df["sample_pass"].sum()) if total else 0
		pass_pct = passing / total * 100.0 if total else float("nan")
		rows.append(
			{
				"profile": profile,
				"profile_label": spec["label"],
				"gt_speed_range_mps": _format_speed_range(spec),
				"heading_limit_deg": spec["heading_limit_deg"],
				"magnitude_limit_pct": spec["mag_limit_pct"],
				"samples": total,
				"passing_samples": passing,
				"pass_pct": pass_pct,
				"avg_heading_error_deg": profile_df["heading_error_deg"].mean(),
				"avg_magnitude_error_pct": profile_df["magnitude_error_pct"].mean(),
			}
		)
	return pd.DataFrame(rows)


def _format_speed_range(spec: Dict[str, float]) -> str:
	left = "<=" if spec["include_min"] else "<"
	return f"{spec['min_speed']:g} {left} v_gt_xy < {spec['max_speed']:g}"


def _print_summary(summary: pd.DataFrame, samples: pd.DataFrame) -> None:
	print("Velocity Heading Error Summary")
	print("Formula uses horizontal x/y velocity components.")
	print("")
	for _, row in summary.iterrows():
		pass_pct = row["pass_pct"]
		avg_heading = row["avg_heading_error_deg"]
		avg_mag = row["avg_magnitude_error_pct"]
		print(f"{row['profile_label']} ({row['gt_speed_range_mps']} m/s)")
		print(
			"  Requirement: "
			f"heading < {row['heading_limit_deg']:.1f} deg, "
			f"magnitude < {row['magnitude_limit_pct']:.1f}%"
		)
		print(
			"  Pass: "
			f"{row['passing_samples']}/{row['samples']} "
			f"({pass_pct:.2f}%)"
			if np.isfinite(pass_pct)
			else "  Pass: 0/0 (nan%)"
		)
		print(
			"  Averages: "
			f"heading={avg_heading:.3f} deg, "
			f"magnitude={avg_mag:.3f}%"
			if np.isfinite(avg_heading) and np.isfinite(avg_mag)
			else "  Averages: heading=nan deg, magnitude=nan%"
		)
		print("")

	out_of_range = int((samples["speed_profile"] == "out_of_range").sum())
	if out_of_range:
		print(
			"Samples outside evaluated GT speed ranges "
			f"(v_gt_xy < 0.2 or v_gt_xy >= 5 m/s): {out_of_range}"
		)


def _derive_output_paths(est_path: Path) -> Tuple[Path, Path]:
	return (
		Path.cwd() / f"{est_path.stem}_vhe_samples{est_path.suffix}",
		Path.cwd() / f"{est_path.stem}_vhe_summary{est_path.suffix}",
	)


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Calculate horizontal velocity heading error and magnitude error, "
			"then evaluate samples by Vicon/GT horizontal speed profile."
		),
	)
	parser.add_argument("--est_csv", required=True, help="Windowed estimate velocity CSV")
	parser.add_argument("--gt_csv", required=True, help="Windowed aligned GT velocity CSV")
	parser.add_argument(
		"--out_samples_csv",
		help="Optional output CSV with per-sample heading/magnitude errors and pass flags",
	)
	parser.add_argument(
		"--out_summary_csv",
		help="Optional output CSV with per-profile pass percentages and averages",
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
	summary = _summarize_profiles(samples)

	samples_path.parent.mkdir(parents=True, exist_ok=True)
	summary_path.parent.mkdir(parents=True, exist_ok=True)
	samples.to_csv(samples_path, index=False)
	summary.to_csv(summary_path, index=False)

	_print_summary(summary, samples)
	print(f"Saved per-sample VHE results to {samples_path}")
	print(f"Saved profile VHE summary to {summary_path}")


if __name__ == "__main__":
	main()
