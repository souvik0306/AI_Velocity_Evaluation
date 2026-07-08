#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnRule:
	name: str
	pattern: str
	# bounds are (min, max). Use None for unbounded.
	bounds: Optional[Tuple[Optional[float], Optional[float]]]
	median_tol: Optional[float]


PROFILE_RULES: Dict[str, List[ColumnRule]] = {
	"velocity": [
		ColumnRule("vel", "vel_*", (-3.0, 3.0), 0.4),
	],
}


def _infer_profile(columns: Iterable[str]) -> str:
	cols = set(columns)
	if any(col.startswith("vel_") for col in cols):
		return "velocity"
	return "unknown"


def _parse_bound_overrides(values: List[str]) -> List[Tuple[str, Optional[float], Optional[float]]]:
	overrides: List[Tuple[str, Optional[float], Optional[float]]] = []
	for item in values:
		parts = item.split(":")
		if len(parts) != 3:
			raise ValueError(f"Invalid --bound entry '{item}'. Use pattern:min:max")
		pattern, raw_min, raw_max = parts
		min_val = None if raw_min.lower() in {"", "none"} else float(raw_min)
		max_val = None if raw_max.lower() in {"", "none"} else float(raw_max)
		overrides.append((pattern, min_val, max_val))
	return overrides


def _match_override(
	column: str,
	overrides: List[Tuple[str, Optional[float], Optional[float]]],
) -> Optional[Tuple[Optional[float], Optional[float]]]:
	for pattern, min_val, max_val in overrides:
		if fnmatch.fnmatch(column, pattern):
			return min_val, max_val
	return None


def _columns_for_rules(df: pd.DataFrame, rules: List[ColumnRule]) -> Dict[str, ColumnRule]:
	columns = [col for col in df.columns if col != "time"]
	selected: Dict[str, ColumnRule] = {}
	for col in columns:
		for rule in rules:
			if fnmatch.fnmatch(col, rule.pattern):
				selected[col] = rule
				break
	return selected


def _apply_physical_pruning(
	df: pd.DataFrame,
	columns: Iterable[str],
	bound_overrides: List[Tuple[str, Optional[float], Optional[float]]],
	rules_by_column: Dict[str, ColumnRule],
) -> Dict[str, int]:
	pruned_counts: Dict[str, int] = {}
	for col in columns:
		rule = rules_by_column.get(col)
		bounds = _match_override(col, bound_overrides)
		if bounds is None and rule is not None:
			bounds = rule.bounds
		if bounds is None:
			continue
		min_val, max_val = bounds
		mask = pd.Series(False, index=df.index)
		if min_val is not None:
			mask |= df[col] < min_val
		if max_val is not None:
			mask |= df[col] > max_val
		count = int(mask.sum())
		if count:
			df.loc[mask, col] = np.nan
		pruned_counts[col] = count
	return pruned_counts


def _apply_median_filter(
	df: pd.DataFrame,
	columns: Iterable[str],
	window: int,
	median_tol_override: Optional[float],
	rules_by_column: Dict[str, ColumnRule],
) -> Dict[str, int]:
	filtered_counts: Dict[str, int] = {}
	for col in columns:
		rule = rules_by_column.get(col)
		if rule is None or rule.median_tol is None:
			continue
		tol = median_tol_override if median_tol_override is not None else rule.median_tol
		median = df[col].rolling(window=window, center=True, min_periods=1).median()
		diff = (df[col] - median).abs()
		mask = diff > tol
		count = int(mask.sum())
		if count:
			df.loc[mask, col] = np.nan
		filtered_counts[col] = count
	return filtered_counts


def _interpolate_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
	for col in columns:
		if df[col].isna().any():
			df[col] = df[col].interpolate(method="linear", limit_direction="both")


def clean_dataframe(
	df: pd.DataFrame,
	window: int,
	median_tol_override: Optional[float],
	bound_overrides: List[Tuple[str, Optional[float], Optional[float]]],
	fill: str,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
	if "time" in df.columns:
		df = df.sort_values("time")

	profile = _infer_profile(df.columns)
	rules = PROFILE_RULES.get(profile, PROFILE_RULES["velocity"])
	rules_by_column = _columns_for_rules(df, rules)
	columns = list(rules_by_column.keys())
	if not columns:
		return df, {}, {}

	pruned_counts = _apply_physical_pruning(df, columns, bound_overrides, rules_by_column)
	filtered_counts = _apply_median_filter(
		df,
		columns,
		window,
		median_tol_override,
		rules_by_column,
	)

	if fill == "linear":
		_interpolate_columns(df, columns)

	return df, pruned_counts, filtered_counts


def _derive_output_path(input_path: Path, out_dir: Optional[Path]) -> Path:
	if out_dir is None:
		return input_path.with_name(f"{input_path.stem}_clean{input_path.suffix}")
	return out_dir / f"{input_path.stem}_clean{input_path.suffix}"


def _normalize_window(window: int) -> int:
	if window < 1:
		raise ValueError("--window must be >= 1")
	if window % 2 == 0:
		window += 1
		print(f"Window size must be odd. Using {window} instead.")
	return window


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Clean CSV datasets with physical pruning and sliding median filtering.",
	)
	parser.add_argument("--input", nargs="*", default=[], help="CSV files to clean")
	parser.add_argument("--est_csv", help="Estimate velocity CSV file to clean")
	parser.add_argument("--gt_csv", help="GT CSV file to clean")
	parser.add_argument("--out_dir", help="Directory for cleaned outputs")
	parser.add_argument("--window", type=int, default=9, help="Sliding median window size")
	parser.add_argument(
		"--median_tol",
		type=float,
		default=None,
		help="Override median tolerance for all filtered columns",
	)
	parser.add_argument(
		"--bound",
		action="append",
		default=[],
		help="Override bounds pattern:min:max (use 'none' for unbounded)",
	)
	parser.add_argument(
		"--fill",
		choices=["none", "linear"],
		default="linear",
		help="Fill NaNs after filtering (default: linear to avoid NaNs)",
	)
	args = parser.parse_args()

	input_files: List[str] = []
	if args.input:
		input_files.extend(args.input)
	if args.est_csv:
		input_files.append(args.est_csv)
	if args.gt_csv:
		input_files.append(args.gt_csv)

	if not input_files:
		raise SystemExit("No input files provided. Use --input, --imu_csv, or --gt_csv.")

	window = _normalize_window(args.window)
	bound_overrides = _parse_bound_overrides(args.bound)
	out_dir = Path(args.out_dir) if args.out_dir else None
	if out_dir is not None:
		out_dir.mkdir(parents=True, exist_ok=True)

	for input_path_str in input_files:
		input_path = Path(input_path_str)
		if not input_path.exists():
			print(f"Skipping missing file: {input_path}")
			continue

		df = pd.read_csv(input_path)
		cleaned, pruned_counts, filtered_counts = clean_dataframe(
			df,
			window,
			args.median_tol,
			bound_overrides,
			args.fill,
		)

		output_path = _derive_output_path(input_path, out_dir)
		cleaned.to_csv(output_path, index=False)

		print(f"Saved cleaned CSV to {output_path}")
		if pruned_counts:
			print("Physical pruning counts:")
			for col, count in pruned_counts.items():
				print(f"  {col}: {count}")
		if filtered_counts:
			print("Median filter counts:")
			for col, count in filtered_counts.items():
				print(f"  {col}: {count}")


if __name__ == "__main__":
	main()

# python3 3_vel_csv_dataset_cleaner.py --est_csv flight_6_27_vel_est.csv --gt_csv flight_6_27_vel_gt.csv 