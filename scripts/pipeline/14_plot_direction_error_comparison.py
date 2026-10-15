#!/usr/bin/env python3

"""Plot multiple direction-error sample CSVs as vertically stacked subplots."""

import argparse
import importlib.util
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_COLUMNS = (
	"time_from_window_start_s",
	"gt_horizontal_speed_mps",
	"velocity_direction_error_deg",
)


def _load_direction_module():
	spec = importlib.util.spec_from_file_location(
		"velocity_direction_error",
		SCRIPT_DIR / "13_velocity_direction_error.py",
	)
	if spec is None or spec.loader is None:
		raise ImportError("Could not load 13_velocity_direction_error.py")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


DIRECTION = _load_direction_module()


def load_samples(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {path}")
	df = df.copy()
	for column in REQUIRED_COLUMNS:
		df[column] = pd.to_numeric(df[column], errors="coerce")
	df = df.dropna(subset=["time_from_window_start_s"]).sort_values(
		"time_from_window_start_s"
	)
	if df.empty:
		raise ValueError(f"No valid direction-error samples in {path}")
	return df


def plot_direction_error_subplots(
	series: Sequence[Tuple[str, pd.DataFrame]],
	output_path: Path,
	min_gt_speed_mps: float,
	title: str,
	dpi: int,
) -> None:
	if not series:
		raise ValueError("At least one direction-error series is required")

	DIRECTION.configure_plot_style()
	fig, axes = plt.subplots(
		len(series),
		1,
		figsize=(20, 4.0 * len(series)),
		sharex=True,
		sharey=False,
		squeeze=False,
	)
	colors = ("#1f4bb3", "#7b2cbf", "#d62728", "#8c564b")

	for index, ((label, samples), axis) in enumerate(zip(series, axes[:, 0])):
		time = samples["time_from_window_start_s"].to_numpy(dtype=float)
		error = samples["velocity_direction_error_deg"].to_numpy(dtype=float)
		raw_error = samples.get(
			"velocity_direction_error_raw_deg",
			samples["velocity_direction_error_deg"],
		).to_numpy(dtype=float)
		gt_speed = samples["gt_horizontal_speed_mps"].to_numpy(dtype=float)
		low_gt_speed = gt_speed < min_gt_speed_mps
		mean_error = float(np.nanmean(error)) if np.any(np.isfinite(error)) else float("nan")

		DIRECTION._shade_low_speed_regions(axis, time, low_gt_speed, min_gt_speed_mps)
		axis.plot(
			time,
			error,
			color=colors[index % len(colors)],
			linewidth=DIRECTION.DATA_LINE_WIDTH,
			label="Direction error (evaluated)",
		)
		low_speed_error = np.where(low_gt_speed, raw_error, np.nan)
		axis.plot(
			time,
			low_speed_error,
			color=colors[index % len(colors)],
			linewidth=DIRECTION.DATA_LINE_WIDTH,
			alpha=DIRECTION.LOW_SPEED_LINE_ALPHA,
			label=f"Direction error where GT < {min_gt_speed_mps:g} m/s",
		)
		if np.isfinite(mean_error):
			axis.axhline(
				mean_error,
				color="#ff7f0e",
				linestyle="--",
				linewidth=DIRECTION.REFERENCE_LINE_WIDTH,
				label=f"Mean ({mean_error:.2f} deg)",
			)
		axis.set_title(label)
		axis.set_ylabel("Direction error (deg)")
		axis.legend(loc="best")
		DIRECTION._style_axis(axis)

	axes[-1, 0].set_xlabel("Time from evaluation-window start (s)", fontsize=24)
	fig.suptitle(title, y=0.995)
	fig.tight_layout(rect=(0, 0, 1, 0.985))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot direction error over time for multiple flights as stacked subplots.",
	)
	parser.add_argument(
		"--series",
		action="append",
		nargs=2,
		metavar=("LABEL", "SAMPLES_CSV"),
		required=True,
		help="Legend/title label and per-sample direction-error CSV; repeat for each subplot",
	)
	parser.add_argument("--out_plot", required=True)
	parser.add_argument("--min_gt_speed", type=float, default=0.2)
	parser.add_argument("--title", default="Horizontal Velocity Direction Error Comparison")
	parser.add_argument("--dpi", type=int, default=300)
	args = parser.parse_args()
	if args.min_gt_speed < 0.0:
		parser.error("--min_gt_speed must be non-negative")
	if args.dpi < 1:
		parser.error("--dpi must be positive")
	return args


def run() -> None:
	args = parse_arguments()
	series: List[Tuple[str, pd.DataFrame]] = []
	for label, raw_path in args.series:
		path = Path(raw_path)
		if not path.exists():
			raise SystemExit(f"Samples CSV not found: {path}")
		series.append((label, load_samples(path)))

	output_path = Path(args.out_plot)
	plot_direction_error_subplots(
		series,
		output_path,
		args.min_gt_speed,
		args.title,
		args.dpi,
	)
	print(f"Saved direction-error comparison plot to {output_path}")


if __name__ == "__main__":
	run()
