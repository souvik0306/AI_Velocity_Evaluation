#!/usr/bin/env python3

"""Plot horizontal velocity error for multiple flights as stacked subplots."""

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


def _load_pipeline_module(filename: str, module_name: str):
	spec = importlib.util.spec_from_file_location(module_name, SCRIPT_DIR / filename)
	if spec is None or spec.loader is None:
		raise ImportError(f"Could not load {filename}")
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


HORIZONTAL_ERROR = _load_pipeline_module("8_xy_error_ai_vs_raw.py", "xy_error")
PLOT_STYLE = _load_pipeline_module("13_velocity_direction_error.py", "direction_plot_style")


def plot_horizontal_error_subplots(
	series: Sequence[Tuple[str, pd.DataFrame]],
	output_path: Path,
	title: str,
	dpi: int,
) -> None:
	if not series:
		raise ValueError("At least one horizontal-error series is required")

	PLOT_STYLE.configure_plot_style()
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
		time, horizontal_error = HORIZONTAL_ERROR.build_xy_error(samples)
		horizontal_rmse = float(np.sqrt(np.mean(horizontal_error ** 2)))
		color = colors[index % len(colors)]

		axis.plot(
			time,
			horizontal_error,
			color=color,
			linewidth=PLOT_STYLE.DATA_LINE_WIDTH,
			label="Horizontal velocity error",
		)
		axis.axhline(
			horizontal_rmse,
			color="#ff7f0e",
			linestyle="--",
			linewidth=PLOT_STYLE.REFERENCE_LINE_WIDTH,
			label=f"Horizontal RMSE ({horizontal_rmse:.3f} m/s)",
		)
		axis.set_title(label)
		axis.set_ylim(bottom=0.0)
		axis.legend(loc="best")
		PLOT_STYLE._style_axis(axis)

	axes[-1, 0].set_xlabel("Time from evaluation-window start (s)", fontsize=24)
	fig.text(
		0.012,
		0.5,
		"Horizontal velocity error (m/s)",
		rotation="vertical",
		va="center",
		ha="center",
		fontsize=24,
	)
	fig.suptitle(title, y=0.995)
	fig.tight_layout(rect=(0.025, 0, 1, 0.985))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
	plt.close(fig)


def parse_arguments() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot horizontal velocity error over time for multiple flights.",
	)
	parser.add_argument(
		"--series",
		action="append",
		nargs=2,
		metavar=("LABEL", "BIAS_ERRORS_CSV"),
		required=True,
		help="Subplot label and bias-error CSV; repeat for each flight",
	)
	parser.add_argument("--out_plot", required=True)
	parser.add_argument(
		"--title",
		default="Horizontal Velocity Error Comparison",
	)
	parser.add_argument("--dpi", type=int, default=300)
	args = parser.parse_args()
	if args.dpi < 1:
		parser.error("--dpi must be positive")
	return args


def run() -> None:
	args = parse_arguments()
	series: List[Tuple[str, pd.DataFrame]] = []
	for label, raw_path in args.series:
		path = Path(raw_path)
		if not path.exists():
			raise SystemExit(f"Bias-error CSV not found: {path}")
		series.append((label, HORIZONTAL_ERROR.load_bias_errors(path)))

	output_path = Path(args.out_plot)
	plot_horizontal_error_subplots(series, output_path, args.title, args.dpi)
	print(f"Saved horizontal velocity-error comparison plot to {output_path}")


if __name__ == "__main__":
	run()
