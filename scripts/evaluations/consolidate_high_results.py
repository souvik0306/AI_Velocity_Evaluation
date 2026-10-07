#!/usr/bin/env python3
"""Consolidate per-flight ``analyze_high`` metrics into AI and RAW CSVs."""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "high"
WINDOW_DURATIONS_S = (5, 10, 20, 30, 40)
GROUPS = ("AI", "RAW")
METRIC_COLUMNS = (
    "bias_corrected_rmse_x_mps",
    "bias_corrected_rmse_y_mps",
    "bias_corrected_rmse_z_mps",
    "bias_corrected_rmse_xy_mps",
    "drift_rate_mps2",
)
OUTPUT_COLUMNS = ("window_duration_s", "flight", *METRIC_COLUMNS)


def flight_sort_key(row: Dict[str, str]) -> tuple:
    """Sort names such as flight_2 before flight_10."""
    flight = row["flight"]
    match = re.search(r"(\d+)$", flight)
    return (int(match.group(1)) if match else float("inf"), flight)


def load_duration_rows(results_dir: Path, duration_s: int) -> List[Dict[str, str]]:
    summary_path = (
        results_dir
        / f"{duration_s}s"
        / f"high_{duration_s}s_summary_by_flight.csv"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing analyze_high summary: {summary_path}")

    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        reader = csv.DictReader(summary_file)
        required = {"group", "flight", *METRIC_COLUMNS}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise ValueError(
                f"{summary_path} is missing required columns: {', '.join(missing)}"
            )
        return list(reader)


def consolidate(results_dir: Path, output_dir: Path) -> Dict[str, Path]:
    """Write one compact, duration-ordered CSV for each AI/RAW group."""
    grouped_rows: Dict[str, List[Dict[str, str]]] = {group: [] for group in GROUPS}

    for duration_s in WINDOW_DURATIONS_S:
        duration_rows = load_duration_rows(results_dir, duration_s)
        for group in GROUPS:
            rows = sorted(
                (row for row in duration_rows if row["group"].strip().upper() == group),
                key=flight_sort_key,
            )
            if not rows:
                raise ValueError(
                    f"No {group} rows found in the {duration_s}s analyze_high summary"
                )
            for row in rows:
                grouped_rows[group].append(
                    {
                        "window_duration_s": str(duration_s),
                        "flight": row["flight"],
                        **{column: row[column] for column in METRIC_COLUMNS},
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for group, rows in grouped_rows.items():
        output_path = output_dir / f"high_{group.lower()}_consolidated.csv"
        with output_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        output_paths[group] = output_path

    return output_paths


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate AI and RAW CSVs from analyze_high summaries, ordered "
            "by 5s, 10s, 20s, 30s, then 40s windows."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"analyze_high results directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for consolidated CSVs (default: the results directory)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir or args.results_dir
    output_paths = consolidate(args.results_dir, output_dir)
    for group in GROUPS:
        print(f"Wrote {group}: {output_paths[group]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
