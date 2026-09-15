#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_bias_errors(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"time", "err_x", "err_y"}
    missing = required.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {csv_path}: {missing_str}")

    df = df.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["err_x"] = pd.to_numeric(df["err_x"], errors="coerce")
    df["err_y"] = pd.to_numeric(df["err_y"], errors="coerce")
    df = df.dropna(subset=["time", "err_x", "err_y"]).sort_values("time")
    if df.empty:
        raise ValueError(f"No valid rows in {csv_path}")
    return df.reset_index(drop=True)


def build_xy_error(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    t_abs = df["time"].to_numpy(dtype=float)
    t_rel = t_abs - t_abs[0]
    xy_error = np.sqrt(df["err_x"].to_numpy(dtype=float) ** 2 + df["err_y"].to_numpy(dtype=float) ** 2)
    return t_rel, xy_error


def build_axis_error(df: pd.DataFrame, axis: str) -> Tuple[np.ndarray, np.ndarray]:
    if axis not in {"x", "y"}:
        raise ValueError(f"Unsupported axis: {axis}")

    t_abs = df["time"].to_numpy(dtype=float)
    t_rel = t_abs - t_abs[0]
    error = np.abs(df[f"err_{axis}"].to_numpy(dtype=float))
    return t_rel, error


def annotate_series(ax, x: np.ndarray, y: np.ndarray, label: str, color: str) -> None:
    last_x = float(x[-1])
    last_y = float(y[-1])
    y_span = float(np.max(y) - np.min(y))
    y_offset = max(0.01, 0.06 * y_span)

    ax.annotate(
        label,
        xy=(last_x, last_y),
        xytext=(last_x, last_y + y_offset),
        textcoords="data",
        ha="right",
        va="bottom",
        color=color,
        fontsize=16,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "color": color, "lw": 1.2},
    )


def style_ticks(ax) -> None:
    ax.tick_params(axis="both", labelsize=16)


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("time from window start (s)", fontsize=18)
    ax.set_ylabel("horizontal velocity error e_v (m/s)", fontsize=18)
    style_ticks(ax)


def style_axis_component(ax, axis: str) -> None:
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("time from window start (s)", fontsize=16)
    ax.set_ylabel(f"absolute velocity error |err_{axis}| (m/s)", fontsize=16)
    style_ticks(ax)


def plot_overlay(ai_t, ai_ev, raw_t, raw_ev, out_png: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ai_color = "#1f77b4"
    raw_color = "#d62728"

    ax.plot(ai_t, ai_ev, color=ai_color, linewidth=2.6, label="AI")
    ax.plot(raw_t, raw_ev, color=raw_color, linewidth=2.6, label="RAW")

    annotate_series(ax, ai_t, ai_ev, "AI", ai_color)
    annotate_series(ax, raw_t, raw_ev, "RAW", raw_color)

    style_axis(ax)
    ax.legend(loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_side_by_side(ai_t, ai_ev, raw_t, raw_ev, out_png: Path, dpi: int) -> None:
    fig, (ax_ai, ax_raw) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    ai_color = "#1f77b4"
    raw_color = "#d62728"

    ax_ai.plot(ai_t, ai_ev, color=ai_color, linewidth=2.6)
    annotate_series(ax_ai, ai_t, ai_ev, "AI", ai_color)
    style_axis(ax_ai)

    ax_raw.plot(raw_t, raw_ev, color=raw_color, linewidth=2.6)
    annotate_series(ax_raw, raw_t, raw_ev, "RAW", raw_color)
    style_axis(ax_raw)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_component_overlay(axis: str, ai_t, ai_err, raw_t, raw_err, out_png: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ai_color = "#1f77b4"
    raw_color = "#d62728"

    ax.plot(ai_t, ai_err, color=ai_color, linewidth=2.6, label="AI")
    ax.plot(raw_t, raw_err, color=raw_color, linewidth=2.6, label="RAW")

    annotate_series(ax, ai_t, ai_err, "AI", ai_color)
    annotate_series(ax, raw_t, raw_err, "RAW", raw_color)

    style_axis_component(ax, axis)
    ax.legend(loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_component_side_by_side(axis: str, ai_t, ai_err, raw_t, raw_err, out_png: Path, dpi: int) -> None:
    fig, (ax_ai, ax_raw) = plt.subplots(1, 2, figsize=(16, 6.5), sharey=True)

    ai_color = "#1f77b4"
    raw_color = "#d62728"

    ax_ai.plot(ai_t, ai_err, color=ai_color, linewidth=2.6)
    annotate_series(ax_ai, ai_t, ai_err, "AI", ai_color)
    style_axis_component(ax_ai, axis)

    ax_raw.plot(raw_t, raw_err, color=raw_color, linewidth=2.6)
    annotate_series(ax_raw, raw_t, raw_err, "RAW", raw_color)
    style_axis_component(ax_raw, axis)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot AI and RAW horizontal velocity error over time, without drift-rate fit lines.",
    )
    parser.add_argument("--ai_csv", required=True, help="AI bias-error CSV")
    parser.add_argument("--raw_csv", required=True, help="RAW bias-error CSV")
    parser.add_argument(
        "--overlay_png",
        default="xy_error_ai_vs_raw_overlay.png",
        help="Output PNG for the overlay plot",
    )
    parser.add_argument(
        "--side_by_side_png",
        default="xy_error_ai_vs_raw_side_by_side.png",
        help="Output PNG for the side-by-side plot",
    )
    parser.add_argument(
        "--x_overlay_png",
        default="x_error_ai_vs_raw_overlay.png",
        help="Output PNG for the x-component overlay plot",
    )
    parser.add_argument(
        "--x_side_by_side_png",
        default="x_error_ai_vs_raw_side_by_side.png",
        help="Output PNG for the x-component side-by-side plot",
    )
    parser.add_argument(
        "--y_overlay_png",
        default="y_error_ai_vs_raw_overlay.png",
        help="Output PNG for the y-component overlay plot",
    )
    parser.add_argument(
        "--y_side_by_side_png",
        default="y_error_ai_vs_raw_side_by_side.png",
        help="Output PNG for the y-component side-by-side plot",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    args = parser.parse_args()

    ai_csv = Path(args.ai_csv)
    raw_csv = Path(args.raw_csv)
    overlay_png = Path(args.overlay_png)
    side_by_side_png = Path(args.side_by_side_png)

    if not ai_csv.exists():
        raise SystemExit(f"AI CSV not found: {ai_csv}")
    if not raw_csv.exists():
        raise SystemExit(f"RAW CSV not found: {raw_csv}")

    ai_df = load_bias_errors(ai_csv)
    raw_df = load_bias_errors(raw_csv)

    ai_t, ai_ev = build_xy_error(ai_df)
    raw_t, raw_ev = build_xy_error(raw_df)

    plot_overlay(ai_t, ai_ev, raw_t, raw_ev, overlay_png, args.dpi)
    plot_side_by_side(ai_t, ai_ev, raw_t, raw_ev, side_by_side_png, args.dpi)

    ai_tx, ai_ex = build_axis_error(ai_df, "x")
    raw_tx, raw_ex = build_axis_error(raw_df, "x")
    ai_ty, ai_ey = build_axis_error(ai_df, "y")
    raw_ty, raw_ey = build_axis_error(raw_df, "y")

    plot_component_overlay("x", ai_tx, ai_ex, raw_tx, raw_ex, Path(args.x_overlay_png), args.dpi)
    plot_component_side_by_side("x", ai_tx, ai_ex, raw_tx, raw_ex, Path(args.x_side_by_side_png), args.dpi)
    plot_component_overlay("y", ai_ty, ai_ey, raw_ty, raw_ey, Path(args.y_overlay_png), args.dpi)
    plot_component_side_by_side("y", ai_ty, ai_ey, raw_ty, raw_ey, Path(args.y_side_by_side_png), args.dpi)

    print(f"Saved overlay plot: {overlay_png}")
    print(f"Saved side-by-side plot: {side_by_side_png}")
    print(f"Saved x overlay plot: {args.x_overlay_png}")
    print(f"Saved x side-by-side plot: {args.x_side_by_side_png}")
    print(f"Saved y overlay plot: {args.y_overlay_png}")
    print(f"Saved y side-by-side plot: {args.y_side_by_side_png}")


if __name__ == "__main__":
    main()
