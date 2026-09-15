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


def build_drift_curve(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    t_abs = df["time"].to_numpy(dtype=float)
    t_rel = t_abs - t_abs[0]
    e_v = np.sqrt(df["err_x"].to_numpy(dtype=float) ** 2 + df["err_y"].to_numpy(dtype=float) ** 2)

    if len(t_rel) >= 2:
        drift_rate, drift_intercept = np.polyfit(t_rel, e_v, 1)
        fit = drift_rate * t_rel + drift_intercept
    else:
        drift_rate, drift_intercept = float("nan"), float("nan")
        fit = np.full_like(t_rel, np.nan)

    return t_rel, e_v, fit, float(drift_rate), float(drift_intercept)


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
    ax.set_xlabel("time from window start (s)", fontsize=18)
    ax.set_ylabel("horizontal velocity error e_v (m/s)", fontsize=18)
    ax.grid(True, alpha=0.3)
    style_ticks(ax)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay AI and RAW drift behavior using horizontal velocity error magnitude "
            "e_v(t)=sqrt(err_x^2+err_y^2), with linear drift fit e_v(t)~=a*t+b."
        ),
    )
    parser.add_argument("--ai_csv", required=True, help="AI bias-error CSV")
    parser.add_argument("--raw_csv", required=True, help="RAW bias-error CSV")
    parser.add_argument(
        "--out_png",
        default="overlay_drift_rate_ai_vs_raw.png",
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    args = parser.parse_args()

    ai_csv = Path(args.ai_csv)
    raw_csv = Path(args.raw_csv)
    out_png = Path(args.out_png)

    if not ai_csv.exists():
        raise SystemExit(f"AI CSV not found: {ai_csv}")
    if not raw_csv.exists():
        raise SystemExit(f"RAW CSV not found: {raw_csv}")

    ai_df = load_bias_errors(ai_csv)
    raw_df = load_bias_errors(raw_csv)

    ai_t, ai_ev, ai_fit, ai_rate, ai_b = build_drift_curve(ai_df)
    raw_t, raw_ev, raw_fit, raw_rate, raw_b = build_drift_curve(raw_df)

    plt.figure(figsize=(14, 7.5))
    plt.plot(ai_t, ai_ev, color="#1f77b4", alpha=0.35, linewidth=1.8, label="AI e_v(t)")
    plt.plot(raw_t, raw_ev, color="#d62728", alpha=0.35, linewidth=1.8, label="RAW e_v(t)")

    plt.plot(
        ai_t,
        ai_fit,
        color="#1f77b4",
        linewidth=2.8,
        label=f"AI fit: rate={ai_rate:.6e} m/s^2",
    )
    plt.plot(
        raw_t,
        raw_fit,
        color="#d62728",
        linewidth=2.8,
        label=f"RAW fit: rate={raw_rate:.6e} m/s^2",
    )

    annotate_series(plt.gca(), ai_t, ai_fit, "AI", "#1f77b4")
    annotate_series(plt.gca(), raw_t, raw_fit, "RAW", "#d62728")

    style_axis(plt.gca())
    plt.legend(loc="best")
    plt.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print(f"AI drift rate: {ai_rate:.6e} m/s^2")
    print(f"AI drift intercept: {ai_b:.6e} m/s")
    print(f"RAW drift rate: {raw_rate:.6e} m/s^2")
    print(f"RAW drift intercept: {raw_b:.6e} m/s")
    print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()
