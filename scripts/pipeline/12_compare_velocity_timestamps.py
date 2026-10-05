#!/usr/bin/env python3
"""Compare estimate and ground-truth velocity timestamps in a ROS 1 bag.

Each estimate message is paired with the nearest GT message twice: once using
``header.stamp`` and once using the timestamp stored by rosbag.  The resulting
signed differences are therefore::

    estimate timestamp - nearest GT timestamp

Positive values mean that the estimate timestamp is later than its matched GT
timestamp.  This measures timestamp-grid alignment; it does not estimate a
sensor latency when the two topics do not describe the same physical sample.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rosbag
from scipy.stats import wilcoxon


DEFAULT_EST_TOPIC = "/mavros/local_position/velocity_local"
DEFAULT_GT_TOPIC = "/vrpn_client_node/AIIMU1/twist"


def header_time_to_sec(msg) -> float:
    if not hasattr(msg, "header") or not hasattr(msg.header, "stamp"):
        message_type = getattr(msg, "_type", type(msg).__name__)
        raise AttributeError(
            f"message type {message_type} has no header.stamp timestamp"
        )
    return float(msg.header.stamp.to_sec())


def read_topic_timestamps(
    bag_path: Path, est_topic: str, gt_topic: str
) -> Dict[str, Dict[str, np.ndarray]]:
    records = {
        "estimate": {"header": [], "bag": []},
        "gt": {"header": [], "bag": []},
    }
    topic_to_name = {est_topic: "estimate", gt_topic: "gt"}

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, bag_stamp in bag.read_messages(topics=list(topic_to_name)):
            name = topic_to_name[topic]
            try:
                header_stamp = header_time_to_sec(msg)
            except AttributeError as exc:
                raise ValueError(f"{topic}: {exc}") from exc
            records[name]["header"].append(header_stamp)
            records[name]["bag"].append(float(bag_stamp.to_sec()))

    output: Dict[str, Dict[str, np.ndarray]] = {}
    for name, clocks in records.items():
        if not clocks["header"]:
            topic = est_topic if name == "estimate" else gt_topic
            raise ValueError(f"No messages found for topic {topic!r}")
        order = np.argsort(clocks["bag"])
        output[name] = {
            clock: np.asarray(values, dtype=float)[order]
            for clock, values in clocks.items()
        }
    return output


def nearest_indices(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Return the index of the closest candidate for every reference value."""
    candidate_order = np.argsort(candidates)
    sorted_candidates = candidates[candidate_order]
    right = np.searchsorted(sorted_candidates, reference, side="left")
    right = np.clip(right, 0, len(sorted_candidates) - 1)
    left = np.clip(right - 1, 0, len(sorted_candidates) - 1)
    choose_left = np.abs(reference - sorted_candidates[left]) <= np.abs(
        sorted_candidates[right] - reference
    )
    sorted_index = np.where(choose_left, left, right)
    return candidate_order[sorted_index]


def calculate_differences(
    estimate: np.ndarray,
    gt: np.ndarray,
    max_match_s: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = nearest_indices(estimate, gt)
    differences = estimate - gt[indices]
    valid = np.isfinite(differences)
    if max_match_s is not None:
        valid &= np.abs(differences) <= max_match_s
    return differences, indices, valid


def spaced_values(
    values: np.ndarray, sample_times: np.ndarray, spacing_s: float
) -> np.ndarray:
    """Thin correlated data before a significance test."""
    finite = np.isfinite(values) & np.isfinite(sample_times)
    values = values[finite]
    sample_times = sample_times[finite]
    if not len(values) or spacing_s <= 0:
        return values

    order = np.argsort(sample_times)
    selected = []
    last_time = -np.inf
    for index in order:
        if sample_times[index] - last_time >= spacing_s:
            selected.append(index)
            last_time = sample_times[index]
    return values[np.asarray(selected, dtype=int)]


def wilcoxon_zero_pvalue(values: np.ndarray) -> float:
    if len(values) < 2 or np.allclose(values, 0.0):
        return 1.0
    try:
        return float(wilcoxon(values, alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def summarize(
    name: str,
    differences_s: np.ndarray,
    sample_times_s: np.ndarray,
    threshold_s: float,
    alpha: float,
    test_spacing_s: float,
) -> dict:
    values_ms = differences_s * 1e3
    threshold_ms = threshold_s * 1e3
    test_values = spaced_values(differences_s, sample_times_s, test_spacing_s)
    p_value = wilcoxon_zero_pvalue(test_values)
    median_ms = float(np.median(values_ms))
    median_abs_ms = float(np.median(np.abs(values_ms)))
    p95_abs_ms = float(np.percentile(np.abs(values_ms), 95))
    exceedance = float(np.mean(np.abs(values_ms) > threshold_ms) * 100.0)
    statistically_different = bool(np.isfinite(p_value) and p_value < alpha)
    practically_different = median_abs_ms > threshold_ms

    print(f"\n{name}")
    print(f"  matched samples: {len(values_ms)}")
    print(f"  median signed estimate - GT: {median_ms:+.3f} ms")
    print(f"  median absolute difference:  {median_abs_ms:.3f} ms")
    print(f"  95th percentile absolute:   {p95_abs_ms:.3f} ms")
    print(f"  above {threshold_ms:g} ms:               {exceedance:.1f}%")
    print(
        f"  Wilcoxon vs zero (n={len(test_values)}, alpha={alpha:g}): "
        f"p={p_value:.3g} -> "
        f"{'SIGNIFICANT' if statistically_different else 'not significant'}"
    )
    print(
        f"  Practical threshold (median absolute > {threshold_ms:g} ms): "
        f"{'SIGNIFICANT' if practically_different else 'not significant'}"
    )
    return {
        "median_ms": median_ms,
        "median_abs_ms": median_abs_ms,
        "p_value": p_value,
        "statistically_different": statistically_different,
        "practically_different": practically_different,
    }


def create_plot(
    output_path: Path,
    elapsed_s: np.ndarray,
    header_diff_s: np.ndarray,
    header_valid: np.ndarray,
    bag_diff_s: np.ndarray,
    bag_valid: np.ndarray,
    threshold_s: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    panels = (
        (axes[0], header_diff_s, header_valid, "Header timestamp"),
        (axes[1], bag_diff_s, bag_valid, "Rosbag record timestamp"),
    )
    threshold_ms = threshold_s * 1e3
    for axis, differences, valid, title in panels:
        axis.plot(
            elapsed_s[valid], differences[valid] * 1e3, ".", markersize=2.5, alpha=0.65
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(threshold_ms, color="tab:red", linestyle="--", linewidth=0.9)
        axis.axhline(-threshold_ms, color="tab:red", linestyle="--", linewidth=0.9)
        axis.set_title(f"{title}: estimate - nearest GT")
        axis.set_ylabel("Time difference (ms)")
        axis.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Elapsed rosbag time (s)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path, help="ROS 1 bag to inspect")
    parser.add_argument("--velocity_topic", default=DEFAULT_EST_TOPIC)
    parser.add_argument("--gt_velocity_topic", default=DEFAULT_GT_TOPIC)
    parser.add_argument(
        "--out",
        type=Path,
        help="Plot path (default: <bag>_velocity_timestamp_difference.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path for all matched timestamps and differences",
    )
    parser.add_argument(
        "--threshold_ms",
        type=float,
        default=5.0,
        help="Practical significance threshold in milliseconds (default: 5)",
    )
    parser.add_argument(
        "--max_match_ms",
        type=float,
        help="Discard nearest matches farther apart than this many milliseconds",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Statistical significance level"
    )
    parser.add_argument(
        "--test_spacing_s",
        type=float,
        default=0.1,
        help="Minimum spacing of samples used by Wilcoxon test (default: 0.1)",
    )
    args = parser.parse_args()
    if args.threshold_ms < 0:
        parser.error("--threshold_ms must be non-negative")
    if args.max_match_ms is not None and args.max_match_ms <= 0:
        parser.error("--max_match_ms must be positive")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.test_spacing_s < 0:
        parser.error("--test_spacing_s must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.bag.is_file():
        raise SystemExit(f"Bag not found: {args.bag}")

    timestamps = read_topic_timestamps(
        args.bag, args.velocity_topic, args.gt_velocity_topic
    )
    estimate = timestamps["estimate"]
    gt = timestamps["gt"]
    max_match_s = None if args.max_match_ms is None else args.max_match_ms / 1e3
    threshold_s = args.threshold_ms / 1e3

    header_diff, header_gt_index, header_valid = calculate_differences(
        estimate["header"], gt["header"], max_match_s
    )
    bag_diff, bag_gt_index, bag_valid = calculate_differences(
        estimate["bag"], gt["bag"], max_match_s
    )
    elapsed_s = estimate["bag"] - estimate["bag"][0]

    print(f"Bag: {args.bag}")
    print(f"Estimate topic: {args.velocity_topic} ({len(estimate['bag'])} messages)")
    print(f"GT topic:       {args.gt_velocity_topic} ({len(gt['bag'])} messages)")
    summarize(
        "HEADER TIME",
        header_diff[header_valid],
        estimate["bag"][header_valid],
        threshold_s,
        args.alpha,
        args.test_spacing_s,
    )
    summarize(
        "ROSBAG TIME",
        bag_diff[bag_valid],
        estimate["bag"][bag_valid],
        threshold_s,
        args.alpha,
        args.test_spacing_s,
    )

    output_path = args.out or args.bag.with_name(
        f"{args.bag.stem}_velocity_timestamp_difference.png"
    )
    create_plot(
        output_path,
        elapsed_s,
        header_diff,
        header_valid,
        bag_diff,
        bag_valid,
        threshold_s,
    )
    print(f"\nSaved plot: {output_path}")

    if args.csv:
        frame = pd.DataFrame(
            {
                "elapsed_bag_s": elapsed_s,
                "est_header_s": estimate["header"],
                "header_matched_gt_s": gt["header"][header_gt_index],
                "header_difference_ms": header_diff * 1e3,
                "header_match_valid": header_valid,
                "est_bag_s": estimate["bag"],
                "bag_matched_gt_s": gt["bag"][bag_gt_index],
                "bag_difference_ms": bag_diff * 1e3,
                "bag_match_valid": bag_valid,
            }
        )
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"Saved matched timestamps: {args.csv}")


if __name__ == "__main__":
    main()
