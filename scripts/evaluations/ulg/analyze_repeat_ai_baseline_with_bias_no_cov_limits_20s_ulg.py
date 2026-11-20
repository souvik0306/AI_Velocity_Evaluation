#!/usr/bin/env python3
"""Evaluate the repeated AI baseline over 20-second log-relative windows."""

from typing import Dict, Tuple

from eval_ulg import LOG_TIME_REFERENCE, REPO_ROOT, RESULTS_ROOT, main


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "scaling_experiments" / "Scaled_AI_Repeat_bias_true_limits_false"
DEFAULT_OUTPUT = (
	RESULTS_ROOT
	/ "datasets"
	/ "repeat_ai_baseline_with_bias_no_cov_limits"
	/ "repeat_ai_baseline_with_bias_no_cov_limits_20s_bias_corrected_rmse.csv"
)
TIME_REFERENCE = LOG_TIME_REFERENCE

# Supplied times are measured from the start of each ULog, not from PX4 boot.
FLIGHTS: Tuple[Dict[str, object], ...] = (
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_1", "ulg": "07_47_36.ulg", "start_s": 31.64, "end_s": 51.64},
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_2", "ulg": "07_50_51.ulg", "start_s": 33.99, "end_s": 53.99},
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_3", "ulg": "07_54_47.ulg", "start_s": 34.19, "end_s": 54.19},
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_4", "ulg": "07_58_48.ulg", "start_s": 33.42, "end_s": 53.42},
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_5", "ulg": "08_06_04.ulg", "start_s": 29.06, "end_s": 49.06},
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_6", "ulg": "08_09_11.ulg", "start_s": 32.08, "end_s": 52.08},
	{"flight": "repeat_ai_baseline_with_bias_no_cov_limits_7", "ulg": "08_11_47.ulg", "start_s": 42.87, "end_s": 62.87},
)


if __name__ == "__main__":
	raise SystemExit(
		main(
			DEFAULT_DATA_DIR,
			DEFAULT_OUTPUT,
			FLIGHTS,
			time_reference=TIME_REFERENCE,
		)
	)
