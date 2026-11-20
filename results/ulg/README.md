# ULog results

Start with the [combined summary](summary/run_summary.csv),
[15s averages](summary/averages_15s.csv), or
[20s averages](summary/averages_20s.csv).

```text
summary/              Current combined reports and average tables
datasets/<dataset>/  Current per-flight evaluation CSVs
diagnostics/          Timestamp-alignment audit and diagnostic plots
archive/<dataset>/   Historical CSVs, excluded from current summaries
```

`datasets/` groups each dataset's 15s and 20s CSVs together where both are
configured. Original and Scaled 2/5/10 have both durations: the 20s evaluations
keep the same boot-clock starts as their 15s counterparts and extend each end
to `start_s + 20.0`.
The existing Custom folder labels are retained in
`datasets/scaled_custom_gyr_008/` and `datasets/scaled_custom_new_gyr_0003/`;
filenames, timings, and calculations are unchanged.

The four older 20s files under `archive/original/`, `archive/scaled_2/`,
`archive/scaled_5/`, and `archive/scaled_10/` are preserved unchanged. They have
not been validated by the current evaluator and are excluded from all current
summaries. Freshly evaluated 20s CSVs in `datasets/<dataset>/` supersede these
historical copies. No result files were deleted during reorganization.

To regenerate all currently configured evaluations and summaries from the
repository root:

```bash
python3 scripts/evaluations/ulg/eval_ulg.py --all
```

To regenerate the Original Repeat Y-axis diagnostic plot:

```bash
python3 scripts/evaluations/ulg/plots/diagnose_original_repeat_flight1_y.py
```

The active clock conventions and evaluation process are documented in
[the ULog evaluation README](../../scripts/evaluations/ulg/README.md).
