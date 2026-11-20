# Results directory

Generated outputs are grouped by purpose. Flight-session results mirror the
dated scenarios under `data/flight_sessions/`.

```text
results/
├── flight_sessions/  Per-flight CSVs and plots grouped by date and scenario
├── reports/          Cross-session tables, comparisons, and exports
├── diagnostics/      Sensor overlays and timestamp inspections
└── ulg/              ULog scaling-experiment outputs
```

The flight-session hierarchy is:

```text
flight_sessions/<date>/<scenario>/
├── extracted_csv/  Values extracted from the source bag
├── cleaned_csv/    Cleaned and aligned full-flight values
├── <duration>/     Evaluation windows and summaries, such as `20s/`
├── plots/          Scenario-specific plots, when present
└── comparisons/    Comparisons scoped to this scenario, when present
```

Date and scenario names match the corresponding folders in
[`data/flight_sessions/`](../data/flight_sessions/). Results spanning several
dates or scenarios belong in `reports/`, while inspection artifacts that are
not evaluation results belong in `diagnostics/`.

No result artifacts were removed during this reorganization. Existing summary
file names and dataset labels were retained for compatibility.
