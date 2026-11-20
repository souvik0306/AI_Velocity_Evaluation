# Evaluation scripts

Evaluation code is grouped by input type and purpose. Flight-session analyzers
mirror the dated groups under `data/flight_sessions/`; ULog analyzers remain
grouped by format under `ulg/`.

```text
scripts/evaluations/
├── flight_sessions/      ROS bag analyzers grouped by capture date
│   ├── july_28/
│   ├── july_31/
│   ├── september_10/
│   ├── september_22/
│   ├── october_02/
│   ├── october_07/
│   ├── october_14/
│   ├── october_15/
│   └── november_19/
├── ulg/                  ULog scaling-experiment analyzers and diagnostics
├── reports/              Cross-session tables, exports, and summaries
├── diagnostics/          Raw sensor inspection tools
├── plots/                Result and comparison plots
└── evaluation_common.py  Shared ROS bag evaluation pipeline
```

Run a flight-session analyzer from the repository root, for example:

```bash
python3 scripts/evaluations/flight_sessions/september_10/high.py
python3 scripts/evaluations/flight_sessions/october_15/medium.py --duration 20s
python3 scripts/evaluations/flight_sessions/november_19/hover.py
python3 scripts/evaluations/flight_sessions/november_19/medium.py
```

Run all ULog evaluations or one scaling configuration:

```bash
python3 scripts/evaluations/ulg/eval_ulg.py --all
python3 scripts/evaluations/ulg/analyze_scaled_5_20s_ulg.py
```

ULog outputs remain in [`results/ulg/`](../../results/ulg/README.md). This
organization changes script locations only; flight windows, clock conventions,
and calculations are unchanged. Flight-session analyzers write into the
matching date and scenario under
[`results/flight_sessions/`](../../results/README.md); cross-session reports
write into `results/reports/`.
