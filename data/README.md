# Data directory

The datasets are grouped by purpose and capture date so the directory stays
easy to scan.

## Layout

```text
data/
├── flight_sessions/       # Dated ROS bag and PX4 ULog captures
│   └── <date>/
│       └── [scenario]/    # Present when a date has multiple scenarios
│           ├── AI/        # AI-assisted runs, when available
│           ├── RAW/       # Raw or baseline runs, when available
│           └── UN/        # Unmodified runs used by the July sessions
└── scaling_experiments/   # ULog-only estimator/scaling parameter studies
    └── <experiment>/
```

Sessions from the same date share one directory. Scenario and experiment names
retain the terminology used by the evaluation scripts and result files.
The two July date folders contain `RAW/` and `UN/` directly because each date is
already a single session.

## Contents

- `flight_sessions/`: nine dated groups containing 23 capture scenarios with
  paired `.bag` and `.ulg` files, or `.bag` files alone for some October sessions.
- `scaling_experiments/`: 12 parameter-study datasets containing `.ulg` files.

Large `.bag` and `.ulg` artifacts remain excluded from Git by the repository's
ignore rules. This index is tracked so future datasets can follow the same layout.
