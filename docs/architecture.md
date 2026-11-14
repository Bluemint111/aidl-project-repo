# ShopSphere Architecture

## Principles

- `src/` contains all runnable Python code.
- `data/raw/` receives new manual files; `data/clean/` is the validated
  prediction input; `data/processed/` is generated runtime data.
- `artifacts/` contains features, snapshots, models, reports, and exported files.
- Root entry points are thin wrappers that add `src` to `sys.path`.
- No hardcoded run dates in the active path; as-of dates are resolved from the
  data or passed through environment variables.

## Layout

```
run_all.py                 full pipeline entry point
app.py                     Streamlit entry point
planner.py                 weekly planner entry point
src/shopsphere/
  common.py                paths, feature lists, shared helpers
  data/
    asof_time.py           as-of resolution and window bounds
    asof_features.py       point-in-time feature builder
    build_features.py      fixed official feature builder
    validate_clean.py      model-input validation and data-quality report
  features/
    prep_contacts.py       historical/current contact preparation
    build_analytics.py     W6/W10 analytics tables
  models/
    mlp_model.py           PyTorch MLP
    scoring.py             fixed and rolling scorers
    train_models.py        fixed W1 model
    snapshot_builder.py    monthly as-of snapshots
    train_rolling.py       rolling W1 train/backtest/live scoring
    train_offer.py         offer purchase and redemption models
    score_offers.py        customer x offer x channel scoring
  pipelines/
    run_all.py             end-to-end orchestration
    planner.py             weekly budgeted recommendation plan
  app/
    app.py                 Streamlit dashboard
    approvals.py           W9 approval workflow and SQLite audit log
data/raw/             new manual files awaiting validation and cleaning
data/clean/                cleaned source CSVs; sole prediction input
data/processed/            contacts, scores, analytics CSVs, approvals.sqlite
artifacts/features/        fixed feature pickles and metadata
artifacts/snapshots/       rolling snapshots and manifest
artifacts/models/          fixed/, rolling/, offer/
artifacts/reports/         fixed/, rolling/, offer/, analytics/, plans/, logs/
artifacts/outputs/         plans, submissions, approved exports
```

## Data flow

New data is added manually to `data/raw/`, cleaned, and promoted into
`data/clean/`; the latest transaction date there is the live as-of. Prediction
code reads only from `data/clean/`. If an agentic cleaning step is added later,
it should work in `data/raw/` and promote only validated, cleaned files
into `data/clean/`.

1. `validate_clean.py` checks files, keys, references, dates and ranges, then
   writes `artifacts/reports/data_quality.json`.
2. `snapshot_builder.py` builds monthly as-of snapshots from clean data.
3. `train_rolling.py` trains MLP, HGB, and logistic regression, tunes the
   ensemble threshold on the validation month, refits, and writes live scores.
4. `prep_contacts.py` joins live scores to current customers and historical
   contact features.
5. `train_offer.py` trains purchase/redemption models on a dynamic time split.
6. `score_offers.py` creates customer x offer x channel candidates.
7. `build_analytics.py` writes W6/W10 tables.
8. `planner.py` and `app.py` consume processed data and artifacts.

## Artifact ownership

| Path | Producer |
| --- | --- |
| `data/processed/*.pkl`, `*.parquet`, `*.csv` | prep/scoring/analytics |
| `data/processed/approvals.sqlite` | approvals |
| `artifacts/features/` | build_features |
| `artifacts/snapshots/` | snapshot_builder |
| `artifacts/models/fixed/` | train_models |
| `artifacts/models/rolling/` | train_rolling |
| `artifacts/models/offer/` | train_offer |
| `artifacts/reports/fixed/` | train_models |
| `artifacts/reports/rolling/` | train_rolling |
| `artifacts/reports/offer/` | train_offer, score_offers |
| `artifacts/reports/analytics/` | build_analytics |
| `artifacts/reports/data_quality.json` | validate_clean |
| `artifacts/reports/plans/` | planner |
| `artifacts/outputs/` | planner, approvals, train_models |
