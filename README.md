# ShopSphere Retention Intelligence

Rolling 30-day purchase propensity (W1), budgeted offer planning (W3),
performance analytics (W6), approval workflow (W9), and campaign effectiveness
(W10).

## Quick start

```powershell
py run_all.py
py -m streamlit run app.py
```

`run_all.py` builds the rolling W1 model when it is missing, prepares contact
features, trains the offer-response models, scores candidates, builds analytics
tables, and initializes the approval database. Set
`SHOPSPHERE_RETRAIN_ROLLING=1` to force a rolling-model rebuild, or
`SHOPSPHERE_SKIP_ROLLING=1` to reuse existing rolling artifacts.

## Layout

```
configs/                  runtime configuration examples
data/raw/            new raw files awaiting validation and cleaning
data/clean/               cleaned source CSVs; sole prediction input
data/processed/           generated customer, score, analytics, and approval data
artifacts/features/       fixed-model feature artifacts
artifacts/snapshots/      rolling as-of snapshots
artifacts/models/         fixed/, rolling/, and offer/ model artifacts
artifacts/reports/        fixed/, rolling/, offer/, analytics/, plans/, logs/
artifacts/outputs/        plans, submissions, approved exports
src/shopsphere/           application package
  common.py               shared paths, helpers, feature lists
  data/                   as-of feature engineering
  features/               contact preparation and analytics
  models/                 training, scoring, snapshots, offer models
  pipelines/              run_all and planner entry points
  app/                    Streamlit dashboard and approvals
notebooks/                model checkpoint notebook
docs/                     brief, architecture, and solution notes
scripts/                  optional operational scripts
tests/                    test suite
archive/                  legacy files kept for reference
```

## Entry points

- `py run_all.py` - full pipeline
- `py -m streamlit run app.py` - dashboard
- `py planner.py --country US` - weekly plan from the command line
- `py -m shopsphere.models.snapshot_builder` - build rolling snapshots
- `py -m shopsphere.models.train_rolling` - train/backtest the rolling W1 model
- `py -m shopsphere.models.snapshot_builder --live-only` - refresh only the live snapshot
- `py -m shopsphere.models.train_rolling --live-only` - rescore live features with existing models
- `py -m shopsphere.data.validate_clean` - validate `data/clean/` before training or scoring

When running a module directly, add `src` to `PYTHONPATH`:

```powershell
$env:PYTHONPATH="$PWD\src"; py -m shopsphere.pipelines.planner --country US
```

## Week 8 model checkpoint

Rebuild the fixed 30-day propensity features, retrain the baselines and deep MLP
ensemble, and rewrite the first test submission:

```powershell
$env:PYTHONPATH="$PWD\src"
py -m shopsphere.data.build_features
py -m shopsphere.models.train_models
py -m pytest -q
```

Artifacts are written to `artifacts/features/`, `artifacts/reports/fixed/`, and
`artifacts/outputs/submission.csv`. The executed checkpoint write-up is
`notebooks/Week8_ShopSphere_Model_Checkpoint.ipynb`; the short report is
`artifacts/reports/fixed/Week8_model_report.md`.

## New data

New data is added manually to `data/raw/`, cleaned, and promoted into
`data/clean/`. The pipeline always treats the latest transaction date in
`data/clean/` as the live as-of date, so no simulation or separate live pack is
used. After new rows are added, refresh the newest scores without retraining:

```powershell
$env:PYTHONPATH="$PWD\src"
py -m shopsphere.data.validate_clean
py -m shopsphere.models.snapshot_builder --live-only
py -m shopsphere.models.train_rolling --live-only
py -m shopsphere.models.score_offers
py -m shopsphere.pipelines.planner --country US
```

`SHOPSPHERE_AS_OF` can pin a specific as-of date when needed. The app runs
locally with `py app.py` or `py -m streamlit run app.py`; no hosted server is
required.

If an agentic cleaning step is added later, keep the trust boundary explicit:
new files land in `data/raw/`, are validated and cleaned there, and only
cleaned files are promoted into `data/clean/`. Prediction and planning code
read only from `data/clean/`.
