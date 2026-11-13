# ShopSphere Architecture Scaffold

Empty project frame matching the ShopSphere `src` + `data` + `artifacts`
layout. Folders contain `.gitkeep` placeholders; the `src/shopsphere` packages
contain only empty `__init__.py` files.

## Layout

```
archive/
artifacts/
  features/
  snapshots/
  models/
    fixed/
    rolling/
    offer/
  reports/
    fixed/
    rolling/
    offer/
    analytics/
    plans/
    logs/
  outputs/
configs/
data/
  raw/
  processed/
docs/
notebooks/
scripts/
src/
  shopsphere/
    data/
    features/
    models/
    pipelines/
    app/
tests/
```

## Basic files

- `README.md`
- `.gitignore`
- `requirements.txt`
- `pyproject.toml`
- `configs/config.example.yaml`

## Intended flow

1. Put immutable source CSVs in `data/raw`.
2. Generate processed tables and live scores in `data/processed`.
3. Write features, snapshots, models, reports, and exports under `artifacts`.
4. Keep runnable code under `src/shopsphere` and thin entry points at the
   project root.
