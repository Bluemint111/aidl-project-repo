# ShopSphere Retention Intelligence

Integrated solution for W1, W3, W6, W9 and W10.

## Quick start

```powershell
py run_all.py
py -m streamlit run app.py
```

`run_all.py` rebuilds the rolling W1 model when needed, contact features, the
offer-response models, candidate scores, analytics tables and the approval
database. The Streamlit app reads the generated files.

New data is added manually to `data/raw/`, cleaned, and promoted into
`data/clean/`. The as-of date always comes from the latest transaction in
`data/clean/`, so the pipeline treats the newest cleaned data as live. After
adding rows, refresh the scores without retraining:

```powershell
$env:PYTHONPATH="$PWD\src"
py -m shopsphere.data.validate_clean
py -m shopsphere.models.snapshot_builder --live-only
py -m shopsphere.models.train_rolling --live-only
py -m shopsphere.models.score_offers
```

`SHOPSPHERE_AS_OF` can pin a date when needed. The app runs locally; no hosted
server is required.

For a future agentic cleaning step, keep incoming files separate from
`data/clean/`: the agent should validate and clean files in `data/raw/`
first, then promote only cleaned files into `data/clean/`, which remains the
sole prediction input.

## What each tab covers

| Tab | Requirement | Content |
| --- | --- | --- |
| Overview | W6 | Country/date KPIs, revenue trends, country scorecard, carrier reliability |
| Campaigns | W10 | Campaign ROAS/CPA, retention-offer lift, regional difference-in-differences |
| Seasonality | W6 | Month/year revenue index, holiday uplift, dormant-customer reactivation, YoY trends |
| Propensity | W1 | Model comparison, validation metrics, score distribution, segment summary |
| Recommendations | W3 | Weekly budgeted offer/channel plan with guardrails, seasonal demand factor and CSV download |
| Approvals | W9 | Create batches, approve/reject with reviewer and reason, audit trail, export |

## Key outputs

- `data/processed/historical_contacts.pkl`: contact-time features for 9,000 historical contacts.
- `data/processed/current_customers.pkl`: 80,000-customer snapshot with W1 scores.
- `data/processed/offer_candidate_scores.parquet`: 466,349 customer x offer x channel candidates.
- `data/processed/analytics_*.csv`: W6/W10 dashboard tables, including
  `analytics_seasonality_month.csv` and `analytics_seasonality_calendar.csv`.
- `artifacts/models/fixed|rolling|offer/`: preprocessing and model artifacts.
- `artifacts/reports/offer/offer_metrics.json`: offer-model validation metrics.
- `artifacts/reports/analytics/analytics_summary.json`: analytics row counts and date coverage.
- `data/processed/approvals.sqlite`: W9 batches, recommendations and audit log.
- `artifacts/outputs/`: downloaded plans and approved-batch exports.

## Model summary

W1 selected model: rolling Ensemble (MLP + HGB), validation ROC-AUC 0.9194,
PR-AUC 0.6230, F1 0.5967 at threshold 0.608. The model is trained on monthly
as-of snapshots and predicts purchases in the 30-day window after each snapshot.
Target-window seasonality features include target-month shares, holiday
proximity, and country/region demand indices.

Offer purchase model: Gradient Boosting, validation ROC-AUC 0.8347, PR-AUC 0.3199,
F1 0.3686 at threshold 0.603. Redemption model: ROC-AUC 0.9795, PR-AUC 0.9469.

## Seasonality

Revenue is strongly seasonal: November averages about 1.6x and December about 2.2x a normal
month (holiday average 1.9x), while January is about 0.6x. Dormant customers are more likely
to return during the Nov-Dec peak. The Seasonality tab exposes these month/year patterns, and
the W3 planner applies a same-month seasonal demand factor (clamped to 0.75x-1.35x for
stability) to expected incremental revenue. This stops a seasonal lull from being mistaken
for churn and stops a campaign being credited for demand that would have occurred anyway.

## W3 assumptions

- Incremental effect is modelled as `purchase_probability - baseline_probability`.
- Expected incentive cost is redemption-weighted: `offer_face_value * redemption_probability`.
- Default planner objective is expected incremental revenue with `minimum expected net = 0`.
- Guardrails: country weekly budget, DNC/consent filters, service-recovery routing,
  14-day contact cooldown, one contact per customer per batch.
- The planner intentionally produces a small profitable plan rather than spending the
  full budget when incremental net value would be negative.

## W9 approval workflow

1. Build a W3 plan.
2. Enter a preparer name and create an approval batch.
3. Open the Approvals tab, choose a batch, and record an approval or rejection with a
   reviewer name and reason.
4. Only approved batches can be exported. Every create/decision/export action is recorded
   in the audit log.

## Privacy

Raw `customer_id` stays inside internal feature artifacts. Plans, approval records,
exports and app displays use pseudonymous `sample_ref` values only. No customer-level
lookup is provided.

## Caveats

- The data is treated as a sample, not a census.
- Retention offers are observational, not randomised; offer lift uses an `OFF-NONE`
  comparison standardised to tier x country x channel, and intervals crossing zero mean
  no distinguishable effect.
- Campaign attribution is platform-attributed. The regional DiD is quasi-experimental.
- Redemption model performance is unusually high; treat its probabilities as a cost
  weighting aid rather than a precise forecast.
