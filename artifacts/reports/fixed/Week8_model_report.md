# ShopSphere - Week 8 Model Checkpoint Report

**Course:** IT2226 AI Innovation & Deep Learning  
**Project:** ShopSphere customer purchase propensity and retention outreach  
**Deliverable:** Week 8 checkpoint - model, evaluation, first test submission

> **Sample framing.** The dataset is a sample of the customer base, not a full census.
> Results describe segment-level tendencies and are not used to make decisions about
> individual customers. The analysis and dashboard contain no customer IDs or individual
> lookup.

## 1. Objective

Estimate the probability that a sampled customer purchases in the 30 days after the
**2024-09-30** data cutoff (October 2024 window) so retention effort can be prioritised by
segment. The modelling grain is one row per sampled customer: 64,000 training rows (14.6%
positive) and 16,000 disjoint test rows.

## 2. Data preparation and features

- Thirteen cleaned source tables are used (customers, customer preferences, geo data,
  sessions, transactions, transaction items, products, deliveries, support tickets, retention
  contacts, campaigns, campaign performance and the retention budget).
- Cleaning follows one convention: ISO dates, quoted booleans, `_clean` categorical fields,
  exact duplicates removed, invalid values blanked, IQR caps only on duration-like fields.
- **180 customer-level features** (174 numeric, 6 categorical) are built strictly from data
  on or before the cutoff, then one-hot encoded and scaled to a 294-column matrix.
- **Seasonality is modelled explicitly.** Because the data spans January 2022 to September
  2024, the pipeline adds historical October and Nov-Dec behaviour per customer: October
  transaction/session counts and shares, holiday-season shares, monthly seasonality indices,
  days since the last October purchase, and a dormant × holiday-affinity interaction. The
  target window is October, so prior-October activity is a direct seasonal signal; holiday
  affinity captures seasonal shoppers who are otherwise inactive.
- The train/test split is provided and disjoint, and the target is in the future window, so
  there is no target leakage.

## 3. Models and validation

A stratified 80/20 split of the training sample gives 51,200 fit rows and 12,800 validation
rows. Because positives are only 14.6%, **PR-AUC** is the primary metric; ROC-AUC, F1,
precision, recall, log loss and Brier score are also reported. Classification thresholds are
tuned on validation F1.

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Dummy prior | 0.5000 | 0.1457 | 0.2544 | 0.1457 | 1.0000 |
| Logistic regression | 0.8717 | 0.5689 | 0.5391 | 0.4901 | 0.5989 |
| Gradient boosting (HGB) | 0.8759 | 0.5747 | 0.5473 | 0.4882 | 0.6225 |
| **Deep MLP (3-seed ensemble)** | 0.8715 | 0.5712 | 0.5465 | 0.4845 | 0.6268 |
| **Ensemble (MLP + HGB) - selected** | **0.8758** | **0.5767** | **0.5493** | 0.5096 | 0.5957 |
| Ensemble (MLP + HGB + LogReg) | 0.8756 | 0.5764 | 0.5469 | 0.4715 | 0.6509 |

The selected model is the **blend of the deep MLP and gradient boosting** (threshold 0.707).
It gives the best PR-AUC and F1 among the strong models. On the test sample it flags
**2,651 of 16,000 rows (16.6%)**. In a like-for-like HGB ablation, adding the seasonality
features improved validation PR-AUC from 0.5695 to 0.5740 and ROC-AUC from 0.8740 to 0.8756.

### Deep-learning model

A 256-128-64 MLP with batch normalisation, dropout, class-weighted binary cross-entropy,
AdamW and early stopping. Three seeds are trained and averaged, which reduces variance.
Validation PR-AUC peaks within 5-7 epochs and then overfits, so early stopping restores each
seed's best epoch. Blending with gradient boosting improves ranking stability over the MLP
alone.

### Main drivers

Permutation importance on the boosted model is dominated by behavioural recency and
frequency: sessions in the last 180 days, days since the last session, monthly average
session volume, loyalty score and tenure. Transaction spend/recency and support experience
follow. Seasonality features are used: `sess_monthly_mean` is the third most important
feature and `sess_oct_share` also enters the top ten, confirming that prior-October browsing
adds signal beyond recency alone.

### Seasonality check

Monthly revenue is strongly seasonal. Relative to the average month in the same country-year,
November averages **1.59x** and December **2.22x** (holiday average **1.90x**), while January
averages **0.64x**. Dormant customers also return in the peak: buyers whose previous order was
180+ days earlier are a larger share of buyers in Nov-Dec than in the off-season. This is why
the model includes month-aware features and why the retention planner applies a seasonal
demand adjustment instead of treating every week as average.

## 4. Segment analysis

Because the data is a sample, results are interpreted as segment estimates. Grouping by tier
shows a clear gradient: **Bronze is 63.7% of the sample but only 33.9% of expected buyers**, while
**Platinum is 8.1% of the sample but 19.4% of expected buyers**.

| Tier | Sample customers | Mean propensity | Share of expected buyers | Flag rate at 0.707 |
|---|---|---|---|---|
| Bronze | 10,198 | 0.179 | 33.9% | 2.0% |
| Silver | 2,830 | 0.502 | 26.3% | 19.9% |
| Gold | 1,670 | 0.659 | 20.4% | 51.9% |
| Platinum | 1,302 | 0.801 | 19.4% | 78.3% |

Country is much less differentiating than tier: mean propensity is 0.32-0.34 across US, UK,
DE, FR and IN, with expected-buyer shares roughly tracking sample size (US 40.6%, UK 20.4%,
DE 19.8%, FR 9.7%, IN 9.6%).

### Customers who fall into multiple groups

Intersecting dimensions is more useful than any single cut. The table below uses **all
conditions at once** (e.g. Bronze AND active AND high session volume).

| Segment (all conditions) | Customers | Mean propensity | Share of expected buyers | Flag rate |
|---|---|---|---|---|
| Email opt-in + high sessions | 3,132 | 0.753 | 43.8% | 65.4% |
| Bronze + email opt-in | 6,377 | 0.198 | 23.4% | 2.5% |
| Bronze + dormant | 4,594 | 0.039 | 3.3% | 0.0% |
| High loyalty + cooling | 548 | 0.603 | 6.1% | 25.4% |
| Gold/Silver + recent complaint | 221 | 0.661 | 2.7% | 46.2% |
| Bronze + active + high sessions | 197 | 0.654 | 2.4% | 34.5% |
| Open complaint (all tiers) | 70 | 0.431 | 0.6% | 27.1% |

Interpretation:

- **Email opt-in + high sessions** is about one fifth of the sample but carries **44% of
  expected buyers** - the best broad target for a consented campaign.
- **Bronze + email opt-in** is large (6,377) but low propensity (0.198); it is a reach play,
  not a high-yield one.
- **Bronze + active + high sessions** is a small niche (197) with high propensity (0.654) -
  a useful lookalike definition.
- **Bronze + dormant** (4,594) has almost no expected buyers; suppress or use a low-cost
  reactivation test only.
- **Gold/Silver + recent complaint** (221, propensity 0.661) should go to service recovery
  rather than a generic promotion.

## 5. Fairness and responsible use

- **No customer IDs or individual lookup** appear in the analysis, dashboard or exports;
  all reporting is aggregated by segment.
- **No direct identifiers** (name, email, phone) are used as features.
- **Consent:** customers with `do_not_contact` are never selected for outreach.
- **Service fairness:** customers with open complaints are excluded from promotional contact
  and routed to service recovery instead.
- **Monitoring:** selection rate is reported by country, tier and intersection segments.
  Large gaps should be reviewed before launch.
- **Predictive, not causal:** propensity ranks likely buyers; it does not estimate uplift.
  A randomised segment holdout is required before scaling incentives.
- **Human oversight:** high-value and complaint-related segments should require approval.

## 6. Agentic component (preview)

The Week 8 build includes a deterministic **policy layer** that the agent will call as a
tool. It takes propensity scores and applies:

1. consent/DNC and channel eligibility,
2. open-complaint and recent-contact suppression,
3. tier-based offer eligibility (20% / 15% voucher, free shipping, service-recovery credit),
4. weekly per-country budget caps and a maximum contact count.

Filtering happens at row level, but the output is an **aggregated segment plan** (country x
tier x offer x channel) with no customer IDs. The interactive dashboard exposes this planner,
including a policy funnel and audit trail. The next iteration wraps it as an agent tool with
an LLM reasoner for offer explanation and a human approval step.

## 7. Deliverables

| Artifact | Path |
|---|---|
| Executed checkpoint notebook | `notebooks/Week8_ShopSphere_Model_Checkpoint.ipynb` |
| Short report | `artifacts/reports/fixed/Week8_model_report.md` |
| Interactive segment explorer/tester | `archive/legacy_dashboard.py` |
| Trained models | `artifacts/models/fixed/` (preprocessor, logreg, HGB, 3 MLP seeds) |
| Metrics and predictions | `artifacts/reports/fixed/` |
| First test submission | `artifacts/outputs/submission.csv` (16,000 rows, 2,651 positives) |

## 8. Run commands

```powershell
# from the ShopSphere project root
py -m streamlit run archive/legacy_dashboard.py   # interactive segment viewer/tester
$env:PYTHONPATH="$PWD\src"; py -m shopsphere.models.train_models  # retrain + rewrite submission
```

## 9. Limitations and next steps

- **Sample, not census:** estimates describe segment tendencies; widen or reweight the
  sample before making population claims.
- **Causal vs predictive:** run a randomised segment holdout to measure incremental uplift.
- **Temporal drift:** validate on the next 30-day window and monitor calibration monthly.
- **Deep model:** try categorical embeddings and target encoding for high-cardinality fields.
- **Ensembling:** replace the fixed blend weight with a stacked meta-learner on out-of-fold
  scores.
- **Agent:** complete tool calling, explanation, guardrails and approvals.
