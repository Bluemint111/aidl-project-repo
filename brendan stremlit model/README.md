# ShopSphere 30-day purchase propensity

This app trains a supervised multilayer perceptron (MLP) using `train.csv` and scores
the customers in `test.csv`. It engineers customer, transaction, session, support,
retention-contact, delivery, and regional features from `ShopSphere_DataPack`.

The labeled target is the training signal for the 30-day purchase prediction. The
feature cutoff is derived from the latest activity date in the supplied history, so
the app uses behavior available through the historical data cutoff rather than using
the target column as a feature. Validation metrics are measured on a stratified
20% holdout; the final scoring model is then refit on all labeled customers. Scores
are ranking estimates and are not guarantees or causal estimates.

## Run on Windows

From this folder, install the dependencies and start Streamlit:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects the supplied `ShopSphere_DataPack` folder alongside `app.py`.
Click **Train model and score test customers** to train and view the ranked test set.
Download the CSV to get each customer's rank, estimated purchase probability, and
priority flag.

## Gemini API key

Training and scoring are local and do not require a Gemini key. To request a concise
interpretation of aggregate validation metrics and score summaries, enter a Gemini
API key in the app sidebar. The key is used only for that request and is not written
to disk. Gemini receives aggregate statistics only; no customer IDs or individual
features are sent. Alternatively, set `GEMINI_API_KEY` in the environment before
starting the app.

## Reading the metrics

- **ROC-AUC** measures how well the model ranks positive customers above negative
  customers across thresholds.
- **Average precision** summarizes precision/recall performance and is useful with
  the imbalanced positive class.
- **Precision/recall in the top 10%** describes the validation holdout's highest
  scoring decile. The app's priority percentage can be adjusted for the scored set.

Before using scores for outreach, apply consent, do-not-contact, open-support-case,
contact-frequency, and other applicable policy checks. A high score is not evidence
that a particular offer will cause a purchase.
