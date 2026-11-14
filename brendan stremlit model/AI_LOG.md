# AI Work Log

This is a manually maintained summary of AI-assisted work in this chat. It is not
an automatic transcript or a runtime log of future model runs.

## 2026-11-13 — ShopSphere purchase propensity

- **Request:** Build a supervised deep-learning app using the ShopSphere data to
  estimate which customers are likely to purchase in the next 30 days.
- **Implemented:** Added a Streamlit app and supervised multilayer perceptron. The
  app builds customer, transaction, session, support, retention-contact, delivery,
  and regional features; validates on a stratified holdout; refits on all labeled
  customers; and ranks the test customers by estimated purchase probability.
- **Gemini:** Optional aggregate-insights feature added. Training and scoring are
  local. Gemini receives aggregate model metrics and score summaries only when
  explicitly requested; no API key or customer-level data is written to this log.
- **Validation:** Feature generation covered 80,000 customers using history through
  2024-09-30. Scored all 16,000 test customers. Holdout ROC-AUC: 0.867; average
  precision: 0.558; precision in the top decile: 59.5%; holdout positive rate: 14.6%.
  The Streamlit app was launched and its interface and health endpoint were checked.
- **Files added:** `app.py`, `model.py`, `requirements.txt`, `.env.example`,
  `README.md`.
- **Follow-up request:** Create an AI log for this chat, stored as a local text or
  Markdown file. This file records that request and the resulting log.

## Logging and privacy

This file is a project-level summary, not a verbatim conversation export. Add future
entries manually if the log should remain a record of later changes. Do not put
credentials, API keys, or identifiable customer records here.
