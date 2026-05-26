# TODO

Living checklist. Tick items as work progresses; new ones land at the
bottom of the relevant phase.

Going forward the project is `.py` first — scripts under `scripts/`,
reusable code in `polymarket_insider/`, tests in `tests/`. The only
notebook is `01_data_collection.ipynb` (kept because it's already done
and has useful inline outputs). The final report is Markdown.

## Setup

- [x] Project scaffold (package, deps, .gitignore, README)
- [x] Reusable modules extracted from the original notebooks
- [x] Offline smoke tests + online API smoke script
- [x] On-disk wallet cache + tqdm progress

## Phase 1 — Data collection

- [x] `notebooks/01_data_collection.ipynb`
- [x] Initial batch of slugs in `events_to_scrape.txt` (12 events, ~600 wallets)
- [x] First raw dataset (`data/raw/positions_*.csv`, `wallets_*.csv`)
- [ ] (later, if modeling reveals data shortage) Add more slugs and re-run 01

## Phase 2 — Processing & labeling

- [x] `polymarket_insider/labeling.py` — graded behavior score +
      two-part label (`suspicious = behavior_flag AND outcome_won`),
      all thresholds in `LABELING_CONFIG`
- [x] `scripts/02_label_dataset.py` — load latest raw CSV, apply features +
      labels, write `data/processed/labeled_positions_*.csv`
- [x] Same script exports `data/processed/gold_set_template.csv`
      (~100 balanced rows for hand-review; protected against overwrite)
- [ ] Hand-label the gold set: open `gold_set_template.csv`, fill the
      `gold_label` column (1 = looks insider-like, 0 = looks normal), save as
      `gold_set_labeled.csv`. Used in Phase 4 to measure how well the
      heuristic and models agree with human judgment.

## Phase 3 — EDA

- [x] `scripts/03_eda.py` — class balance, score distribution, behavior
      feature distributions, correlation heatmap, rule co-occurrence
- [x] Outputs in `reports/figures/*.png` and `reports/tables/*.csv`
- [x] `features.behavior_feature_columns()` — single source of truth for the
      leak-free feature set used by EDA and modeling
- [x] Bugfix: `position_concentration` no longer explodes for zero-volume wallets

## Phase 4 — Modeling

- [ ] `polymarket_insider/modeling.py` — preprocessing pipeline
      (`RobustScaler` + optional `SMOTE`), `GroupKFold` by wallet,
      cross-validation + metrics helpers, model save/load
- [ ] `scripts/04_train_baselines.py` — Naive Bayes, kNN, Decision Tree
- [ ] `scripts/05_train_advanced.py` — Random Forest, XGBoost, MLP, LogReg
- [ ] `scripts/06_unsupervised_baseline.py` — Isolation Forest comparison
- [ ] `scripts/07_shap_analysis.py` — feature importance on the best model
- [ ] `scripts/08_association_rules.py` — mlxtend rules on the flag columns
- [ ] Persist trained models to `data/processed/models/`
- [ ] Compare model predictions against the hand-labeled gold set
- [ ] Unit tests for `modeling.py` under `tests/`

## Phase 5 — App and report

- [ ] `app/app.py` — Streamlit dashboard: event slug -> wallet table w/ risk score
- [ ] Wallet inspector view (all positions for one wallet)
- [ ] Update `notebooks/00_problem_definition.md` to match the final
      labeling design (it still describes the original 6-rule version)
- [ ] `reports/REPORT.md` — final report (data, preprocessing, models,
      results, limitations, related work)
- [ ] Slide deck for defense
- [ ] Bibliography (5-10 entries; include Snorkel / weak supervision,
      Easley & O'Hara PIN, `pselamy/polymarket-insider-tracker`)
- [ ] Dry-run presentation under 15 min
