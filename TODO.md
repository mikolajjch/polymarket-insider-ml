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

- [x] `polymarket_insider/modeling.py` — preprocessing pipeline
      (`RobustScaler` + `SMOTE`), `GroupKFold` by wallet,
      cross-validation + metrics helpers, `save_model` / `load_model`
- [x] `scripts/04_train_baselines.py` — Naive Bayes, kNN, Decision Tree
      (best baseline: DT, ROC-AUC 0.97 ± 0.01, F1 0.86 under GroupKFold)
- [x] Persist trained models to `data/processed/models/*.joblib`
- [x] Baseline metrics table at `reports/tables/baseline_metrics.csv`
- [ ] `scripts/05_train_advanced.py` — Random Forest, XGBoost, MLP, LogReg
- [x] `scripts/06_unsupervised_baseline.py` — Isolation Forest, 90% total
      agreement with heuristic, 22% overlap on flagged set (independent
      anomaly detector confirms part of the heuristic's picks)
- [x] `scripts/07_shap_analysis.py` — SHAP on XGBoost; top features
      `totalBought`, `avgPrice`, `very_low_price_entry`, `position_concentration`
- [x] `scripts/08_association_rules.py` — mlxtend apriori + association_rules
      on the rule_* flags + label (top rule: `big_position + contrarian_entry +
      outcome_won -> suspicious` at confidence 72.5%, lift 11.3×)
- [x] `scripts/09_evaluate_vs_gold.py` — predictions vs the hand-labeled gold
      set; XGBoost / RF / heuristic tie at F1 0.80, accuracy 83%
- [ ] (optional, nice-to-have) Unit tests for `modeling.py` under `tests/`

## Phase 5 — App and report

- [x] `app/app.py` — Streamlit dashboard with three views:
      - Event explorer: slug -> top positions ranked by XGBoost risk score
      - Wallet inspector: all positions + risk profile for one wallet
      - Model overview: ranking table, SHAP importance, association rules,
        gold-set evaluation
- [ ] Update `notebooks/00_problem_definition.md` to match the final
      labeling design (it still describes the original 6-rule version)
- [ ] `reports/REPORT.md` — final report (data, preprocessing, models,
      results, limitations, related work)
- [ ] Slide deck for defense
- [ ] Bibliography (5-10 entries; include Snorkel / weak supervision,
      Easley & O'Hara PIN, `pselamy/polymarket-insider-tracker`)
- [ ] Dry-run presentation under 15 min
