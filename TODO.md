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

- [ ] `scripts/02_label_dataset.py` — load latest raw CSV, apply features +
      heuristic labels, write `data/processed/labeled_positions_*.csv`
- [ ] Same script exports `data/processed/top50_suspicious.csv` and
      `bottom50_suspicious.csv` for hand-review

## Phase 3 — EDA

- [ ] `scripts/03_eda.py` — load processed CSV, write plots to
      `reports/figures/*.png` and summary tables to `reports/tables/*.csv`
- [ ] Cover: feature distributions, correlations, class balance,
      rule-flag co-occurrence

## Phase 4 — Modeling

- [ ] `polymarket_insider/modeling.py` — preprocessing pipeline
      (RobustScaler, GroupKFold by wallet), class-balance helpers
- [ ] `scripts/04_train_baselines.py` — Naive Bayes, kNN, Decision Tree
- [ ] `scripts/05_train_advanced.py` — Random Forest, XGBoost, MLP, LogReg
- [ ] `scripts/06_unsupervised_baseline.py` — Isolation Forest comparison
- [ ] `scripts/07_shap_analysis.py` — feature importance on the best model
- [ ] `scripts/08_association_rules.py` — mlxtend rules on the flag columns
- [ ] Persist trained models to `data/processed/models/`
- [ ] Unit tests for `modeling.py` under `tests/`

## Phase 5 — App and report

- [ ] `app/app.py` — Streamlit dashboard: event slug -> wallet table w/ risk score
- [ ] Wallet inspector view (all positions for one wallet)
- [ ] `reports/REPORT.md` — final report (data, preprocessing, models,
      results, limitations, related work)
- [ ] Slide deck for defense
- [ ] Bibliography (5-10 entries; include `pselamy/polymarket-insider-tracker`)
- [ ] Dry-run presentation under 15 min
