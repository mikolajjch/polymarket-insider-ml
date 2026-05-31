"""Streamlit dashboard: inspect Polymarket positions and their risk scores.

Three views, navigable from the sidebar:

  - Event explorer     pick an event slug, see positions ranked by model risk
  - Wallet inspector   inspect one wallet's positions and overall risk
  - Model overview     metrics, SHAP importance, association rules, gold-set eval

Risk score per position = XGBoost predict_proba on the leak-free behavior
features (the same features the model saw during cross-validation training).

Run from the project root:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polymarket_insider import features, modeling


PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
TABLES_DIR = ROOT / "reports" / "tables"
FIGS_DIR = ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Polymarket Insider Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -- Caching -----------------------------------------------------------------


@st.cache_data
def load_dataset():
    files = sorted(PROCESSED_DIR.glob("labeled_positions_*.csv"))
    if not files:
        return None
    return pd.read_csv(files[-1])


@st.cache_resource
def load_xgboost():
    path = MODELS_DIR / "xgboost.joblib"
    if not path.exists():
        return None
    return modeling.load_model(path)


@st.cache_data
def add_model_risk(df):
    """Append `model_risk_score` (XGBoost predict_proba for class 1)."""
    pipe = load_xgboost()
    if pipe is None or df is None:
        return df
    feature_cols = features.behavior_feature_columns(df)
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = df.copy()
    out["model_risk_score"] = pipe.predict_proba(X)[:, 1]
    return out


# -- Formatting helpers ------------------------------------------------------


def fmt_money(x):
    if pd.isna(x):
        return "-"
    return "${:,.0f}".format(x)


def styled_table(sub, cols, height=600):
    """Render a DataFrame with risk-score gradient and pretty money columns."""
    fmt = {
        "totalBought": fmt_money,
        "account_total_traded_volume": fmt_money,
        "avgPrice": "{:.3f}",
        "position_concentration": "{:.3f}",
        "suspicious_score": "{:.3f}",
        "model_risk_score": "{:.3f}",
        "outcome_return_ratio": "{:+.1%}",
    }
    fmt = {k: v for k, v in fmt.items() if k in cols}
    styled = (
        sub[cols]
        .style.background_gradient(
            subset=[c for c in ["model_risk_score"] if c in cols],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
        )
        .format(fmt, na_rep="-")
    )
    st.dataframe(styled, use_container_width=True, height=height)


# -- Sidebar -----------------------------------------------------------------


st.sidebar.title("Polymarket Insider Detector")
st.sidebar.caption("Weak-supervision ML on Polymarket positions")

view = st.sidebar.radio(
    "Widok",
    ["Event explorer", "Wallet inspector", "Model overview"],
)

st.sidebar.markdown("---")
st.sidebar.caption("**Najlepszy model:** XGBoost  ROC-AUC 0.997")
st.sidebar.caption("GroupKFold-by-wallet, 5 foldów, SMOTE")
st.sidebar.caption("12 161 pozycji,  5 496 portfeli,  12 eventów")


# -- Data + model loading ----------------------------------------------------


df_raw = load_dataset()
if df_raw is None:
    st.error(
        "Brak zlabelowanego zbioru. Uruchom najpierw "
        "`python scripts/02_label_dataset.py`."
    )
    st.stop()

if load_xgboost() is None:
    st.error(
        "Brak modelu XGBoost. Uruchom najpierw "
        "`python scripts/05_train_advanced.py`."
    )
    st.stop()

df = add_model_risk(df_raw)


# -- Views -------------------------------------------------------------------


if view == "Event explorer":
    st.header("Event explorer")
    st.caption(
        "Wybierz event, by zobaczyć jego pozycje uszeregowane po "
        "przewidywanym przez model ryzyku insider tradingu. Wyższy wynik "
        "= mocniejsze zachowanie insiderskie."
    )

    events = sorted(df["event_slug"].dropna().unique())
    chosen = st.selectbox("Event slug", events)

    sub = df[df["event_slug"] == chosen].copy()
    n_markets = sub["market_question"].nunique()
    st.write(
        "**{:,} pozycji** w tym evencie (w {} rynkach)".format(len(sub), n_markets)
    )

    sub = sub.sort_values("model_risk_score", ascending=False)
    cols = [
        "market_question", "side", "proxyWallet",
        "totalBought", "avgPrice", "account_total_traded_volume",
        "position_concentration",
        "suspicious_score", "model_risk_score",
        "outcome_won", "outcome_return_ratio",
    ]
    cols = [c for c in cols if c in sub.columns]
    styled_table(sub.head(50), cols)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pozycje", "{:,}".format(len(sub)))
    c2.metric("Średni risk score", "{:.3f}".format(sub["model_risk_score"].mean()))
    c3.metric(
        "Pozycje z risk >= 0.5",
        "{:,}".format(int((sub["model_risk_score"] >= 0.5).sum())),
    )


elif view == "Wallet inspector":
    st.header("Wallet inspector")
    st.caption("Sprawdź pozycje wybranego portfela i jego ogólny profil ryzyka.")

    wallet_summary = df.groupby("proxyWallet").agg(
        max_risk=("model_risk_score", "max"),
        n_positions=("market_question", "count"),
        lifetime_vol=("account_total_traded_volume", "first"),
    ).sort_values("max_risk", ascending=False)

    st.markdown("**Top 10 portfeli po maksymalnym risk score modelu:**")
    st.dataframe(
        wallet_summary.head(10).style.format({
            "max_risk": "{:.3f}",
            "n_positions": "{:.0f}",
            "lifetime_vol": fmt_money,
        }),
        use_container_width=True,
    )

    wallet = st.text_input(
        "Adres portfela (wklej z tabeli wyżej)",
        value=wallet_summary.index[0],
    )
    sub = df[df["proxyWallet"] == wallet].copy()
    if sub.empty:
        st.warning("Nie znaleziono pozycji dla tego portfela.")
        st.stop()

    st.markdown("**{} pozycji** dla `{}`".format(len(sub), wallet))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lifetime volume", fmt_money(sub["account_total_traded_volume"].iloc[0]))
    c2.metric("Łączna stawka", fmt_money(sub["totalBought"].sum()))
    c3.metric("Max risk", "{:.3f}".format(sub["model_risk_score"].max()))
    c4.metric("Średni risk", "{:.3f}".format(sub["model_risk_score"].mean()))

    sub = sub.sort_values("model_risk_score", ascending=False)
    cols = [
        "event_slug", "market_question", "side",
        "totalBought", "avgPrice",
        "suspicious_score", "model_risk_score",
        "outcome_won", "outcome_return_ratio",
    ]
    cols = [c for c in cols if c in sub.columns]
    styled_table(sub, cols, height=500)


else:  # Model overview
    st.header("Model overview")
    st.caption("Porównanie wszystkich wytrenowanych modeli + interpretacja.")

    metrics_path = TABLES_DIR / "all_models_metrics.csv"
    if metrics_path.exists():
        st.subheader("Ranking modeli (GroupKFold-by-wallet, 5 foldów, SMOTE)")
        metrics = pd.read_csv(metrics_path).set_index("model")
        st.dataframe(
            metrics.style.format("{:.3f}").background_gradient(
                subset=[c for c in ["roc_auc_mean"] if c in metrics.columns],
                cmap="Greens",
            ),
            use_container_width=True,
        )

    shap_path = TABLES_DIR / "shap_importance.csv"
    if shap_path.exists():
        st.subheader("XGBoost — istotność cech (SHAP)")
        st.caption(
            "Średnia magnituda wkładu cechy do predykcji, liczona na 500 próbkach."
        )
        st.bar_chart(
            pd.read_csv(shap_path).set_index("feature")["mean_abs_shap"]
        )

    fig_shap = FIGS_DIR / "shap_beeswarm.png"
    if fig_shap.exists():
        st.image(str(fig_shap), caption="SHAP beeswarm: per-row contribution")

    rules_path = TABLES_DIR / "association_rules_predict_suspicious.csv"
    if rules_path.exists():
        st.subheader("Reguły asocjacyjne predykujące `suspicious`")
        st.caption("mlxtend apriori, posortowane po lifcie")
        st.dataframe(
            pd.read_csv(rules_path).head(10).style.format({
                "support": "{:.3f}",
                "confidence": "{:.3f}",
                "lift": "{:.2f}",
            }),
            use_container_width=True,
        )

    gold_path = TABLES_DIR / "gold_evaluation.csv"
    if gold_path.exists():
        st.subheader("Ewaluacja vs ręczny gold set (100 wierszy)")
        st.caption(
            "Pierwszy uczciwy benchmark: model vs człowiek, "
            "nie model vs heurystyka."
        )
        gold = pd.read_csv(gold_path).set_index("predictor")
        st.dataframe(
            gold.style.format({
                "accuracy": "{:.3f}", "precision": "{:.3f}",
                "recall": "{:.3f}", "f1": "{:.3f}", "agreement": "{:.3f}",
                "n_pred_pos": "{:.0f}", "n_true_pos": "{:.0f}", "n": "{:.0f}",
            }),
            use_container_width=True,
        )
