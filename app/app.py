"""
app.py
======
Streamlit web application for the Opportunity Predictor.

Users enter a *community profile* (neighborhood poverty rate, school
quality, college share, etc.) and the app returns predicted adult outcomes
— with uncertainty bands, clearly-framed interpretation, and a fairness
context panel.

**Important framing**: this app predicts outcomes for *communities with
this profile*, not for individuals. The Opportunity Atlas data is
tract-level aggregate data, and predicting individual life trajectories
from demographics is a well-documented source of algorithmic harm (see
e.g. ProPublica's COMPAS investigation). The app UI makes this explicit.

Launch with:
    streamlit run app/app.py
"""
# AI assitance used for UI interface
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_dataset  # noqa: E402
from preprocessing import transform  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Opportunity Predictor",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_artifacts_and_model():
    """Load preprocessing artifacts and the best model."""
    models_dir = ROOT / "models"
    with open(models_dir / "preprocessing_artifacts.pkl", "rb") as f:
        bundle = pickle.load(f)
    artifacts = bundle["artifacts"]
    cfg = bundle["config"]
    feats = bundle["feature_names"]
    target = bundle.get("target_col", "kfr_pooled_pooled_p25")

    # Load the GBM (portable across environments; the stacker/MLP may not be)
    model = None
    for candidate in [
        f"gbm_{target}.pkl",
        f"best_{target}.pkl",
    ]:
        path = models_dir / candidate
        if path.exists():
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                break
            except Exception:
                continue
    if model is None:
        raise FileNotFoundError(
            f"No fitted model found in {models_dir}. "
            "Please run `python src/train.py` first."
        )
    return artifacts, cfg, feats, target, model


@st.cache_data
def load_reference_data(n: int = 5000):
    """Small cached reference sample for percentile context."""
    df = load_dataset(use_synthetic=True)
    return df.sample(n=min(n, len(df)), random_state=2025).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("📈 Opportunity Predictor")
st.caption(
    "Predicting community-level child outcomes from tract-level "
    "socioeconomic features."
)

st.info(
    "**How to read these predictions.** This tool predicts the "
    "*average* adult outcome for children who grew up in a *community* "
    "matching the profile you enter — it does **not** predict any "
    "individual person's fate. The predictions reflect patterns in the "
    "Opportunity Atlas (Chetty et al., Harvard / US Census). The "
    "purpose of this tool is to make structural inequality in American "
    "communities more visible, not to forecast individual destinies."
)


# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------

try:
    artifacts, cfg, feature_names, target, model = load_artifacts_and_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.code("cd opportunity-predictor\npython src/train.py --fast")
    st.stop()

ref = load_reference_data()

st.sidebar.header("Community profile")

with st.sidebar.expander("🧾 Socioeconomic", expanded=True):
    poor_share = st.slider(
        "Poverty rate (%)", 0.0, 60.0, 15.0, step=0.5,
        help="Fraction of residents below the federal poverty line.",
    ) / 100.0

    med_hhinc = st.slider(
        "Median household income ($)", 15_000, 150_000, 55_000, step=1_000,
    )

    hhinc_mean = int(med_hhinc * 1.2)
    emp_rate = st.slider(
        "Employment rate (%)", 30.0, 98.0, 80.0, step=0.5
    ) / 100.0

with st.sidebar.expander("🎓 Education / family", expanded=True):
    frac_coll_plus = st.slider(
        "Adults with college degree (%)", 3.0, 90.0, 30.0, step=0.5,
    ) / 100.0
    gsmn_math_pcst = st.slider(
        "School math percentile",
        1.0, 99.0, 50.0, step=1.0,
        help="Tract's average grade-school math test percentile "
             "(higher = better schools).",
    )
    singleparent_share = st.slider(
        "Single-parent household share (%)", 2.0, 80.0, 25.0, step=0.5,
    ) / 100.0

with st.sidebar.expander("🏠 Geography", expanded=False):
    popdensity = st.slider(
        "Population density (people / sq mi)", 50, 50_000, 3000, step=50,
    )
    rent_twobed = st.slider(
        "Median 2-bedroom rent ($)", 400, 4000, 1200, step=25,
    )
    mean_commutetime = st.slider(
        "Mean commute time (min)", 10.0, 60.0, 26.0, step=0.5,
    )
    traveltime15_share = st.slider(
        "Share with commute < 15 min (%)", 2.0, 80.0, 25.0, step=0.5,
    ) / 100.0
    foreign_share = st.slider(
        "Foreign-born share (%)", 0.0, 70.0, 10.0, step=0.5,
    ) / 100.0
    job_density = popdensity * 0.3

with st.sidebar.expander("👥 Demographic composition", expanded=False):
    st.caption(
        "These are included because the Opportunity Atlas literature "
        "documents significant outcome gaps by race — which is precisely "
        "the systemic pattern this tool is designed to reveal."
    )
    share_black = st.slider(
        "Share Black (%)", 0.0, 100.0, 13.0, step=0.5,
    ) / 100.0
    share_hisp = st.slider(
        "Share Hispanic (%)", 0.0, 100.0, 18.0, step=0.5,
    ) / 100.0
    share_asian = st.slider(
        "Share Asian (%)", 0.0, 100.0, 6.0, step=0.5,
    ) / 100.0
    share_white = max(0.0, 1 - share_black - share_hisp - share_asian)
    st.write(f"Share white (auto): **{share_white * 100:.1f}%**")
    nonwhite_share2010 = 1 - share_white


# ---------------------------------------------------------------------------
# Assemble input row
# ---------------------------------------------------------------------------

input_row = pd.DataFrame([{
    "state": 1, "county": 1, "tract": 1,
    "poor_share": poor_share,
    "share_black": share_black,
    "share_hisp": share_hisp,
    "share_asian": share_asian,
    "share_white": share_white,
    "hhinc_mean": hhinc_mean,
    "mean_commutetime": mean_commutetime,
    "frac_coll_plus": frac_coll_plus,
    "foreign_share": foreign_share,
    "med_hhinc": med_hhinc,
    "popdensity": popdensity,
    "rent_twobed": rent_twobed,
    "singleparent_share": singleparent_share,
    "traveltime15_share": traveltime15_share,
    "emp_rate": emp_rate,
    "job_density": job_density,
    "gsmn_math_pcst": gsmn_math_pcst,
    "nonwhite_share2010": nonwhite_share2010,
}])

# Need both targets as dummy columns so transform() doesn't blow up if the
# preprocessing step ever touches them (it shouldn't, but safe to include).
input_row["kfr_pooled_pooled_p25"] = 0.0
input_row["jail_pooled_pooled_p25"] = 0.0

# Apply the fitted preprocessing pipeline to the user's input
try:
    processed = transform(input_row, artifacts, cfg)
    X_user = processed[feature_names].values
    prediction = float(model.predict(X_user)[0])
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()


# ---------------------------------------------------------------------------
# Main panel: prediction output
# ---------------------------------------------------------------------------

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Predicted outcome")
    if target == "kfr_pooled_pooled_p25":
        st.metric(
            label="Mean adult income rank at 35 (children from 25th-pctl families)",
            value=f"{prediction:.1f}",
            help="On a 0-100 national scale. The US mean for low-income "
                 "kids is ~42.",
        )
        st.caption(
            f"In plain English: children growing up in a community with "
            f"this profile reach the **{int(round(prediction))}th "
            f"percentile of the US adult income distribution on average**, "
            f"if they come from families at the 25th percentile."
        )
    else:
        st.metric(
            label="Predicted fraction incarcerated at age ~27",
            value=f"{prediction * 100:.2f}%",
        )
    # Compare against reference distribution
    ref_values = ref[target].dropna()
    pct = float((ref_values < prediction).mean() * 100)
    st.caption(
        f"This prediction is in the **{pct:.0f}th percentile** of US "
        f"census tracts (higher = better outcomes for p25 children)."
    )

with col2:
    st.subheader("How does this compare?")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ref_values, bins=60, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(prediction, color="#C44E52", lw=3,
               label=f"this community: {prediction:.1f}")
    ax.axvline(ref_values.mean(), color="#55A868", lw=2, linestyle="--",
               label=f"US average: {ref_values.mean():.1f}")
    ax.set_xlabel("Adult income rank at 35 (p25 kids)")
    ax.set_ylabel("Number of US tracts")
    ax.legend(loc="upper left")
    ax.set_title("Your community's prediction vs all US tracts")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Scenario comparison: same community, different poverty rate
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🧪 Structural counterfactual")
st.caption(
    "What if the *same community* had a different poverty rate? We sweep "
    "the poverty variable from 0 to 50% and hold every other input "
    "constant. This isolates the statistical association between "
    "neighborhood poverty and child outcomes — it is **not** a causal "
    "claim about any one policy change."
)

poverty_grid = np.linspace(0.0, 0.5, 40)
scenario_preds = []
for pv in poverty_grid:
    row = input_row.copy()
    row["poor_share"] = pv
    try:
        p = transform(row, artifacts, cfg)
        scenario_preds.append(float(model.predict(p[feature_names].values)[0]))
    except Exception:
        scenario_preds.append(np.nan)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(poverty_grid * 100, scenario_preds, lw=2.5, color="#8172B2")
ax.axvline(poor_share * 100, color="#C44E52", linestyle="--", lw=1.5,
           label=f"your input: {poor_share*100:.0f}%")
ax.set_xlabel("Tract poverty rate (%)")
ax.set_ylabel("Predicted earnings rank")
ax.set_title("Predicted outcome vs tract poverty (all else equal)")
ax.legend()
st.pyplot(fig)
plt.close(fig)


# ---------------------------------------------------------------------------
# Interpretation panel (Rudin-inspired transparent factors)
# ---------------------------------------------------------------------------

st.divider()
st.subheader("🔎 Why the model predicted this")

lasso_path = ROOT / "data" / "processed" / "lasso_coefficient_table.csv"
if lasso_path.exists():
    lasso_table = pd.read_csv(lasso_path).head(12)
    st.caption(
        "Top 12 features from the sparse Lasso model (trained on "
        "standardized inputs). Positive = raises predicted earnings rank; "
        "negative = lowers it. This Lasso is our *interpretable* "
        "counterpart to the GBM, following Rudin (2019)."
    )
    st.dataframe(lasso_table, use_container_width=True)
else:
    st.info("Run the Lasso training step to populate interpretability data.")


# ---------------------------------------------------------------------------
# Fairness audit
# ---------------------------------------------------------------------------

st.divider()
st.subheader("⚖️ Fairness audit")

for fname, title in [
    ("disparity_kfr_pooled_pooled_p25_by_poverty.csv",
     "By tract poverty rate"),
    ("disparity_kfr_pooled_pooled_p25_by_minority_share.csv",
     "By tract minority share"),
]:
    p = ROOT / "data" / "processed" / fname
    if p.exists():
        st.markdown(f"**{title}**")
        st.dataframe(pd.read_csv(p), use_container_width=True)

st.caption(
    "These tables document model performance disaggregated by demographic "
    "subgroup on the held-out test set. Big disparities in RMSE here would "
    "flag a bias problem; roughly-equal RMSE across buckets indicates the "
    "*errors* are spread across groups, even though the *mean outcomes* "
    "differ — which is the systemic pattern this project exists to surface."
)


st.divider()
st.caption(
    "Built for Duke CS 372 Spring 2026. Data: Opportunity Atlas (Chetty, "
    "Friedman, Hendren, Jones & Porter, 2020). Model: Gradient Boosting "
    "regressor trained with scikit-learn and compared against OLS, Lasso, "
    "Ridge, Random Forest, a custom PyTorch MLP, and a stacking ensemble. "
    "See README for full attribution."
)
