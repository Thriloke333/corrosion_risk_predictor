"""
AI-Based Corrosion Risk Prediction System for Metallic Materials
================================================================
EDUCATIONAL PROTOTYPE - trained on a SYNTHETIC (computer-generated) dataset.

This app does NOT predict real corrosion rates and does NOT guarantee real-world
corrosion behaviour. The "labels" (Low / Medium / High) in the training data were
produced by a simple rule-based scoring function written from general
corrosion-engineering principles (see RULES_TEXT below), not measured in a lab
or in the field.

Run with:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------
# 1. SETTINGS
# --------------------------------------------------------------------------

RANDOM_SEED = 42
N_SAMPLES = 3000
CLASSES = ["Low", "Medium", "High"]  # order used in graphs and tables

# Each metal has two numbers (simplified, qualitative, illustrative values):
#   base     : general tendency to corrode in a wet, aggressive atmosphere (0 to 1)
#   cl_sens  : how strongly chloride raises the risk for that metal (0 to 1.5)
# Principles used to pick them:
#   - Iron/steel have no protective passive film, so they rust readily.
#   - Galvanized steel is protected by a sacrificial zinc layer (slower, not zero).
#   - Aluminium, stainless steel and titanium form passive oxide films, so general
#     corrosion is low, BUT chloride can break the film and cause pitting
#     (stainless 316 contains Mo, which improves pitting resistance over 304).
#   - Copper forms a fairly stable patina; titanium is highly resistant overall.
METALS = {
    "Mild steel": {"base": 1.00, "cl_sens": 1.00},
    "Cast iron": {"base": 0.90, "cl_sens": 1.00},
    "Galvanized steel": {"base": 0.45, "cl_sens": 0.90},
    "Aluminium alloy": {"base": 0.30, "cl_sens": 1.30},
    "Copper": {"base": 0.30, "cl_sens": 0.80},
    "Stainless steel 304": {"base": 0.15, "cl_sens": 1.50},
    "Stainless steel 316": {"base": 0.10, "cl_sens": 1.00},
    "Titanium": {"base": 0.03, "cl_sens": 0.10},
}

ENV_FACTOR = {"Indoor": 0.6, "Outdoor": 1.0, "Marine": 1.3}
COATING_FACTOR = 0.35  # a sound protective coating cuts the risk score a lot
MAX_CHLORIDE = 20000  # mg/L (seawater is roughly 19,000 mg/L chloride)

# Score cut-offs that turn the numeric score into a category (illustrative)
LOW_CUTOFF = 0.10
HIGH_CUTOFF = 0.32

RULES_TEXT = """
**How the synthetic labels were created (documented principles):**

1. **Humidity / time of wetness** - atmospheric corrosion needs an electrolyte
   (a moisture film). Below roughly 60 % relative humidity, corrosion of most
   metals is very slow; above it, the risk rises steeply.
2. **Chloride** - chloride ions make the moisture film more conductive and attack
   passive films (pitting of stainless steel and aluminium). Risk increases with
   chloride concentration, with diminishing effect at very high levels
   (a logarithmic relationship was used).
3. **Temperature** - higher temperature generally speeds up electrochemical
   reactions, so risk increases mildly with temperature.
4. **Environment** - indoor < outdoor < marine, in the spirit of the atmospheric
   corrosivity categories of ISO 9223 (C1 to C5).
5. **Metal type** - each metal has a different inherent tendency to corrode and a
   different sensitivity to chloride (see table in the Dataset tab).
6. **Protective coating** - an intact coating (paint, polymer, etc.) acts as a
   barrier and reduces the risk. Real coatings can fail (scratches, pinholes,
   ageing); this simple model does not capture that.
7. **Random scatter** - a little random noise was added so that the data is not
   perfectly clean (real data never is).

The numbers (weights, cut-offs) are **illustrative choices**, not measured values.
"""


# --------------------------------------------------------------------------
# 2. SYNTHETIC DATASET
# --------------------------------------------------------------------------

def risk_score(metal, temp, rh, chloride, environment, coated):
    """Rule-based corrosion-risk SCORE (illustrative, not a corrosion rate)."""
    m = METALS[metal]
    wetness = 1.0 / (1.0 + np.exp(-(rh - 60.0) / 8.0))  # S-curve around 60 % RH
    cl_term = np.log10(1.0 + chloride) / np.log10(1.0 + MAX_CHLORIDE)  # 0 to 1
    temp_factor = 0.7 + 0.8 * np.clip(temp / 40.0, 0.0, 1.0)  # 0.7 to 1.5
    aggressiveness = (
        wetness
        * (0.5 + 1.5 * cl_term * m["cl_sens"])
        * temp_factor
        * ENV_FACTOR[environment]
    )
    score = m["base"] * aggressiveness
    if coated:
        score *= COATING_FACTOR
    return score


def score_to_label(score):
    if score < LOW_CUTOFF:
        return "Low"
    if score < HIGH_CUTOFF:
        return "Medium"
    return "High"


@st.cache_data
def make_synthetic_dataset(n=N_SAMPLES, seed=RANDOM_SEED):
    """Create the SYNTHETIC demonstration dataset."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        metal = rng.choice(list(METALS.keys()))
        env = rng.choice(["Indoor", "Outdoor", "Marine"], p=[0.3, 0.35, 0.35])
        coated = int(rng.random() < 0.5)

        # Realistic-looking ranges for each environment
        if env == "Indoor":
            temp = rng.uniform(10, 35)
            rh = rng.uniform(20, 75)
            chloride = rng.lognormal(mean=2.0, sigma=1.2)  # very low
        elif env == "Outdoor":
            temp = rng.uniform(-5, 42)
            rh = rng.uniform(30, 100)
            chloride = rng.lognormal(mean=4.5, sigma=1.5)  # low to moderate
        else:  # Marine
            temp = rng.uniform(5, 40)
            rh = rng.uniform(55, 100)
            chloride = rng.lognormal(mean=7.5, sigma=1.4)  # moderate to high

        chloride = float(np.clip(chloride, 0, MAX_CHLORIDE))
        score = risk_score(metal, temp, rh, chloride, env, coated)
        score *= rng.lognormal(mean=0.0, sigma=0.25)  # random scatter
        rows.append(
            {
                "metal": metal,
                "temperature_c": round(temp, 1),
                "humidity_pct": round(rh, 1),
                "chloride_mg_l": round(chloride, 1),
                "environment": env,
                "coated": coated,
                "risk_category": score_to_label(score),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. MACHINE LEARNING MODEL (Decision Tree)
# --------------------------------------------------------------------------

CATEGORICAL = ["metal", "environment"]
NUMERIC = ["temperature_c", "humidity_pct", "chloride_mg_l", "coated"]

FRIENDLY_NAMES = {
    "metal": "Metal type",
    "environment": "Environment type",
    "temperature_c": "Temperature",
    "humidity_pct": "Relative humidity",
    "chloride_mg_l": "Chloride concentration",
    "coated": "Protective coating",
}


def build_pipeline():
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", "passthrough", NUMERIC),
        ]
    )
    tree = DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, random_state=RANDOM_SEED
    )
    return Pipeline([("prep", pre), ("tree", tree)])


@st.cache_resource
def train_model():
    """Train on 80 % of the data, keep 20 % aside for testing."""
    df = make_synthetic_dataset()
    X = df[CATEGORICAL + NUMERIC]
    y = df["risk_category"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": pd.DataFrame(
            classification_report(
                y_test, y_pred, labels=CLASSES, output_dict=True, zero_division=0
            )
        ).T.round(3),
        "confusion": confusion_matrix(y_test, y_pred, labels=CLASSES),
        "cv_scores": cross_val_score(build_pipeline(), X, y, cv=5),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return model, metrics


def grouped_importances(model):
    """Add up the importance of one-hot columns so each INPUT gets one number."""
    names = model.named_steps["prep"].get_feature_names_out()
    imps = model.named_steps["tree"].feature_importances_
    totals = {v: 0.0 for v in FRIENDLY_NAMES.values()}
    for name, imp in zip(names, imps):
        short = name.split("__", 1)[1]  # remove "cat__" / "num__"
        for key, friendly in FRIENDLY_NAMES.items():
            if short == key or short.startswith(key + "_"):
                totals[friendly] += imp
                break
    s = pd.Series(totals).sort_values(ascending=False)
    return s


def decision_path_text(model, x_row):
    """Turn the path the tree followed for this input into plain English."""
    prep = model.named_steps["prep"]
    tree = model.named_steps["tree"]
    X_t = prep.transform(x_row)
    names = prep.get_feature_names_out()
    node_ids = tree.decision_path(X_t).indices
    values = np.asarray(X_t.todense()).ravel() if hasattr(X_t, "todense") else X_t.ravel()
    steps = []
    for node in node_ids:
        if tree.tree_.children_left[node] == tree.tree_.children_right[node]:
            continue  # leaf node
        feat_idx = tree.tree_.feature[node]
        thr = tree.tree_.threshold[node]
        short = names[feat_idx].split("__", 1)[1]
        val = values[feat_idx]
        went_left = val <= thr
        if short.startswith("metal_"):
            steps.append(f"Metal is {short[6:]}? **{'No' if went_left else 'Yes'}**")
        elif short.startswith("environment_"):
            steps.append(f"Environment is {short[12:]}? **{'No' if went_left else 'Yes'}**")
        elif short == "coated":
            steps.append(f"Coated? **{'No (uncoated)' if went_left else 'Yes'}**")
        else:
            label = FRIENDLY_NAMES[short]
            unit = {"temperature_c": "°C", "humidity_pct": "%", "chloride_mg_l": "mg/L"}[short]
            op = "≤" if went_left else ">"
            steps.append(f"{label} {op} {thr:.1f} {unit}  (your value: {val:.1f})")
    return steps


def domain_factors(metal, temp, rh, chloride, environment, coated):
    """Plain-language corrosion-engineering reading of the inputs (not ML output)."""
    notes = []
    if rh >= 80:
        notes.append(("up", f"Humidity of {rh:.0f}% is high - a moisture film is likely on the surface most of the time."))
    elif rh >= 60:
        notes.append(("up", f"Humidity of {rh:.0f}% is above the ~60% level where atmospheric corrosion starts to accelerate."))
    else:
        notes.append(("down", f"Humidity of {rh:.0f}% is below ~60%, so the surface is mostly dry and corrosion is slow."))

    if chloride >= 2000:
        notes.append(("up", f"Chloride of {chloride:,.0f} mg/L is high; it promotes rusting and pitting."))
    elif chloride >= 200:
        notes.append(("up", f"Chloride of {chloride:,.0f} mg/L is moderate and increases corrosion risk."))
    else:
        notes.append(("down", f"Chloride of {chloride:,.0f} mg/L is low."))

    if environment == "Marine":
        notes.append(("up", "Marine environment: salt-laden air and spray make it the most aggressive category here."))
    elif environment == "Outdoor":
        notes.append(("up", "Outdoor exposure means rain/dew wetting and pollutants (moderate aggressiveness)."))
    else:
        notes.append(("down", "Indoor environment is generally the least aggressive."))

    if temp >= 30:
        notes.append(("up", f"{temp:.0f} °C is warm, which speeds up electrochemical reactions."))
    elif temp <= 5:
        notes.append(("down", f"{temp:.0f} °C is cold, which slows reaction rates."))

    base = METALS[metal]["base"]
    if base >= 0.8:
        notes.append(("up", f"{metal} has no protective passive film and rusts readily."))
    elif metal.startswith("Stainless") or metal in ("Aluminium alloy", "Titanium"):
        if metal == "Titanium":
            notes.append(("down", "Titanium forms a very stable oxide film and resists corrosion extremely well."))
        else:
            notes.append(("down", f"{metal} relies on a passive film: resistant in general, but chloride can cause pitting."))
    elif metal == "Galvanized steel":
        notes.append(("down", "Galvanized steel has a sacrificial zinc layer that slows steel corrosion."))
    else:
        notes.append(("down", f"{metal} is moderately resistant (forms a fairly stable surface layer)."))

    if coated:
        notes.append(("down", "A protective coating acts as a barrier (assumed intact; real coatings can fail)."))
    else:
        notes.append(("up", "No protective coating: the metal surface is directly exposed."))
    return notes


# --------------------------------------------------------------------------
# 4. USER INTERFACE (Streamlit)
# --------------------------------------------------------------------------

RISK_STYLE = {
    "Low": ("#1b9e5a", "#e6f6ed", "🟢"),
    "Medium": ("#e08a00", "#fff4e0", "🟠"),
    "High": ("#d63031", "#fdeaea", "🔴"),
}

CSS = """
<style>
.main-title {font-size: 2.1rem; font-weight: 800; color: #0b3c5d; margin-bottom: 0;}
.sub-title {color: #5b6b79; margin-top: 0.1rem; margin-bottom: 0.8rem;}
.banner {background:#fff8e1; border-left:6px solid #f2b705; padding:0.7rem 1rem;
         border-radius:6px; color:#5c4800; font-size:0.92rem; margin-bottom:1rem;}
.risk-card {padding:1.2rem 1.4rem; border-radius:14px; text-align:center;
            border:2px solid; margin-bottom:0.8rem;}
.risk-card h1 {margin:0; font-size:2.4rem;}
.risk-card p {margin:0.2rem 0 0 0; color:#444;}
</style>
"""


def plot_distribution(df):
    counts = df["risk_category"].value_counts().reindex(CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    colors = [RISK_STYLE[c][0] for c in CLASSES]
    bars = ax.bar(CLASSES, counts.values, color=colors)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 10, f"{int(v)}", ha="center", fontsize=10)
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Risk category (synthetic label)")
    ax.set_title("Risk categories in the SYNTHETIC training dataset")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual (synthetic label)")
    ax.set_title("Confusion matrix (test set)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def plot_importance(series):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    s = series.sort_values()
    ax.barh(s.index, s.values * 100, color="#2e86c1")
    ax.set_xlabel("Importance in the trained tree (%)")
    ax.set_title("Which inputs does the model rely on most?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def tab_predict(model):
    st.subheader("Enter exposure conditions")
    c1, c2 = st.columns(2)
    with c1:
        metal = st.selectbox("Metal type", list(METALS.keys()))
        temp = st.slider("Temperature (°C)", -10.0, 50.0, 28.0, 0.5)
        rh = st.slider("Relative humidity (%)", 0, 100, 80)
    with c2:
        environment = st.selectbox("Environment type", ["Indoor", "Outdoor", "Marine"], index=1)
        chloride = st.number_input(
            "Chloride concentration (mg/L)",
            min_value=0.0, max_value=float(MAX_CHLORIDE), value=500.0, step=50.0,
            help="Chloride in the surrounding moisture/water. Typical guide: "
                 "indoor ≈ 0-50, outdoor inland ≈ 10-500, coastal/marine ≈ 500-20,000 "
                 "(seawater ≈ 19,000 mg/L).",
        )
        coating = st.radio("Protective coating status", ["Uncoated", "Coated"], horizontal=True)

    if st.button("🔍 Predict corrosion risk", type="primary", use_container_width=True):
        coated = 1 if coating == "Coated" else 0
        x = pd.DataFrame([{
            "metal": metal, "environment": environment,
            "temperature_c": temp, "humidity_pct": float(rh),
            "chloride_mg_l": float(chloride), "coated": coated,
        }])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0]
        proba_map = dict(zip(model.named_steps["tree"].classes_, proba))
        color, bg, icon = RISK_STYLE[pred]

        st.markdown(
            f"""<div class="risk-card" style="border-color:{color}; background:{bg};">
                <p>Predicted corrosion-risk category</p>
                <h1 style="color:{color};">{icon} {pred.upper()}</h1>
                <p>Model confidence for this leaf: {proba_map[pred]*100:.0f}%
                (a property of the decision tree on synthetic data, not a real-world probability)</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Probability for each category (from the tree):**")
        pc = st.columns(3)
        for col, c in zip(pc, CLASSES):
            col.metric(c, f"{proba_map.get(c, 0.0) * 100:.0f}%")

        st.markdown("### Why this prediction?")
        left, right = st.columns(2)
        with left:
            st.markdown("**Corrosion-engineering reading of your inputs**")
            for direction, text in domain_factors(metal, temp, rh, chloride, environment, coated):
                arrow = "🔺 raises risk" if direction == "up" else "🔻 lowers risk"
                st.markdown(f"- {arrow}: {text}")
        with right:
            st.markdown("**Path the decision tree followed for your input**")
            for i, step in enumerate(decision_path_text(model, x), 1):
                st.markdown(f"{i}. {step}")
            st.caption("These are the yes/no questions the trained tree asked, in order, "
                       "before reaching its answer.")

        st.pyplot(plot_importance(grouped_importances(model)))
        st.caption("Overall importance is learned from the synthetic data, so it mainly reflects "
                   "the rules used to create that data.")

    st.info("⚠️ Educational prototype. The result is a *relative* risk category learned from "
            "synthetic data. It is not a corrosion rate and not a guarantee of real-world behaviour.")


def tab_dataset():
    df = make_synthetic_dataset()
    st.subheader("Training dataset: SYNTHETIC (computer-generated)")
    st.warning("This dataset is **synthetic**. It was generated by code using simplified "
               "corrosion-engineering rules. It is **not** experimental or field data, and "
               "its Low/Medium/High labels are **not** measured results.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.pyplot(plot_distribution(df))
    with c2:
        st.markdown("**Dataset summary**")
        st.write(f"- Total samples: **{len(df):,}**")
        st.write("- Inputs: metal, temperature, humidity, chloride, environment, coating")
        st.write("- Output label: Low / Medium / High (rule-generated)")
        st.write(f"- Random seed: {RANDOM_SEED} (same data on every run)")
        st.dataframe(df["risk_category"].value_counts().reindex(CLASSES).rename("count"))
    st.markdown(RULES_TEXT)
    st.markdown("**Metal parameters used in the rule-based scoring (illustrative):**")
    st.dataframe(pd.DataFrame(METALS).T.rename(
        columns={"base": "Base tendency to corrode", "cl_sens": "Chloride sensitivity"}))
    st.markdown("**First 15 rows of the dataset**")
    st.dataframe(df.head(15), use_container_width=True)
    st.download_button("⬇️ Download synthetic dataset (CSV)",
                       df.to_csv(index=False).encode("utf-8"),
                       "synthetic_corrosion_dataset.csv", "text/csv")


def tab_evaluation(metrics):
    st.subheader("Model evaluation (on SYNTHETIC test data)")
    st.write(f"Training samples: **{metrics['n_train']}**, test samples (never seen during training): "
             f"**{metrics['n_test']}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['accuracy'] * 100:.1f}%")
    c2.metric("5-fold CV accuracy", f"{metrics['cv_scores'].mean() * 100:.1f}%")
    c3.metric("CV std. deviation", f"± {metrics['cv_scores'].std() * 100:.1f}%")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Precision / recall / F1 per class**")
        st.dataframe(metrics["report"])
    with right:
        st.pyplot(plot_confusion(metrics["confusion"]))

    st.markdown("### ⚠️ Limitations of evaluating on synthetic data")
    st.markdown(
        """
- **The "ground truth" is just my own rules.** The labels came from a formula, so a high
  accuracy only shows that the tree can *re-learn that formula*. It does **not** show the
  model can predict real corrosion.
- **No real-world validation.** Real corrosion depends on many things missing here:
  exact alloy composition, surface finish, pollutants (SO₂), time of wetness,
  rainfall/wash-off, crevices, galvanic coupling, coating damage, exposure duration, etc.
- **Test data comes from the same generator as training data**, so train and test are
  unrealistically similar. Real data would be messier and the score would likely be lower.
- **Category cut-offs are arbitrary.** Where "Medium" ends and "High" begins was chosen by me.
- **Random noise is artificial.** Real measurement scatter is different and not random in the same way.
- **Accuracy can be misleading** when classes are unequal; check the per-class recall and the
  confusion matrix too. Errors are mostly between *neighbouring* categories (Low↔Medium, Medium↔High).
- **To make this scientific**, the model must be retrained and validated on real data
  (e.g., ISO 9223 exposure-test results, weight-loss coupons, electrochemical measurements, field inspections).
"""
    )


def tab_about():
    st.subheader("About this project")
    st.markdown(
        """
**Aim:** to show, as a learning exercise, how machine learning can classify corrosion-risk
level from basic exposure conditions.

**Method:**
1. Generate a labelled **synthetic** dataset using simplified corrosion-engineering rules.
2. One-hot encode the categorical inputs (metal, environment).
3. Train a **Decision Tree classifier** (scikit-learn, max depth 6) on 80 % of the data.
4. Evaluate on the remaining 20 % and with 5-fold cross-validation.
5. Predict a new case and explain the result (decision path + engineering reasoning).

**Tools:** Python, Streamlit, scikit-learn, pandas, NumPy, Matplotlib. No API key, database, or paid service.

**Important disclaimer**
- This is an **educational prototype**.
- It does **not** predict actual corrosion rate (mm/year, mpy, etc.) or service life.
- It does **not** guarantee real-world corrosion behaviour.
- It must **not** be used for material selection, safety, or design decisions unless it is
  validated against real experimental or field data.

**Possible future work:** train on real exposure/field data, add more variables (pH, SO₂, time of wetness,
alloy grade), compare Decision Tree with Random Forest, add corrosion-rate regression.
"""
    )


def main():
    st.set_page_config(page_title="Corrosion Risk Predictor", page_icon="🛠️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<p class="main-title">🛠️ AI-Based Corrosion Risk Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">for Metallic Materials · B.Tech Metallurgical &amp; Materials Engineering · Mini-Project</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="banner"><b>Educational prototype.</b> The model is trained on a '
        '<b>synthetic dataset</b> (not experimental data). It does not predict real corrosion rates '
        'and cannot guarantee real-world corrosion behaviour.</div>',
        unsafe_allow_html=True,
    )

    model, metrics = train_model()

    t1, t2, t3, t4 = st.tabs(["🔮 Predict", "📊 Dataset & Distribution", "✅ Model Evaluation", "ℹ️ About & Limitations"])
    with t1:
        tab_predict(model)
    with t2:
        tab_dataset()
    with t3:
        tab_evaluation(metrics)
    with t4:
        tab_about()


main()
