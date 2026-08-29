import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report, confusion_matrix


# ============================================================
# App Configuration
# ============================================================

st.set_page_config(
    page_title="Pokémon Sustainability Resilience Sandbox",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "pokemon_dataset.csv"

RANDOM_STATE = 42

CLASS_ORDER = ["Low", "Medium", "High"]
CLASS_TO_ID = {"Low": 0, "Medium": 1, "High": 2}
ID_TO_CLASS = {0: "Low", 1: "Medium", 2: "High"}

REQUIRED_STAT_COLS = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]
FINAL_FEATURES = ["Type1", "Type2", "HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]

FINAL_REVIEW_THRESHOLD = 0.75
HIGH_RISK_OVERRIDE_THRESHOLD = 0.35


# ============================================================
# Utility Functions
# ============================================================

def clean_column_names(df):
    """Make column names Python-friendly."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )
    return df


def normalize_type2(series):
    """Treat blank Type2 values as missing, because single-type Pokémon have no Type2."""
    return series.replace(["", " ", "None", "none", "nan", "NaN"], np.nan)


def safe_one_hot_encoder():
    """Handle different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# ============================================================
# Data + Model Builder
# ============================================================

@st.cache_resource(show_spinner="Building model from pokemon_dataset.csv...")
def build_model_from_csv():
    """
    Train-on-startup builder.

    This version intentionally trains the model inside Streamlit Cloud instead of loading a .joblib model.
    It avoids joblib / Python / scikit-learn serialization mismatches.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "pokemon_dataset.csv not found. Please upload pokemon_dataset.csv to the root of the GitHub repository."
        )

    df = pd.read_csv(DATA_PATH)
    df = clean_column_names(df)

    required_cols = ["Name", "Type1", "Type2"] + REQUIRED_STAT_COLS
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns are: {df.columns.tolist()}"
        )

    sdf = df.copy()

    sdf["Type1"] = sdf["Type1"].astype(str).str.strip()
    sdf["Type2"] = normalize_type2(sdf["Type2"])

    for col in REQUIRED_STAT_COLS:
        sdf[col] = pd.to_numeric(sdf[col], errors="coerce")

    sdf[REQUIRED_STAT_COLS] = sdf[REQUIRED_STAT_COLS].fillna(
        sdf[REQUIRED_STAT_COLS].median()
    )

    # --------------------------------------------------------
    # Synthetic Sustainability Proxy Construction
    # --------------------------------------------------------

    stat_scaler = MinMaxScaler()

    scaled_stats = pd.DataFrame(
        stat_scaler.fit_transform(sdf[REQUIRED_STAT_COLS]),
        columns=[f"{c}_Scaled" for c in REQUIRED_STAT_COLS],
        index=sdf.index
    )

    sdf = pd.concat([sdf, scaled_stats], axis=1)

    sdf["Resilience_Index"] = sdf[
        ["HP_Scaled", "Defense_Scaled", "Sp_Def_Scaled"]
    ].mean(axis=1)

    sdf["Pressure_Index"] = sdf[
        ["Attack_Scaled", "Sp_Atk_Scaled", "Speed_Scaled"]
    ].mean(axis=1)

    sdf["Adaptability_Index"] = sdf[
        ["Speed_Scaled", "Sp_Def_Scaled", "HP_Scaled"]
    ].mean(axis=1)

    scaled_cols = [
        "HP_Scaled", "Attack_Scaled", "Defense_Scaled",
        "Sp_Atk_Scaled", "Sp_Def_Scaled", "Speed_Scaled"
    ]

    sdf["Stat_Imbalance"] = sdf[scaled_cols].std(axis=1)

    imbalance_scaler = MinMaxScaler()

    sdf["Balance_Index"] = 1 - imbalance_scaler.fit_transform(
        sdf[["Stat_Imbalance"]]
    ).ravel()

    sdf["Balance_Index"] = sdf["Balance_Index"].clip(0, 1)

    sdf["Is_Dual_Type"] = sdf["Type2"].notna().astype(int)
    sdf["Diversity_Index"] = sdf["Is_Dual_Type"]

    sdf["Sustainability_Risk_Score"] = (
        0.40 * sdf["Pressure_Index"] +
        0.30 * (1 - sdf["Resilience_Index"]) +
        0.20 * (1 - sdf["Balance_Index"]) +
        0.10 * (1 - sdf["Diversity_Index"])
    )

    sdf["Sustainability_Risk_Tier"] = pd.qcut(
        sdf["Sustainability_Risk_Score"],
        q=3,
        labels=CLASS_ORDER
    )

    _, risk_bins = pd.qcut(
        sdf["Sustainability_Risk_Score"],
        q=3,
        labels=CLASS_ORDER,
        retbins=True
    )

    low_medium_boundary = float(risk_bins[1])
    medium_high_boundary = float(risk_bins[2])

    # --------------------------------------------------------
    # ML Training
    # --------------------------------------------------------

    X = sdf[FINAL_FEATURES].copy()
    y = sdf["Sustainability_Risk_Tier"].map(CLASS_TO_ID).astype(int)

    categorical_cols = ["Type1", "Type2"]
    numerical_cols = ["HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed"]

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", safe_one_hot_encoder())
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )

    model = HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro F1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall High": recall_score(
            y_test,
            y_pred,
            labels=[CLASS_TO_ID["High"]],
            average="macro",
            zero_division=0
        ),
        "Confusion Matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1, 2]
        ),
        "Classification Report": classification_report(
            y_test,
            y_pred,
            target_names=CLASS_ORDER,
            zero_division=0,
            output_dict=True
        )
    }

    # Refit on full data for app prediction.
    pipeline.fit(X, y)

    pokemon_types = sorted(
        set(sdf["Type1"].dropna().astype(str).unique().tolist()) |
        set(sdf["Type2"].dropna().astype(str).unique().tolist())
    )

    return {
        "raw_df": df,
        "sdf": sdf,
        "pipeline": pipeline,
        "stat_scaler": stat_scaler,
        "imbalance_scaler": imbalance_scaler,
        "metrics": metrics,
        "pokemon_types": pokemon_types,
        "low_medium_boundary": low_medium_boundary,
        "medium_high_boundary": medium_high_boundary
    }


# Build all artifacts.
artifacts = build_model_from_csv()

raw_df = artifacts["raw_df"]
sdf = artifacts["sdf"]
pipeline = artifacts["pipeline"]
stat_scaler = artifacts["stat_scaler"]
imbalance_scaler = artifacts["imbalance_scaler"]
metrics = artifacts["metrics"]
POKEMON_TYPES = artifacts["pokemon_types"]
LOW_MEDIUM_BOUNDARY = artifacts["low_medium_boundary"]
MEDIUM_HIGH_BOUNDARY = artifacts["medium_high_boundary"]


# ============================================================
# Sustainability Scoring Engine
# ============================================================

def compute_sustainability_score(input_df):
    """
    Compute synthetic sustainability proxy score for one Pokémon-like input.
    This is a simulation score, not empirical sustainability evidence.
    """

    temp = input_df.copy()

    temp["Type2"] = normalize_type2(temp["Type2"])

    for col in REQUIRED_STAT_COLS:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    scaled = pd.DataFrame(
        stat_scaler.transform(temp[REQUIRED_STAT_COLS]),
        columns=[f"{c}_Scaled" for c in REQUIRED_STAT_COLS],
        index=temp.index
    ).clip(0, 1)

    resilience_index = scaled[
        ["HP_Scaled", "Defense_Scaled", "Sp_Def_Scaled"]
    ].mean(axis=1)

    pressure_index = scaled[
        ["Attack_Scaled", "Sp_Atk_Scaled", "Speed_Scaled"]
    ].mean(axis=1)

    adaptability_index = scaled[
        ["Speed_Scaled", "Sp_Def_Scaled", "HP_Scaled"]
    ].mean(axis=1)

    stat_imbalance = scaled[
        [
            "HP_Scaled", "Attack_Scaled", "Defense_Scaled",
            "Sp_Atk_Scaled", "Sp_Def_Scaled", "Speed_Scaled"
        ]
    ].std(axis=1).to_frame("Stat_Imbalance")

    balance_index = 1 - imbalance_scaler.transform(stat_imbalance).ravel()
    balance_index = np.clip(balance_index, 0, 1)

    diversity_index = temp["Type2"].notna().astype(int)

    risk_score = (
        0.40 * pressure_index +
        0.30 * (1 - resilience_index) +
        0.20 * (1 - balance_index) +
        0.10 * (1 - diversity_index)
    )

    score_based_tier = pd.cut(
        risk_score,
        bins=[-np.inf, LOW_MEDIUM_BOUNDARY, MEDIUM_HIGH_BOUNDARY, np.inf],
        labels=CLASS_ORDER,
        include_lowest=True
    ).astype(str)

    return {
        "Resilience_Index": float(resilience_index.iloc[0]),
        "Pressure_Index": float(pressure_index.iloc[0]),
        "Adaptability_Index": float(adaptability_index.iloc[0]),
        "Balance_Index": float(balance_index[0]),
        "Diversity_Index": float(diversity_index.iloc[0]),
        "Sustainability_Risk_Score": float(risk_score.iloc[0]),
        "Score_Based_Tier": str(score_based_tier.iloc[0])
    }


# ============================================================
# Decision Engine
# ============================================================

def decision_engine_action(predicted_tier, score_based_tier, confidence, p_high):
    """
    Convert model output into human-in-the-loop decision action.
    """

    if confidence < FINAL_REVIEW_THRESHOLD:
        return "Human Review - Low Confidence"

    if predicted_tier != score_based_tier:
        return "Human Review - Model/Score Conflict"

    if (predicted_tier != "High") and (p_high >= HIGH_RISK_OVERRIDE_THRESHOLD):
        return "Human Review - Possible High Risk"

    if predicted_tier == "High":
        return "Auto Escalate as High Risk"

    if predicted_tier == "Medium":
        return "Auto Monitor as Medium Risk"

    if predicted_tier == "Low":
        return "Auto Clear as Low Risk"

    return "Human Review - Undefined"


def predict_single(type1, type2, hp, attack, defense, sp_atk, sp_def, speed):
    """
    Full prediction function:
    user input → ML prediction → synthetic score → decision action.
    """

    type2_value = np.nan if type2 == "None" else type2

    input_df = pd.DataFrame([{
        "Type1": type1,
        "Type2": type2_value,
        "HP": float(hp),
        "Attack": float(attack),
        "Defense": float(defense),
        "Sp_Atk": float(sp_atk),
        "Sp_Def": float(sp_def),
        "Speed": float(speed)
    }])

    model_input = input_df[FINAL_FEATURES].copy()

    pred_id = pipeline.predict(model_input)[0]
    proba = pipeline.predict_proba(model_input)[0]

    predicted_tier = ID_TO_CLASS[int(pred_id)]

    p_low = float(proba[0])
    p_medium = float(proba[1])
    p_high = float(proba[2])
    confidence = float(np.max(proba))

    score_info = compute_sustainability_score(input_df)

    action = decision_engine_action(
        predicted_tier=predicted_tier,
        score_based_tier=score_info["Score_Based_Tier"],
        confidence=confidence,
        p_high=p_high
    )

    return {
        "Predicted Tier": predicted_tier,
        "Score-Based Tier": score_info["Score_Based_Tier"],
        "Recommended Action": action,
        "Confidence": confidence,
        "P_Low": p_low,
        "P_Medium": p_medium,
        "P_High": p_high,
        "Synthetic Risk Score": score_info["Sustainability_Risk_Score"],
        "Pressure Index": score_info["Pressure_Index"],
        "Resilience Index": score_info["Resilience_Index"],
        "Adaptability Index": score_info["Adaptability_Index"],
        "Balance Index": score_info["Balance_Index"],
        "Diversity Index": score_info["Diversity_Index"]
    }


# ============================================================
# Scenario Engine
# ============================================================

def apply_scenario(input_df, scenario_name):
    """
    Apply simple synthetic sustainability scenarios.
    """

    scenario_df = input_df.copy()

    if scenario_name == "Baseline":
        return scenario_df

    if scenario_name == "Resource Scarcity":
        scenario_df["HP"] *= 0.90
        scenario_df["Defense"] *= 0.90
        scenario_df["Sp_Def"] *= 0.90

    elif scenario_name == "Rapid Disruption":
        scenario_df["Speed"] *= 1.10
        scenario_df["Defense"] *= 0.92
        scenario_df["Sp_Def"] *= 0.92

    elif scenario_name == "High Intervention Pressure":
        scenario_df["Attack"] *= 1.12
        scenario_df["Sp_Atk"] *= 1.12

    elif scenario_name == "Resilience Investment":
        scenario_df["HP"] *= 1.08
        scenario_df["Defense"] *= 1.08
        scenario_df["Sp_Def"] *= 1.08

    elif scenario_name == "Balanced Adaptation":
        scenario_df["HP"] *= 1.05
        scenario_df["Defense"] *= 1.05
        scenario_df["Sp_Def"] *= 1.05
        scenario_df["Speed"] *= 1.05

    for col in REQUIRED_STAT_COLS:
        scenario_df[col] = scenario_df[col].clip(lower=1, upper=255)

    return scenario_df


# ============================================================
# UI Styling
# ============================================================

st.markdown(
    """
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .small-note {
        color: #8a8f98;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Sidebar Inputs
# ============================================================

st.sidebar.header("Input Profile")

type2_choices = ["None"] + POKEMON_TYPES

default_type1_index = POKEMON_TYPES.index("Dragon") if "Dragon" in POKEMON_TYPES else 0
default_type2_index = type2_choices.index("Flying") if "Flying" in type2_choices else 0

type1 = st.sidebar.selectbox(
    "Primary Type",
    POKEMON_TYPES,
    index=default_type1_index
)

type2 = st.sidebar.selectbox(
    "Secondary Type",
    type2_choices,
    index=default_type2_index
)

hp = st.sidebar.number_input("HP", min_value=1, max_value=255, value=90, step=1)
attack = st.sidebar.number_input("Attack", min_value=1, max_value=255, value=130, step=1)
defense = st.sidebar.number_input("Defense", min_value=1, max_value=255, value=95, step=1)
sp_atk = st.sidebar.number_input("Sp. Attack", min_value=1, max_value=255, value=120, step=1)
sp_def = st.sidebar.number_input("Sp. Defense", min_value=1, max_value=255, value=90, step=1)
speed = st.sidebar.number_input("Speed", min_value=1, max_value=255, value=100, step=1)

current_result = predict_single(
    type1, type2, hp, attack, defense, sp_atk, sp_def, speed
)


# ============================================================
# Header
# ============================================================

st.title("🌱 Pokémon Sustainability Resilience Sandbox")

st.markdown(
    """
This app is a **synthetic sustainability-risk simulation prototype** using Pokémon-style stats.

It demonstrates:

- Sustainability proxy design
- Machine learning risk classification
- Human-in-the-loop decision logic
- Scenario simulation

**Important caution:** This is not empirical ESG scoring or real conservation evidence.
"""
)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Risk Screening",
    "Scenario Simulation",
    "Model Diagnostics",
    "Decision Logic"
])


# ============================================================
# Tab 1: Risk Screening
# ============================================================

with tab1:
    st.subheader("Risk Screening")

    result = current_result

    st.write("### Decision Summary")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.metric("Predicted Tier", result["Predicted Tier"])

    with col2:
        action = result["Recommended Action"]

        if "Human Review" in action:
            st.warning(f"Recommended Action: {action}")
        elif "High Risk" in action:
            st.error(f"Recommended Action: {action}")
        elif "Medium Risk" in action:
            st.info(f"Recommended Action: {action}")
        else:
            st.success(f"Recommended Action: {action}")

    with col3:
        st.metric("Confidence", f'{result["Confidence"]:.4f}')

    col4, col5, col6 = st.columns(3)

    col4.metric("P_Low", f'{result["P_Low"]:.4f}')
    col5.metric("P_Medium", f'{result["P_Medium"]:.4f}')
    col6.metric("P_High", f'{result["P_High"]:.4f}')

    st.write("### Probability Distribution")

    prob_df = pd.DataFrame({
        "Risk Tier": ["Low", "Medium", "High"],
        "Probability": [
            result["P_Low"],
            result["P_Medium"],
            result["P_High"]
        ]
    })

    st.bar_chart(prob_df.set_index("Risk Tier"))

    st.dataframe(
        prob_df.round(4),
        use_container_width=True
    )

    st.write("### Sustainability Proxy Details")

    proxy_output = pd.DataFrame([{
        "Synthetic Risk Score": result["Synthetic Risk Score"],
        "Pressure Index": result["Pressure Index"],
        "Resilience Index": result["Resilience Index"],
        "Adaptability Index": result["Adaptability Index"],
        "Balance Index": result["Balance Index"],
        "Diversity Index": result["Diversity Index"],
        "Score-Based Tier": result["Score-Based Tier"]
    }])

    st.dataframe(
        proxy_output.round(4),
        use_container_width=True
    )


# ============================================================
# Tab 2: Scenario Simulation
# ============================================================

with tab2:
    st.subheader("Scenario Simulation")

    scenario_name = st.selectbox(
        "Select Scenario",
        [
            "Baseline",
            "Resource Scarcity",
            "Rapid Disruption",
            "High Intervention Pressure",
            "Resilience Investment",
            "Balanced Adaptation"
        ]
    )

    base_type2_value = np.nan if type2 == "None" else type2

    base_df = pd.DataFrame([{
        "Type1": type1,
        "Type2": base_type2_value,
        "HP": float(hp),
        "Attack": float(attack),
        "Defense": float(defense),
        "Sp_Atk": float(sp_atk),
        "Sp_Def": float(sp_def),
        "Speed": float(speed)
    }])

    scenario_df = apply_scenario(base_df, scenario_name)

    base_result = predict_single(
        type1, type2, hp, attack, defense, sp_atk, sp_def, speed
    )

    scenario_result = predict_single(
        scenario_df.iloc[0]["Type1"],
        "None" if pd.isna(scenario_df.iloc[0]["Type2"]) else scenario_df.iloc[0]["Type2"],
        scenario_df.iloc[0]["HP"],
        scenario_df.iloc[0]["Attack"],
        scenario_df.iloc[0]["Defense"],
        scenario_df.iloc[0]["Sp_Atk"],
        scenario_df.iloc[0]["Sp_Def"],
        scenario_df.iloc[0]["Speed"]
    )

    comparison = pd.DataFrame([
        {
            "Case": "Baseline",
            "Predicted Tier": base_result["Predicted Tier"],
            "Recommended Action": base_result["Recommended Action"],
            "P_High": base_result["P_High"],
            "Confidence": base_result["Confidence"],
            "Risk Score": base_result["Synthetic Risk Score"],
            "Pressure": base_result["Pressure Index"],
            "Resilience": base_result["Resilience Index"],
            "Balance": base_result["Balance Index"]
        },
        {
            "Case": scenario_name,
            "Predicted Tier": scenario_result["Predicted Tier"],
            "Recommended Action": scenario_result["Recommended Action"],
            "P_High": scenario_result["P_High"],
            "Confidence": scenario_result["Confidence"],
            "Risk Score": scenario_result["Synthetic Risk Score"],
            "Pressure": scenario_result["Pressure Index"],
            "Resilience": scenario_result["Resilience Index"],
            "Balance": scenario_result["Balance Index"]
        }
    ])

    comparison["Risk Score Change"] = (
        comparison["Risk Score"] - comparison.loc[0, "Risk Score"]
    )

    comparison["P_High Change"] = (
        comparison["P_High"] - comparison.loc[0, "P_High"]
    )

    st.dataframe(comparison.round(4), use_container_width=True)

    st.write("### Risk Score and P_High Comparison")
    st.bar_chart(comparison.set_index("Case")[["Risk Score", "P_High"]])


# ============================================================
# Tab 3: Model Diagnostics
# ============================================================

with tab3:
    st.subheader("Model Diagnostics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Holdout Accuracy", f'{metrics["Accuracy"]:.4f}')
    col2.metric("Macro F1", f'{metrics["Macro F1"]:.4f}')
    col3.metric("Recall High", f'{metrics["Recall High"]:.4f}')

    st.write("### Risk Tier Distribution")

    tier_counts = (
        sdf["Sustainability_Risk_Tier"]
        .value_counts()
        .reindex(CLASS_ORDER)
    )

    st.bar_chart(tier_counts)

    st.write("### Confusion Matrix")

    cm_df = pd.DataFrame(
        metrics["Confusion Matrix"],
        index=[f"Actual {c}" for c in CLASS_ORDER],
        columns=[f"Pred {c}" for c in CLASS_ORDER]
    )

    st.dataframe(cm_df, use_container_width=True)

    st.write("### Dataset Preview")

    preview_cols = [
        "Name", "Type1", "Type2", "HP", "Attack", "Defense", "Sp_Atk", "Sp_Def", "Speed",
        "Sustainability_Risk_Score", "Sustainability_Risk_Tier"
    ]

    preview_cols = [c for c in preview_cols if c in sdf.columns]

    st.dataframe(
        sdf[preview_cols].head(30),
        use_container_width=True
    )


# ============================================================
# Tab 4: Decision Logic
# ============================================================

with tab4:
    st.subheader("Decision Engine Logic")

    st.markdown(
        f"""
### Thresholds

- Final Review Threshold: `{FINAL_REVIEW_THRESHOLD}`
- High Risk Override Threshold: `{HIGH_RISK_OVERRIDE_THRESHOLD}`
- Low / Medium Boundary: `{LOW_MEDIUM_BOUNDARY:.4f}`
- Medium / High Boundary: `{MEDIUM_HIGH_BOUNDARY:.4f}`

### Decision Rules

1. Confidence below threshold → Human Review  
2. Model prediction conflicts with score-based tier → Human Review  
3. P_High is high while prediction is not High → Human Review  
4. Predicted High with sufficient confidence → Auto Escalate as High Risk  
5. Predicted Medium with sufficient confidence → Auto Monitor as Medium Risk  
6. Predicted Low with sufficient confidence → Auto Clear as Low Risk  

### Sustainability Proxy Logic

- Pressure Index = Attack + Sp. Attack + Speed  
- Resilience Index = HP + Defense + Sp. Defense  
- Adaptability Index = Speed + Sp. Defense + HP  
- Balance Index = lower stat imbalance means higher balance  
- Diversity Index = dual type indicates broader functional diversity  

### Important Caution

This is a synthetic sustainability simulation.  
It should not be used as real ESG scoring, conservation assessment, or policy evidence.
"""
    )
