import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

# ============================================================
# App Config
# ============================================================

st.set_page_config(
    page_title="Pokémon Sustainability Resilience Sandbox",
    page_icon="🌱",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "pokemon_sustainability_final_pipeline.joblib"
REFERENCE_PATH = BASE_DIR / "simulation_reference.joblib"


# ============================================================
# Load Artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    ref = joblib.load(REFERENCE_PATH)
    return model, ref


model, ref = load_artifacts()

CLASS_ORDER = ref["class_order"]
ID_TO_CLASS = {int(k): v for k, v in ref["id_to_class"].items()}
FINAL_FEATURES = ref["final_features"]
POKEMON_TYPES = ref["pokemon_types"]
REQUIRED_STAT_COLS = ref["required_stat_cols"]

FINAL_REVIEW_THRESHOLD = ref["final_review_threshold"]
HIGH_RISK_OVERRIDE_THRESHOLD = ref["high_risk_override_threshold"]
LOW_MEDIUM_BOUNDARY = ref["low_medium_boundary"]
MEDIUM_HIGH_BOUNDARY = ref["medium_high_boundary"]

stat_scaler_app = ref["stat_scaler_app"]
imbalance_scaler_app = ref["imbalance_scaler_app"]


# ============================================================
# Sustainability Score Engine
# ============================================================

def compute_sustainability_score(input_df):
    temp = input_df.copy()

    for col in REQUIRED_STAT_COLS:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    scaled = pd.DataFrame(
        stat_scaler_app.transform(temp[REQUIRED_STAT_COLS]),
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
            "HP_Scaled",
            "Attack_Scaled",
            "Defense_Scaled",
            "Sp_Atk_Scaled",
            "Sp_Def_Scaled",
            "Speed_Scaled"
        ]
    ].std(axis=1).to_frame("Stat_Imbalance")

    balance_index = 1 - imbalance_scaler_app.transform(stat_imbalance).ravel()
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

    pred_id = model.predict(model_input)[0]
    proba = model.predict_proba(model_input)[0]

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

    output = {
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

    return output


# ============================================================
# Scenario Engine
# ============================================================

def apply_scenario(input_df, scenario_name):
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
# UI
# ============================================================

st.title("🌱 Pokémon Sustainability Resilience Sandbox")

st.markdown(
    """
This is a **synthetic sustainability-risk simulation prototype** using Pokémon-style stats.  
It demonstrates a full data product workflow: proxy design, ML classification, human-in-the-loop decision logic, and scenario simulation.

**Important caution:** This is not empirical ESG scoring or real conservation evidence.
"""
)

tab1, tab2, tab3 = st.tabs([
    "Risk Screening",
    "Scenario Simulation",
    "Decision Logic"
])

type2_choices = ["None"] + POKEMON_TYPES


# ============================================================
# Tab 1: Risk Screening
# ============================================================

with tab1:
    st.subheader("Risk Screening")

    col1, col2 = st.columns(2)

    with col1:
        type1 = st.selectbox("Primary Type", POKEMON_TYPES, index=POKEMON_TYPES.index("Dragon") if "Dragon" in POKEMON_TYPES else 0)
        hp = st.number_input("HP", min_value=1, max_value=255, value=90, step=1)
        defense = st.number_input("Defense", min_value=1, max_value=255, value=95, step=1)
        sp_def = st.number_input("Sp. Defense", min_value=1, max_value=255, value=90, step=1)

    with col2:
        type2 = st.selectbox("Secondary Type", type2_choices, index=type2_choices.index("Flying") if "Flying" in type2_choices else 0)
        attack = st.number_input("Attack", min_value=1, max_value=255, value=130, step=1)
        sp_atk = st.number_input("Sp. Attack", min_value=1, max_value=255, value=120, step=1)
        speed = st.number_input("Speed", min_value=1, max_value=255, value=100, step=1)

    if st.button("Run Risk Screening", type="primary"):
        result = predict_single(type1, type2, hp, attack, defense, sp_atk, sp_def, speed)

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric("Predicted Tier", result["Predicted Tier"])
        metric_col2.metric("Recommended Action", result["Recommended Action"])
        metric_col3.metric("Confidence", f'{result["Confidence"]:.4f}')
        metric_col4.metric("P_High", f'{result["P_High"]:.4f}')

        st.write("### Detailed Output")
        st.dataframe(
            pd.DataFrame([result]).round(4),
            use_container_width=True
        )

        st.write("### Probability Distribution")
        st.bar_chart(
            pd.DataFrame({
                "Probability": [
                    result["P_Low"],
                    result["P_Medium"],
                    result["P_High"]
                ]
            }, index=["Low", "Medium", "High"])
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

    if st.button("Run Scenario Simulation"):
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

        base_result = predict_single(type1, type2, hp, attack, defense, sp_atk, sp_def, speed)

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

        comparison["Risk Score Change"] = comparison["Risk Score"] - comparison.loc[0, "Risk Score"]
        comparison["P_High Change"] = comparison["P_High"] - comparison.loc[0, "P_High"]

        st.dataframe(comparison.round(4), use_container_width=True)

        st.write("### Risk Score Comparison")
        st.bar_chart(comparison.set_index("Case")[["Risk Score", "P_High"]])


# ============================================================
# Tab 3: Decision Logic
# ============================================================

with tab3:
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

### Interpretation

This system is a learning sandbox for ML workflow, decision routing, and synthetic sustainability simulation.
It should not be used as real ESG scoring, conservation assessment, or policy evidence.
"""
    )