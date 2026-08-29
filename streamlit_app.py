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
        ).tolist(),
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
# Batch Prediction Engine
# ============================================================

def clean_batch_columns(input_df):
    """
    Clean uploaded CSV column names to match the model schema.
    Example:
    Sp. Atk -> Sp_Atk
    Sp. Def -> Sp_Def
    """
    temp = input_df.copy()

    temp.columns = (
        temp.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
    )

    return temp


def validate_batch_schema(input_df):
    """
    Validate uploaded batch CSV schema.
    Name is optional.
    Type2 is optional and will be treated as single-type if missing.
    """
    required_batch_cols = [
        "Type1",
        "HP",
        "Attack",
        "Defense",
        "Sp_Atk",
        "Sp_Def",
        "Speed"
    ]

    missing_cols = [
        col for col in required_batch_cols
        if col not in input_df.columns
    ]

    if missing_cols:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_cols)
        )

    return True


def compute_batch_sustainability_scores(input_df):
    """
    Compute synthetic sustainability proxy scores for multiple rows.
    """
    temp = input_df.copy()

    if "Type2" not in temp.columns:
        temp["Type2"] = np.nan

    temp["Type2"] = temp["Type2"].replace("", np.nan)
    temp["Type2"] = temp["Type2"].replace("None", np.nan)

    for col in REQUIRED_STAT_COLS:
        temp[col] = pd.to_numeric(temp[col], errors="coerce")

    for col in REQUIRED_STAT_COLS:
        if temp[col].isna().sum() > 0:
            temp[col] = temp[col].fillna(sdf[col].median())

    scaled = pd.DataFrame(
        stat_scaler.transform(temp[REQUIRED_STAT_COLS]),
        columns=[f"{c}_Scaled" for c in REQUIRED_STAT_COLS],
        index=temp.index
    ).clip(0, 1)

    temp["Resilience_Index"] = scaled[
        ["HP_Scaled", "Defense_Scaled", "Sp_Def_Scaled"]
    ].mean(axis=1)

    temp["Pressure_Index"] = scaled[
        ["Attack_Scaled", "Sp_Atk_Scaled", "Speed_Scaled"]
    ].mean(axis=1)

    temp["Adaptability_Index"] = scaled[
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

    temp["Balance_Index"] = 1 - imbalance_scaler.transform(
        stat_imbalance
    ).ravel()

    temp["Balance_Index"] = temp["Balance_Index"].clip(0, 1)

    temp["Diversity_Index"] = temp["Type2"].notna().astype(int)

    temp["Sustainability_Risk_Score"] = (
        0.40 * temp["Pressure_Index"] +
        0.30 * (1 - temp["Resilience_Index"]) +
        0.20 * (1 - temp["Balance_Index"]) +
        0.10 * (1 - temp["Diversity_Index"])
    )

    temp["Score_Based_Tier"] = pd.cut(
        temp["Sustainability_Risk_Score"],
        bins=[-np.inf, LOW_MEDIUM_BOUNDARY, MEDIUM_HIGH_BOUNDARY, np.inf],
        labels=CLASS_ORDER,
        include_lowest=True
    ).astype(str)

    return temp


def decision_engine_action_batch(predicted_tier, score_based_tier, confidence, p_high):
    """
    Human-in-the-loop decision rule for each batch row.
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


def predict_batch(batch_df):
    """
    Full batch prediction pipeline:
    uploaded CSV
    → schema validation
    → sustainability proxy scores
    → ML prediction
    → probabilities
    → recommended action
    """
    batch = clean_batch_columns(batch_df)

    if "Name" not in batch.columns:
        batch["Name"] = [f"Batch_Item_{i+1:04d}" for i in range(len(batch))]

    if "Type2" not in batch.columns:
        batch["Type2"] = np.nan

    validate_batch_schema(batch)

    scored_batch = compute_batch_sustainability_scores(batch)

    model_input = scored_batch[FINAL_FEATURES].copy()

    pred_id = pipeline.predict(model_input)
    proba = pipeline.predict_proba(model_input)

    scored_batch["Predicted_Tier"] = [
        ID_TO_CLASS[int(i)] for i in pred_id
    ]

    scored_batch["P_Low"] = proba[:, 0]
    scored_batch["P_Medium"] = proba[:, 1]
    scored_batch["P_High"] = proba[:, 2]
    scored_batch["Confidence"] = proba.max(axis=1)

    scored_batch["Recommended_Action"] = scored_batch.apply(
        lambda row: decision_engine_action_batch(
            predicted_tier=row["Predicted_Tier"],
            score_based_tier=row["Score_Based_Tier"],
            confidence=row["Confidence"],
            p_high=row["P_High"]
        ),
        axis=1
    )

    output_cols = [
        "Name",
        "Type1",
        "Type2",
        "HP",
        "Attack",
        "Defense",
        "Sp_Atk",
        "Sp_Def",
        "Speed",
        "Predicted_Tier",
        "Score_Based_Tier",
        "P_Low",
        "P_Medium",
        "P_High",
        "Confidence",
        "Recommended_Action",
        "Sustainability_Risk_Score",
        "Pressure_Index",
        "Resilience_Index",
        "Adaptability_Index",
        "Balance_Index",
        "Diversity_Index"
    ]

    output_cols = [c for c in output_cols if c in scored_batch.columns]

    return scored_batch[output_cols].copy()


def create_batch_template():
    """
    Create downloadable CSV template for batch prediction.
    """
    template = pd.DataFrame([
        {
            "Name": "Example_Dragon_Flying",
            "Type1": "Dragon",
            "Type2": "Flying",
            "HP": 90,
            "Attack": 130,
            "Defense": 95,
            "Sp_Atk": 120,
            "Sp_Def": 90,
            "Speed": 100
        },
        {
            "Name": "Example_Water_None",
            "Type1": "Water",
            "Type2": "",
            "HP": 70,
            "Attack": 80,
            "Defense": 85,
            "Sp_Atk": 70,
            "Sp_Def": 75,
            "Speed": 60
        },
        {
            "Name": "Example_Electric_None",
            "Type1": "Electric",
            "Type2": "",
            "HP": 45,
            "Attack": 55,
            "Defense": 40,
            "Sp_Atk": 50,
            "Sp_Def": 50,
            "Speed": 90
        }
    ])

    return template


# ============================================================
# Portfolio Optimizer Engine
# ============================================================

def prepare_portfolio_pool(sdf):
    """
    Prepare candidate pool for sustainable portfolio optimization.
    """

    pool = sdf.copy().reset_index(drop=True)

    required_cols = [
        "Name",
        "Type1",
        "Type2",
        "HP",
        "Attack",
        "Defense",
        "Sp_Atk",
        "Sp_Def",
        "Speed",
        "Sustainability_Risk_Tier",
        "Sustainability_Risk_Score",
        "Resilience_Index",
        "Pressure_Index",
        "Adaptability_Index",
        "Balance_Index",
        "Diversity_Index"
    ]

    missing_cols = [c for c in required_cols if c not in pool.columns]

    if missing_cols:
        raise ValueError(
            "Portfolio pool is missing required columns: "
            + ", ".join(missing_cols)
        )

    pool = pool[required_cols].copy()

    pool["Portfolio_ID"] = [
        f"P{i:04d}" for i in range(1, len(pool) + 1)
    ]

    pool["Type2"] = pool["Type2"].replace("", np.nan)
    pool["Sustainability_Risk_Tier"] = pool["Sustainability_Risk_Tier"].astype(str)

    return pool


def evaluate_portfolio(team_df, team_size=6):
    """
    Evaluate one selected team or portfolio.
    Higher Portfolio_Score means a more sustainable and balanced portfolio.
    """

    team = team_df.copy()

    type_values = []

    for _, row in team.iterrows():
        if pd.notna(row["Type1"]):
            type_values.append(str(row["Type1"]))

        if pd.notna(row["Type2"]) and str(row["Type2"]).strip() != "":
            type_values.append(str(row["Type2"]))

    unique_types = sorted(list(set(type_values)))
    unique_type_count = len(unique_types)

    max_possible_types = team_size * 2
    type_diversity_score = unique_type_count / max_possible_types

    avg_resilience = team["Resilience_Index"].mean()
    avg_balance = team["Balance_Index"].mean()
    avg_adaptability = team["Adaptability_Index"].mean()
    avg_pressure = team["Pressure_Index"].mean()
    avg_risk_score = team["Sustainability_Risk_Score"].mean()

    low_pressure_score = 1 - avg_pressure
    risk_control_score = 1 - avg_risk_score

    risk_counts = (
        team["Sustainability_Risk_Tier"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )

    high_risk_share = risk_counts["High"] / team_size
    medium_risk_share = risk_counts["Medium"] / team_size
    low_risk_share = risk_counts["Low"] / team_size

    raw_score = (
        0.20 * type_diversity_score +
        0.20 * avg_resilience +
        0.15 * avg_balance +
        0.15 * avg_adaptability +
        0.20 * risk_control_score +
        0.10 * low_pressure_score
    )

    high_risk_penalty = 0.10 * high_risk_share

    portfolio_score = raw_score - high_risk_penalty
    portfolio_score = float(np.clip(portfolio_score, 0, 1))

    result = {
        "Portfolio_Score": portfolio_score,
        "Raw_Score": float(raw_score),
        "High_Risk_Penalty": float(high_risk_penalty),
        "Type_Diversity_Score": float(type_diversity_score),
        "Unique_Type_Count": int(unique_type_count),
        "Unique_Types": ", ".join(unique_types),
        "Avg_Resilience": float(avg_resilience),
        "Avg_Balance": float(avg_balance),
        "Avg_Adaptability": float(avg_adaptability),
        "Avg_Pressure": float(avg_pressure),
        "Avg_Risk_Score": float(avg_risk_score),
        "Risk_Control_Score": float(risk_control_score),
        "Low_Pressure_Score": float(low_pressure_score),
        "Low_Risk_Count": int(risk_counts["Low"]),
        "Medium_Risk_Count": int(risk_counts["Medium"]),
        "High_Risk_Count": int(risk_counts["High"]),
        "Low_Risk_Share": float(low_risk_share),
        "Medium_Risk_Share": float(medium_risk_share),
        "High_Risk_Share": float(high_risk_share),
        "Team_Member_IDs": " | ".join(team["Portfolio_ID"].astype(str).tolist()),
        "Team_Members": " | ".join(team["Name"].astype(str).tolist())
    }

    return result


def random_portfolio_optimizer(
    pool,
    team_size=6,
    n_iterations=5000,
    max_high_risk_count=None,
    min_unique_type_count=None,
    random_state=42
):
    """
    Random sustainable portfolio optimizer.

    The optimizer randomly generates teams, scores each team,
    applies optional governance constraints, and returns ranked results.
    """

    if len(pool) < team_size:
        raise ValueError("Candidate pool is smaller than team size.")

    rng = np.random.default_rng(random_state)
    results = []

    pool_indices = pool.index.to_numpy()

    for i in range(n_iterations):
        selected_indices = rng.choice(
            pool_indices,
            size=team_size,
            replace=False
        )

        team_df = pool.loc[selected_indices].copy()

        evaluated = evaluate_portfolio(
            team_df=team_df,
            team_size=team_size
        )

        if max_high_risk_count is not None:
            if evaluated["High_Risk_Count"] > max_high_risk_count:
                continue

        if min_unique_type_count is not None:
            if evaluated["Unique_Type_Count"] < min_unique_type_count:
                continue

        evaluated["Iteration"] = i + 1
        results.append(evaluated)

    if len(results) == 0:
        raise ValueError(
            "No portfolios passed the selected constraints. "
            "Please relax the High Risk or Type Diversity constraints."
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=[
            "Portfolio_Score",
            "Type_Diversity_Score",
            "Avg_Resilience",
            "Avg_Balance",
            "Avg_Risk_Score"
        ],
        ascending=[False, False, False, False, True]
    ).reset_index(drop=True)

    results_df["Rank"] = np.arange(1, len(results_df) + 1)

    front_cols = [
        "Rank",
        "Portfolio_Score",
        "Raw_Score",
        "High_Risk_Penalty",
        "Type_Diversity_Score",
        "Unique_Type_Count",
        "Avg_Resilience",
        "Avg_Balance",
        "Avg_Adaptability",
        "Avg_Pressure",
        "Avg_Risk_Score",
        "Low_Risk_Count",
        "Medium_Risk_Count",
        "High_Risk_Count",
        "Unique_Types",
        "Team_Members",
        "Team_Member_IDs"
    ]

    remaining_cols = [
        c for c in results_df.columns
        if c not in front_cols
    ]

    results_df = results_df[front_cols + remaining_cols]

    return results_df


def get_portfolio_member_details(pool, team_member_ids):
    """
    Get detailed member table for a selected portfolio.
    """

    if isinstance(team_member_ids, str):
        selected_ids = [
            x.strip()
            for x in team_member_ids.split("|")
        ]
    else:
        selected_ids = list(team_member_ids)

    selected_team = pool[
        pool["Portfolio_ID"].isin(selected_ids)
    ].copy()

    detail_cols = [
        "Portfolio_ID",
        "Name",
        "Type1",
        "Type2",
        "Sustainability_Risk_Tier",
        "Sustainability_Risk_Score",
        "Resilience_Index",
        "Pressure_Index",
        "Adaptability_Index",
        "Balance_Index",
        "Diversity_Index",
        "HP",
        "Attack",
        "Defense",
        "Sp_Atk",
        "Sp_Def",
        "Speed"
    ]

    detail_cols = [
        c for c in detail_cols
        if c in selected_team.columns
    ]

    return selected_team[detail_cols]


# ============================================================
# Explainability Engine
# ============================================================

def describe_level(value, low_threshold=0.40, high_threshold=0.60):
    """
    Convert a numeric index into a readable level.
    """
    if value >= high_threshold:
        return "high"
    elif value >= low_threshold:
        return "moderate"
    else:
        return "low"


def explain_single_prediction(result):
    """
    Generate readable explanation for one prediction result.
    """

    predicted_tier = result["Predicted Tier"]
    score_based_tier = result["Score-Based Tier"]
    action = result["Recommended Action"]

    confidence = result["Confidence"]
    p_low = result["P_Low"]
    p_medium = result["P_Medium"]
    p_high = result["P_High"]

    risk_score = result["Synthetic Risk Score"]
    pressure = result["Pressure Index"]
    resilience = result["Resilience Index"]
    adaptability = result["Adaptability Index"]
    balance = result["Balance Index"]
    diversity = result["Diversity Index"]

    pressure_level = describe_level(pressure)
    resilience_level = describe_level(resilience)
    adaptability_level = describe_level(adaptability)
    balance_level = describe_level(balance)
    risk_level = describe_level(risk_score)

    explanation_lines = []

    explanation_lines.append(
        f"The current profile is classified as **{predicted_tier} Risk** "
        f"with a confidence score of **{confidence:.4f}**."
    )

    explanation_lines.append(
        f"The synthetic risk score is **{risk_score:.4f}**, which is considered **{risk_level}** "
        f"within this simulation framework."
    )

    # Main risk drivers
    driver_points = []

    if pressure >= 0.60:
        driver_points.append(
            f"Pressure Index is high (**{pressure:.4f}**), meaning the profile has strong pressure-related characteristics."
        )
    elif pressure >= 0.40:
        driver_points.append(
            f"Pressure Index is moderate (**{pressure:.4f}**), meaning pressure-related characteristics are present but not extreme."
        )
    else:
        driver_points.append(
            f"Pressure Index is low (**{pressure:.4f}**), which helps reduce the overall risk score."
        )

    if resilience < 0.40:
        driver_points.append(
            f"Resilience Index is low (**{resilience:.4f}**), which increases vulnerability in the synthetic risk logic."
        )
    elif resilience < 0.60:
        driver_points.append(
            f"Resilience Index is moderate (**{resilience:.4f}**), providing partial risk absorption."
        )
    else:
        driver_points.append(
            f"Resilience Index is high (**{resilience:.4f}**), which helps offset risk pressure."
        )

    if balance < 0.40:
        driver_points.append(
            f"Balance Index is low (**{balance:.4f}**), suggesting the profile is uneven across stat dimensions."
        )
    elif balance < 0.60:
        driver_points.append(
            f"Balance Index is moderate (**{balance:.4f}**), suggesting a partially balanced profile."
        )
    else:
        driver_points.append(
            f"Balance Index is high (**{balance:.4f}**), suggesting a relatively balanced profile."
        )

    if adaptability >= 0.60:
        driver_points.append(
            f"Adaptability Index is high (**{adaptability:.4f}**), which improves response capacity."
        )
    elif adaptability >= 0.40:
        driver_points.append(
            f"Adaptability Index is moderate (**{adaptability:.4f}**), giving some adaptive capacity."
        )
    else:
        driver_points.append(
            f"Adaptability Index is low (**{adaptability:.4f}**), which limits adaptive capacity."
        )

    if diversity >= 1:
        driver_points.append(
            "Diversity Index is present because the profile has dual-type characteristics."
        )
    else:
        driver_points.append(
            "Diversity Index is absent because the profile has only one type."
        )

    # Decision explanation
    if "Low Confidence" in action:
        decision_text = (
            f"The case is routed to **Human Review** because confidence "
            f"(**{confidence:.4f}**) is below the review threshold "
            f"(**{FINAL_REVIEW_THRESHOLD:.2f}**)."
        )
    elif "Model/Score Conflict" in action:
        decision_text = (
            f"The case is routed to **Human Review** because the model prediction "
            f"(**{predicted_tier}**) conflicts with the score-based tier "
            f"(**{score_based_tier}**)."
        )
    elif "Possible High Risk" in action:
        decision_text = (
            f"The case is routed to **Human Review** because the probability of High Risk "
            f"(**{p_high:.4f}**) exceeds the override threshold "
            f"(**{HIGH_RISK_OVERRIDE_THRESHOLD:.2f}**), even though the final prediction is not High."
        )
    elif "Auto Escalate" in action:
        decision_text = (
            "The case is automatically escalated because it is classified as High Risk "
            "with sufficient confidence."
        )
    elif "Auto Monitor" in action:
        decision_text = (
            "The case is automatically monitored because it is classified as Medium Risk "
            "with sufficient confidence."
        )
    elif "Auto Clear" in action:
        decision_text = (
            "The case is automatically cleared because it is classified as Low Risk "
            "with sufficient confidence."
        )
    else:
        decision_text = (
            "The case is routed to review because the decision rule returned an undefined status."
        )

    probability_text = (
        f"Probability distribution: Low = **{p_low:.4f}**, "
        f"Medium = **{p_medium:.4f}**, High = **{p_high:.4f}**."
    )

    explanation = {
        "summary": explanation_lines,
        "drivers": driver_points,
        "probability": probability_text,
        "decision": decision_text
    }

    return explanation


def render_single_explanation(result):
    """
    Render explanation block for Streamlit.
    """

    explanation = explain_single_prediction(result)

    st.write("### Explainability Summary")

    for line in explanation["summary"]:
        st.markdown(line)

    st.write("#### Main Drivers")

    for point in explanation["drivers"]:
        st.markdown(f"- {point}")

    st.write("#### Probability Interpretation")
    st.markdown(explanation["probability"])

    st.write("#### Decision Rule Interpretation")
    st.markdown(explanation["decision"])


def explain_portfolio_result(best_portfolio, team_details):
    """
    Generate readable explanation for the selected best portfolio.
    """

    team = team_details.copy()

    portfolio_score = best_portfolio["Portfolio_Score"]
    unique_type_count = int(best_portfolio["Unique_Type_Count"])
    high_risk_count = int(best_portfolio["High_Risk_Count"])
    avg_risk_score = best_portfolio["Avg_Risk_Score"]
    avg_resilience = best_portfolio["Avg_Resilience"]
    avg_balance = best_portfolio["Avg_Balance"]
    avg_adaptability = best_portfolio["Avg_Adaptability"]
    avg_pressure = best_portfolio["Avg_Pressure"]

    team_size = len(team)

    risk_mix = (
        team["Sustainability_Risk_Tier"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
    )

    pressure_leader = team.sort_values(
        "Pressure_Index",
        ascending=False
    ).iloc[0]

    resilience_leader = team.sort_values(
        "Resilience_Index",
        ascending=False
    ).iloc[0]

    balance_leader = team.sort_values(
        "Balance_Index",
        ascending=False
    ).iloc[0]

    risk_leader = team.sort_values(
        "Sustainability_Risk_Score",
        ascending=False
    ).iloc[0]

    explanation_lines = []

    explanation_lines.append(
        f"The selected portfolio achieved a score of **{portfolio_score:.4f}** "
        f"with **{unique_type_count} unique types** across **{team_size} members**."
    )

    explanation_lines.append(
        f"The portfolio contains **{high_risk_count} High Risk member(s)**, "
        f"with an average synthetic risk score of **{avg_risk_score:.4f}**."
    )

    if unique_type_count >= team_size + 3:
        diversity_text = (
            "Type diversity is strong, which improves the portfolio score because the team is not concentrated in a narrow type structure."
        )
    elif unique_type_count >= team_size:
        diversity_text = (
            "Type diversity is moderate to strong, providing a reasonably diversified portfolio."
        )
    else:
        diversity_text = (
            "Type diversity is limited, which may reduce the overall portfolio quality."
        )

    if high_risk_count == 0:
        risk_text = (
            "Risk concentration is low because the portfolio does not include any High Risk member."
        )
    elif high_risk_count == 1:
        risk_text = (
            "Risk concentration is controlled because only one member is classified as High Risk."
        )
    else:
        risk_text = (
            "Risk concentration should be reviewed because multiple members are classified as High Risk."
        )

    balance_text = (
        f"The average resilience is **{avg_resilience:.4f}**, "
        f"average balance is **{avg_balance:.4f}**, "
        f"average adaptability is **{avg_adaptability:.4f}**, "
        f"and average pressure is **{avg_pressure:.4f}**."
    )

    member_text = [
        f"Highest pressure member: **{pressure_leader['Name']}** "
        f"with Pressure Index **{pressure_leader['Pressure_Index']:.4f}**.",
        f"Most resilient member: **{resilience_leader['Name']}** "
        f"with Resilience Index **{resilience_leader['Resilience_Index']:.4f}**.",
        f"Most balanced member: **{balance_leader['Name']}** "
        f"with Balance Index **{balance_leader['Balance_Index']:.4f}**.",
        f"Highest synthetic risk member: **{risk_leader['Name']}** "
        f"with Risk Score **{risk_leader['Sustainability_Risk_Score']:.4f}**."
    ]

    risk_mix_text = (
        f"Risk mix: Low = **{int(risk_mix['Low'])}**, "
        f"Medium = **{int(risk_mix['Medium'])}**, "
        f"High = **{int(risk_mix['High'])}**."
    )

    explanation = {
        "summary": explanation_lines,
        "diversity": diversity_text,
        "risk": risk_text,
        "balance": balance_text,
        "members": member_text,
        "risk_mix": risk_mix_text
    }

    return explanation


def render_portfolio_explanation(best_portfolio, team_details):
    """
    Render portfolio explanation block for Streamlit.
    """

    explanation = explain_portfolio_result(
        best_portfolio=best_portfolio,
        team_details=team_details
    )

    st.write("### Portfolio Explainability Summary")

    for line in explanation["summary"]:
        st.markdown(line)

    st.write("#### Why this portfolio ranks highly")

    st.markdown(f"- {explanation['diversity']}")
    st.markdown(f"- {explanation['risk']}")
    st.markdown(f"- {explanation['balance']}")
    st.markdown(f"- {explanation['risk_mix']}")

    st.write("#### Key Members")

    for point in explanation["members"]:
        st.markdown(f"- {point}")


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

tab1, tab_batch, tab_portfolio, tab2, tab3, tab4 = st.tabs([
    "Risk Screening",
    "Batch Prediction",
    "Portfolio Optimizer",
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

    import altair as alt

    prob_chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X(
                "Risk Tier:N",
                sort=["Low", "Medium", "High"],
                title="Risk Tier"
            ),
            y=alt.Y(
                "Probability:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1])
            ),
            tooltip=["Risk Tier", alt.Tooltip("Probability:Q", format=".4f")]
        )
        .properties(height=350)
    )

    st.altair_chart(prob_chart, use_container_width=True)

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


    st.divider()

    render_single_explanation(result)


# ============================================================
# Tab: Batch Prediction
# ============================================================

with tab_batch:
    st.subheader("Batch Prediction")

    st.markdown(
        """
Upload a CSV file to run sustainability-risk screening for multiple Pokémon-style profiles.

Required columns:

- `Type1`
- `HP`
- `Attack`
- `Defense`
- `Sp_Atk`
- `Sp_Def`
- `Speed`

Optional columns:

- `Name`
- `Type2`
        """
    )

    template_df = create_batch_template()

    template_csv = template_df.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig")

    st.download_button(
        label="Download Batch CSV Template",
        data=template_csv,
        file_name="pokemon_batch_prediction_template.csv",
        mime="text/csv"
    )

    uploaded_batch_file = st.file_uploader(
        "Upload batch CSV",
        type=["csv"]
    )

    if uploaded_batch_file is not None:
        try:
            uploaded_batch_df = pd.read_csv(uploaded_batch_file)

            st.write("### Uploaded Data Preview")
            st.dataframe(
                uploaded_batch_df.head(20),
                use_container_width=True
            )

            batch_results = predict_batch(uploaded_batch_df)

            st.success(
                f"Batch prediction completed for {len(batch_results)} rows."
            )

            # ------------------------------------------------
            # Summary Metrics
            # ------------------------------------------------

            tier_summary = (
                batch_results["Predicted_Tier"]
                .value_counts()
                .reindex(CLASS_ORDER, fill_value=0)
            )

            action_summary = (
                batch_results["Recommended_Action"]
                .value_counts()
            )

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            metric_col1.metric(
                "Total Rows",
                len(batch_results)
            )

            metric_col2.metric(
                "Predicted High",
                int(tier_summary["High"])
            )

            metric_col3.metric(
                "Human Review",
                int(
                    batch_results["Recommended_Action"]
                    .str.contains("Human Review")
                    .sum()
                )
            )

            metric_col4.metric(
                "Average P_High",
                f'{batch_results["P_High"].mean():.4f}'
            )

            # ------------------------------------------------
            # Tier Distribution
            # ------------------------------------------------

            st.write("### Predicted Risk Tier Distribution")

            tier_summary_df = pd.DataFrame({
                "Risk Tier": CLASS_ORDER,
                "Count": tier_summary.values
            })

            st.bar_chart(
                tier_summary_df.set_index("Risk Tier")
            )

            st.dataframe(
                tier_summary_df,
                use_container_width=True
            )

            # ------------------------------------------------
            # Recommended Action Distribution
            # ------------------------------------------------

            st.write("### Recommended Action Distribution")

            action_summary_df = action_summary.reset_index()
            action_summary_df.columns = ["Recommended Action", "Count"]

            st.dataframe(
                action_summary_df,
                use_container_width=True
            )

            # ------------------------------------------------
            # High-Risk Priority List
            # ------------------------------------------------

            st.write("### High-Risk Priority List")

            high_priority_cases = batch_results[
                (
                    batch_results["Predicted_Tier"] == "High"
                ) |
                (
                    batch_results["P_High"] >= HIGH_RISK_OVERRIDE_THRESHOLD
                )
            ].copy()

            high_priority_cases = high_priority_cases.sort_values(
                ["P_High", "Confidence", "Sustainability_Risk_Score"],
                ascending=False
            )

            st.dataframe(
                high_priority_cases.round(4),
                use_container_width=True
            )

            # ------------------------------------------------
            # Full Results
            # ------------------------------------------------

            st.write("### Full Batch Prediction Results")

            st.dataframe(
                batch_results.round(4),
                use_container_width=True
            )

            output_csv = batch_results.to_csv(
                index=False,
                encoding="utf-8-sig"
            ).encode("utf-8-sig")

            st.download_button(
                label="Download Batch Prediction Results",
                data=output_csv,
                file_name="pokemon_batch_prediction_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Batch prediction failed.")
            st.exception(e)


# ============================================================
# Tab: Portfolio Optimizer
# ============================================================

with tab_portfolio:
    st.subheader("Sustainable Portfolio Optimizer")

    st.markdown(
        """
This tool randomly generates Pokémon-style portfolios and ranks them using a synthetic sustainability score.

The optimizer rewards:

- Type diversity
- Resilience
- Balance
- Adaptability
- Risk control
- Low pressure

It penalizes portfolios with excessive High Risk concentration.
        """
    )

    portfolio_pool = prepare_portfolio_pool(sdf)

    st.write("### Optimizer Settings")

    setting_col1, setting_col2, setting_col3, setting_col4 = st.columns(4)

    with setting_col1:
        team_size = st.slider(
            "Team Size",
            min_value=3,
            max_value=6,
            value=6,
            step=1
        )

    with setting_col2:
        n_iterations = st.slider(
            "Random Portfolios",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000
        )

    with setting_col3:
        max_high_risk_option = st.selectbox(
            "Max High Risk Count",
            options=[
                "No constraint",
                0,
                1,
                2,
                3
            ],
            index=0
        )

        if max_high_risk_option == "No constraint":
            max_high_risk_count = None
        else:
            max_high_risk_count = int(max_high_risk_option)

    with setting_col4:
        min_unique_type_count = st.slider(
            "Min Unique Types",
            min_value=0,
            max_value=team_size * 2,
            value=min(8, team_size * 2),
            step=1
        )

        if min_unique_type_count == 0:
            min_unique_type_count_value = None
        else:
            min_unique_type_count_value = int(min_unique_type_count)

    st.caption(
        "Higher iterations may produce better portfolios but will take longer to run."
    )

    run_optimizer = st.button(
        "Run Portfolio Optimizer",
        type="primary"
    )

    if run_optimizer:
        try:
            with st.spinner("Searching for sustainable portfolios..."):
                portfolio_results = random_portfolio_optimizer(
                    pool=portfolio_pool,
                    team_size=team_size,
                    n_iterations=n_iterations,
                    max_high_risk_count=max_high_risk_count,
                    min_unique_type_count=min_unique_type_count_value,
                    random_state=RANDOM_STATE
                )

            st.session_state["portfolio_results"] = portfolio_results
            st.session_state["portfolio_pool"] = portfolio_pool

            st.success(
                f"Optimization completed. {len(portfolio_results)} portfolios passed the constraints."
            )

        except Exception as e:
            st.error("Portfolio optimization failed.")
            st.exception(e)

    if "portfolio_results" in st.session_state:
        portfolio_results = st.session_state["portfolio_results"]
        portfolio_pool = st.session_state["portfolio_pool"]

        st.write("### Best Portfolio Summary")

        best = portfolio_results.iloc[0]

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric(
            "Best Portfolio Score",
            f'{best["Portfolio_Score"]:.4f}'
        )

        metric_col2.metric(
            "Unique Types",
            int(best["Unique_Type_Count"])
        )

        metric_col3.metric(
            "High Risk Count",
            int(best["High_Risk_Count"])
        )

        metric_col4.metric(
            "Avg Risk Score",
            f'{best["Avg_Risk_Score"]:.4f}'
        )
        
        if best["High_Risk_Count"] == 0:
            risk_message = "low high-risk concentration"
        elif best["High_Risk_Count"] == 1:
            risk_message = "controlled high-risk concentration"
        else:
            risk_message = "noticeable high-risk concentration"

        st.info(
            f'The best portfolio achieved a score of {best["Portfolio_Score"]:.4f}, '
            f'with {int(best["Unique_Type_Count"])} unique types and '
            f'{int(best["High_Risk_Count"])} high-risk member(s). '
            f'This suggests a portfolio with {risk_message}, '
            'designed to balance diversity, resilience, adaptability, and risk control.'
        )

        st.write("### Best Portfolio Members")

        best_team_details = get_portfolio_member_details(
            portfolio_pool,
            best["Team_Member_IDs"]
        )

        st.dataframe(
            best_team_details.round(4),
            use_container_width=True,
            hide_index=True
        )

                st.divider()

        render_portfolio_explanation(
            best_portfolio=best,
            team_details=best_team_details
        )

        st.write("### Top Portfolio Rankings")

        display_cols = [
            "Rank",
            "Portfolio_Score",
            "Type_Diversity_Score",
            "Unique_Type_Count",
            "Avg_Resilience",
            "Avg_Balance",
            "Avg_Adaptability",
            "Avg_Pressure",
            "Avg_Risk_Score",
            "Low_Risk_Count",
            "Medium_Risk_Count",
            "High_Risk_Count",
            "Unique_Types",
            "Team_Members"
        ]

        display_cols = [
            c for c in display_cols
            if c in portfolio_results.columns
        ]

        st.dataframe(
            portfolio_results[display_cols].head(50).round(4),
            use_container_width=True,
            hide_index=True
        )

        st.write("### Portfolio Score by Rank")

        st.caption(
            "This line chart shows the top-ranked portfolios ordered by portfolio score. "
            "It is not a histogram; the curve should gradually decline as rank increases."
        )

        score_distribution = portfolio_results[
            ["Rank", "Portfolio_Score"]
        ].copy()

        score_distribution = score_distribution.head(100)

        st.line_chart(
            score_distribution.set_index("Rank")
        )

        st.write("### Inspect a Ranked Portfolio")

        max_rank_to_select = min(50, len(portfolio_results))

        selected_rank = st.selectbox(
            "Select portfolio rank",
            options=list(range(1, max_rank_to_select + 1)),
            index=0
        )

        selected_portfolio = portfolio_results[
            portfolio_results["Rank"] == selected_rank
        ].iloc[0]

        selected_team_details = get_portfolio_member_details(
            portfolio_pool,
            selected_portfolio["Team_Member_IDs"]
        )

        selected_col1, selected_col2, selected_col3, selected_col4 = st.columns(4)

        selected_col1.metric(
            "Selected Score",
            f'{selected_portfolio["Portfolio_Score"]:.4f}'
        )

        selected_col2.metric(
            "Unique Types",
            int(selected_portfolio["Unique_Type_Count"])
        )

        selected_col3.metric(
            "High Risk Count",
            int(selected_portfolio["High_Risk_Count"])
        )

        selected_col4.metric(
            "Avg Pressure",
            f'{selected_portfolio["Avg_Pressure"]:.4f}'
        )

        st.dataframe(
            selected_team_details.round(4),
            use_container_width=True,
            hide_index=True
        )

        st.write("### Download Portfolio Results")

        portfolio_csv = portfolio_results.to_csv(
            index=False,
            encoding="utf-8-sig"
        ).encode("utf-8-sig")

        st.download_button(
            label="Download Top Portfolio Results",
            data=portfolio_csv,
            file_name="pokemon_top_sustainable_portfolios.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "Set optimizer parameters and click 'Run Portfolio Optimizer' to generate ranked sustainable portfolios."
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

    st.info(
        "This page shows global model diagnostics. "
        "The model performance metrics below are calculated from the holdout test set, "
        "so they do not change when the sidebar input changes."
    )

    st.write("## Global Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Holdout Accuracy", f'{metrics["Accuracy"]:.4f}')
    col2.metric("Macro F1", f'{metrics["Macro F1"]:.4f}')
    col3.metric("Recall High", f'{metrics["Recall High"]:.4f}')

    st.caption(
        "These metrics describe the overall trained model, not the current sidebar input."
    )

    st.write("## Current Input Diagnostics")

    current_result = predict_single(
        type1,
        type2,
        hp,
        attack,
        defense,
        sp_atk,
        sp_def,
        speed
    )

    input_diag_cols = st.columns(4)

    input_diag_cols[0].metric(
        "Current Predicted Tier",
        current_result["Predicted Tier"]
    )

    input_diag_cols[1].metric(
        "Current Confidence",
        f'{current_result["Confidence"]:.4f}'
    )

    input_diag_cols[2].metric(
        "Current P_High",
        f'{current_result["P_High"]:.4f}'
    )

    input_diag_cols[3].metric(
        "Current Risk Score",
        f'{current_result["Synthetic Risk Score"]:.4f}'
    )

    current_proxy_df = pd.DataFrame([{
        "Pressure Index": current_result["Pressure Index"],
        "Resilience Index": current_result["Resilience Index"],
        "Adaptability Index": current_result["Adaptability Index"],
        "Balance Index": current_result["Balance Index"],
        "Diversity Index": current_result["Diversity Index"],
        "Score-Based Tier": current_result["Score-Based Tier"],
        "Recommended Action": current_result["Recommended Action"]
    }])

    st.dataframe(
        current_proxy_df.round(4),
        use_container_width=True
    )

    st.write("## Risk Tier Distribution")

    tier_counts = (
        sdf["Sustainability_Risk_Tier"]
        .value_counts()
        .reindex(CLASS_ORDER)
    )

    tier_counts_df = pd.DataFrame({
        "Risk Tier": CLASS_ORDER,
        "Count": tier_counts.values
    })

    try:
        import altair as alt

        tier_chart = (
            alt.Chart(tier_counts_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Risk Tier:N",
                    sort=["Low", "Medium", "High"],
                    title="Risk Tier"
                ),
                y=alt.Y(
                    "Count:Q",
                    title="Number of Pokémon"
                ),
                tooltip=["Risk Tier", "Count"]
            )
            .properties(height=350)
        )

        st.altair_chart(tier_chart, use_container_width=True)

    except Exception:
        st.bar_chart(tier_counts_df.set_index("Risk Tier"))

    st.write("## Confusion Matrix")

    cm_df = pd.DataFrame(
        metrics["Confusion Matrix"],
        index=["Actual Low", "Actual Medium", "Actual High"],
        columns=["Pred Low", "Pred Medium", "Pred High"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True
    )

    st.caption(
        "The confusion matrix is calculated from the holdout test set. "
        "It is used to inspect where the model confuses Low, Medium, and High risk tiers."
    )

    st.write("## Dataset Preview")

    preview_cols = [
        "Name",
        "Type1",
        "Type2",
        "HP",
        "Attack",
        "Defense",
        "Sp_Atk",
        "Sp_Def",
        "Speed"
    ]

    preview_cols = [c for c in preview_cols if c in sdf.columns]

    st.dataframe(
        sdf[preview_cols].head(20),
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
