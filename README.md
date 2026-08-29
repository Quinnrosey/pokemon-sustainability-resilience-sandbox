# Pokémon Sustainability Resilience Sandbox

**Current Release:** `v1.2 Governance Dashboard`  
**Project Type:** Synthetic Machine Learning + Sustainability Decision-Support Prototype  
**Deployment:** Streamlit App  

---

## Project Overview

**Pokémon Sustainability Resilience Sandbox** is a synthetic machine learning and decision-support prototype that transforms Pokémon-style attributes into a sustainability-risk simulation system.

The project demonstrates how a data product can combine:

- Synthetic sustainability proxy design
- Machine learning risk classification
- Human-in-the-loop decision governance
- Batch risk screening
- Scenario simulation
- Sustainable portfolio optimization
- Rule-based explainability
- Governance monitoring dashboard

This project does **not** claim to measure real ESG performance, biodiversity value, supplier risk, conservation outcomes, or environmental impact. It is a methodological sandbox designed to demonstrate how sustainability-risk analytics and AI-assisted decision support can be structured.

---

## Live Demo

Streamlit App:

https://pokemon-sustainability-resilience-sandbox-27zwyunkra6van7dph4y.streamlit.app/

---

## Release Status

| Version | Release Name | Main Capability |
|---|---|---|
| `v1.0` | Streamlit Prototype | Core app deployment with single prediction, batch prediction, portfolio optimizer, scenario simulation, diagnostics, and decision rules |
| `v1.1` | Explainability Upgrade | Rule-based explanation layer for single prediction and portfolio optimization |
| `v1.2` | Governance Dashboard | Governance monitoring for human review burden, automated decisions, high-risk concentration, and priority cases |

---

## Key Features

### 1. Single Risk Screening

Users can input one Pokémon-style profile and receive:

- Predicted sustainability risk tier
- Probability of Low, Medium, and High risk
- Synthetic sustainability risk score
- Recommended decision action
- Human review flag when confidence is low
- Rule-based explanation of the prediction result

The single screening module is designed to simulate how a decision-support system can classify one case and explain the result in a readable way.

---

### 2. Batch Prediction

Users can upload a CSV file and screen multiple profiles at once.

The system returns:

- Predicted risk tier for every row
- Probability scores for Low, Medium, and High risk
- Confidence score
- Recommended action
- High-risk priority list
- Downloadable prediction results

This module demonstrates how the same risk-screening logic can be scaled from one case to many cases.

---

### 3. Governance Dashboard

The governance dashboard summarizes AI-assisted decision outcomes from either:

- The latest uploaded batch prediction result, or
- The full base dataset simulation

The dashboard reports:

- Total analyzed cases
- Automated decision rate
- Human review rate
- Predicted High Risk rate
- Low-confidence cases
- Model/score conflicts
- Average confidence
- Average probability of High Risk
- Governance priority cases

This feature is designed to support human oversight by identifying which cases may require closer review.

---

### 4. Sustainable Portfolio Optimizer

The optimizer randomly generates Pokémon-style portfolios and ranks them using a synthetic sustainability portfolio score.

The scoring function rewards:

- Type diversity
- Resilience
- Balance
- Adaptability
- Risk control
- Low pressure

The optimizer penalizes excessive concentration of High Risk members.

Users can configure:

- Team size
- Number of random portfolios
- Maximum High Risk count
- Minimum unique type count

The system then returns:

- Best portfolio summary
- Best portfolio members
- Top ranked portfolios
- Portfolio score curve by rank
- Downloadable portfolio results
- Rule-based portfolio explanation

---

### 5. Scenario Simulation

The app can simulate how the selected profile changes under hypothetical conditions.

Available scenarios include:

- Resource Scarcity
- Rapid Disruption
- High Intervention Pressure
- Resilience Investment
- Balanced Adaptation

The scenario module helps demonstrate how a risk profile may shift under changing assumptions.

---

### 6. Model Diagnostics

The app reports global model performance, including:

- Holdout accuracy
- Macro F1
- Recall for High Risk
- Confusion matrix
- Risk tier distribution
- Dataset preview

These diagnostics describe the overall trained model and do not necessarily change when the sidebar input changes.

---

### 7. Decision Logic

The system includes human-in-the-loop decision rules based on:

- Prediction confidence
- Probability of High Risk
- Conflict between model prediction and score-based tier
- Review threshold
- High-risk override threshold

The decision logic prevents the app from relying only on raw model prediction. It adds a governance layer that routes uncertain or potentially risky cases to human review.

---

## Dataset

The project uses a Pokémon-style dataset containing attributes such as:

- `Name`
- `Type1`
- `Type2`
- `HP`
- `Attack`
- `Defense`
- `Sp_Atk`
- `Sp_Def`
- `Speed`

These variables are reinterpreted as synthetic indicators for a sustainability-risk simulation.

This reinterpretation is conceptual and experimental. It is not intended to represent real sustainability, ESG, biodiversity, or ecological measurements.

---

## Synthetic Sustainability Logic

The system constructs several synthetic indices from the dataset.

### Resilience Index

Represents absorptive and defensive capacity.

Main inputs:

- `HP`
- `Defense`
- `Sp_Def`

---

### Pressure Index

Represents disruptive, pressure-related, or risk-driving capacity.

Main inputs:

- `Attack`
- `Sp_Atk`
- `Speed`

---

### Adaptability Index

Represents response capability and adaptive capacity.

Main inputs:

- `Speed`
- `Sp_Def`
- `HP`

---

### Balance Index

Represents how balanced the profile is across all six stat dimensions.

A more balanced profile receives a higher Balance Index.

---

### Diversity Index

Represents whether the profile has one or two types.

- Dual type = higher diversity
- Single type = lower diversity

---

## Synthetic Risk Score

The synthetic sustainability risk score is calculated as:

```text
Risk Score =
0.40 × Pressure Index
+ 0.30 × (1 - Resilience Index)
+ 0.20 × (1 - Balance Index)
+ 0.10 × (1 - Diversity Index)
```

Higher scores indicate higher synthetic sustainability risk within this simulation framework.

Risk tiers are divided into:

- Low
- Medium
- High

The risk tiers are generated from the synthetic risk score and are used as the target labels for the machine learning model.

---

## Machine Learning Model

The app uses a train-on-startup design. When the Streamlit app starts, it loads the dataset and trains the model inside the app environment.

This design avoids model serialization issues between different environments such as Colab, local Python, and Streamlit Cloud.

The machine learning pipeline includes:

- Numeric imputation
- Numeric scaling
- Categorical imputation
- One-hot encoding
- HistGradientBoostingClassifier

The model predicts the synthetic sustainability risk tier:

```text
Low / Medium / High
```

---

## Human-in-the-Loop Decision Rules

The app does not rely only on the predicted class. It applies a decision layer after prediction.

Main rules:

1. If confidence is below the review threshold, send to Human Review.
2. If model prediction conflicts with score-based tier, send to Human Review.
3. If `P_High` is high even when the predicted tier is not High, send to Human Review.
4. If predicted High with sufficient confidence, Auto Escalate as High Risk.
5. If predicted Medium with sufficient confidence, Auto Monitor as Medium Risk.
6. If predicted Low with sufficient confidence, Auto Clear as Low Risk.

Current thresholds:

```text
Final Review Threshold = 0.75
High Risk Override Threshold = 0.35
```

These thresholds are used for demonstration and can be adjusted in future versions.

---

## Explainability Layer

The app includes a rule-based explainability layer to help users interpret model outputs.

For single prediction, the explanation describes:

- Predicted risk tier
- Confidence
- Synthetic risk score
- Main risk drivers
- Probability distribution
- Human-in-the-loop decision reason

For portfolio optimization, the explanation describes:

- Why the selected portfolio ranks highly
- Type diversity
- High-risk concentration
- Average resilience
- Average balance
- Average adaptability
- Average pressure
- Key members that contribute to pressure, resilience, balance, and risk

The explainability layer is rule-based. It is designed for interpretation support and should not be interpreted as causal explanation.

---

## Governance Dashboard

The Governance Dashboard monitors AI-assisted decision outcomes.

It summarizes:

- Total analyzed cases
- Automated decision count
- Human review count
- Automated decision rate
- Human review rate
- Predicted High Risk rate
- Low-confidence count
- Model/score conflict count
- Average confidence
- Average probability of High Risk
- Governance priority cases

The dashboard can analyze either the latest batch prediction result or the base dataset simulation.

---

## Governance Priority Score

Priority cases are ranked using a governance priority score.

```text
Governance Priority Score =
0.40 × P_High
+ 0.25 × (1 - Confidence)
+ 0.15 × Human Review Flag
+ 0.10 × Model/Score Conflict Flag
+ 0.10 × Predicted High Risk Flag
```

Higher scores indicate cases that should be reviewed earlier.

This score is used only for review prioritization within the synthetic simulation. It should not be interpreted as a real-world risk score.

---

## Sustainable Portfolio Optimizer Method

The Sustainable Portfolio Optimizer evaluates randomly generated teams.

Each portfolio receives a score based on:

```text
Portfolio Score =
0.20 × Type Diversity
+ 0.20 × Average Resilience
+ 0.15 × Average Balance
+ 0.15 × Average Adaptability
+ 0.20 × Risk Control
+ 0.10 × Low Pressure
- High Risk Concentration Penalty
```

The optimizer can apply constraints such as:

- Maximum number of High Risk members
- Minimum number of unique types
- Team size
- Number of random portfolios evaluated

This module demonstrates how risk classification can be extended into portfolio-level design and optimization.

---

## Project Structure

```text
streamlit_app.py
requirements.txt
pokemon_dataset.csv
README.md
```

---

## Requirements

```text
streamlit
pandas
numpy
scikit-learn
altair
```

---

## How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## How to Use the App

### Single Risk Screening

1. Adjust the input profile in the sidebar.
2. Open the Risk Screening tab.
3. Review predicted tier, probability distribution, decision action, and explanation.

### Batch Prediction

1. Open the Batch Prediction tab.
2. Download the CSV template.
3. Fill or modify the template.
4. Upload the CSV file.
5. Review full results and high-risk priority cases.
6. Download prediction results.

### Governance Dashboard

1. Open the Governance Dashboard tab.
2. Select either Base Dataset Simulation or Latest Batch Prediction.
3. Review decision outcomes, review burden, high-risk rate, and priority cases.
4. Download governance outputs.

### Portfolio Optimizer

1. Open the Portfolio Optimizer tab.
2. Select team size and optimization constraints.
3. Run the optimizer.
4. Review best portfolio, ranked portfolios, and explanation.
5. Download portfolio results.

### Scenario Simulation

1. Adjust the profile in the sidebar.
2. Open the Scenario Simulation tab.
3. Select a scenario.
4. Compare baseline and scenario results.

---

## Intended Use

This project is intended for:

- Demonstrating synthetic sustainability-risk modeling
- Practicing machine learning app deployment
- Exploring human-in-the-loop decision rules
- Demonstrating governance monitoring for AI-assisted decisions
- Building a portfolio-ready data product prototype

---

## Limitations

This project is a synthetic simulation prototype.

Important limitations:

- Pokémon-style attributes are not real ESG indicators.
- The risk tiers are generated from a synthetic scoring rule.
- Model performance reflects the synthetic target, not real-world sustainability evidence.
- The explainability layer is rule-based and not causal.
- The governance score is for prioritization inside the simulation only.
- Results should not be used for policy, investment, supplier evaluation, conservation planning, or real ESG scoring.
- The project is intended for methodological demonstration and educational use.

---

## Disclaimer

This app is not an empirical ESG scoring system, biodiversity assessment tool, supplier evaluation model, investment tool, or conservation decision system.

It is a sandbox prototype for demonstrating how machine learning, proxy design, scenario simulation, batch screening, portfolio optimization, explainability, and governance monitoring can be combined in a sustainability-oriented data product.

This project is not affiliated with Pokémon, Nintendo, Game Freak, or The Pokémon Company. The dataset is used only for educational and methodological demonstration.

---

## Current Version

```text
v1.2 Governance Dashboard
```

Included modules:

- Single Risk Screening
- Batch Prediction
- Governance Dashboard
- Sustainable Portfolio Optimizer
- Scenario Simulation
- Model Diagnostics
- Decision Logic
- CSV Download
- Human-in-the-loop Decision Rules
- Rule-based Single Prediction Explainability
- Rule-based Portfolio Explainability
- Governance Priority Case Ranking

---

## Future Development

Potential future upgrades include:

- Adjustable governance thresholds
- Batch-level explainability summary
- Portfolio comparison dashboard
- Local model export option
- Alternative model comparison
- Real sustainability dataset adaptation
- Supplier sustainability risk template
- ESG-oriented version of the sandbox
