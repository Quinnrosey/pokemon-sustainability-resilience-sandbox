# 🌱 Pokémon Sustainability Resilience Sandbox  
### AI-Powered Synthetic Sustainability Risk & Decision-Support Prototype

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://pokemon-sustainability-resilience-sandbox-27zwyunkra6van7dph4y.streamlit.app/)
![Version](https://img.shields.io/badge/Version-v1.5%20Environmental%20Scenario%20Connector-blue)
![Python](https://img.shields.io/badge/Python-3.x-yellow?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

---

## 🚀 Live Demo

👉 **Launch the Streamlit App here:**  
### [🌱 Pokémon Sustainability Resilience Sandbox](https://pokemon-sustainability-resilience-sandbox-27zwyunkra6van7dph4y.streamlit.app/)

---

## 📌 Project Overview

**Pokémon Sustainability Resilience Sandbox** is a synthetic machine learning and decision-support prototype that transforms Pokémon-style attributes into a sustainability-risk simulation system.

The project uses Pokémon data as a **metaphorical sandbox** to demonstrate how sustainability-risk analytics, AI governance, scenario simulation, and portfolio optimization can be designed before being adapted to real-world datasets.

This is not a Pokémon battle model.  
It is a **methodological prototype** that asks:

> Can we use familiar structured data to prototype a responsible AI system for sustainability risk, governance, and decision support?

---

## 🧠 Core Idea

Pokémon stats are reinterpreted as proxy variables for sustainability-system behavior.

For example:

| Pokémon Attribute | Metaphorical Meaning | Possible Real-World Equivalent |
|---|---|---|
| `HP` | Absorptive capacity | Financial buffer, carrying capacity, resilience reserve |
| `Attack` | Pressure-generating force | Carbon intensity, resource pressure, operational impact |
| `Defense` | Structural protection | Compliance system, infrastructure robustness, risk controls |
| `Sp_Atk` | Hidden or complex pressure | Supply-chain risk, indirect exposure, reputational risk |
| `Sp_Def` | Adaptive or governance protection | ESG governance, disclosure quality, institutional trust |
| `Speed` | Response agility | Crisis response, innovation speed, operational volatility |
| `Type1 / Type2` | Functional category and diversity | Industry, supplier category, ecosystem function |
| `Name` | Unit of analysis | Company, supplier, project, site, asset |

The result is a synthetic sustainability-risk engine that classifies each profile into:

```text
Low Risk
Medium Risk
High Risk
```

---

## 🧩 What This Project Demonstrates

This project demonstrates an end-to-end AI sustainability data product prototype with:

- 🧪 Synthetic sustainability proxy design  
- 🤖 Machine learning risk classification  
- 🧠 Rule-based explainability  
- 👥 Human-in-the-loop decision governance  
- 📊 Batch risk screening  
- 🛡️ Governance dashboard  
- 🎚️ Threshold simulator  
- 🌦️ Environmental scenario connector  
- 🧬 PokéAPI data enrichment layer  
- 🧭 Sustainable portfolio optimizer  
- 📥 CSV export for downstream analysis  

---

## 🏗️ System Architecture

```text
Data Layer
    ↓
Synthetic Proxy Engineering
    ↓
Machine Learning Risk Classification
    ↓
Human-in-the-loop Decision Layer
    ↓
Explainability Layer
    ↓
Governance Dashboard
    ↓
Threshold Simulator
    ↓
Scenario Simulation
    ↓
Portfolio Optimization
    ↓
External Data Enrichment + Environmental Connector
```

---

## ✨ Key Features

### 1. 🔍 Single Risk Screening

Users can input one Pokémon-style profile and receive:

- Predicted sustainability risk tier
- Probability of Low, Medium, and High risk
- Synthetic sustainability risk score
- Recommended decision action
- Human review flag
- Explainability summary

---

### 2. 📂 Batch Prediction

Users can upload a CSV file and screen multiple profiles at once.

The system returns:

- Predicted risk tier for every row
- Probability scores
- Confidence score
- Recommended action
- High-risk priority list
- Downloadable prediction results

---

### 3. 🛡️ Governance Dashboard

The governance dashboard summarizes AI-assisted decision outcomes.

It reports:

- Total analyzed cases
- Automated decision rate
- Human review rate
- Predicted High Risk rate
- Low-confidence cases
- Model/score conflicts
- Average confidence
- Average probability of High Risk
- Governance priority cases

This helps evaluate whether the AI system is making too many automated decisions or routing enough cases to human oversight.

---

### 4. 🎚️ Threshold Simulator

The threshold simulator allows users to adjust:

- Review Threshold
- High Risk Override Threshold

It recalculates:

- Human Review Rate
- Auto Decision Rate
- Low Confidence Count
- Possible High Risk Count
- Simulated Recommended Actions
- Threshold sensitivity curve

This feature demonstrates the trade-off between:

```text
Automation efficiency
vs
Human oversight and risk control
```

---

### 5. 🧬 Data Enrichment Layer

The app includes a PokéAPI-enriched dataset layer.

Additional contextual variables include:

- PokéAPI ID
- Base experience
- Height and weight
- Ability count
- Hidden ability count
- Move count
- Generation
- Habitat
- Growth rate
- Rarity context
- Evolution context
- Resource intensity index
- Ability complexity index
- Enrichment context score

This layer is used for contextual analysis and future model expansion.  
It does not replace the main prediction model in the current release.

---

### 6. 🌦️ Environmental Scenario Connector

The app connects to weather forecast data and converts weather variables into a synthetic environmental stress score.

Users can enter:

- Latitude
- Longitude
- Forecast days

The connector uses weather variables such as:

- Temperature
- Precipitation
- Wind speed

It then compares:

```text
Baseline Prediction
vs
Weather-Adjusted Environmental Scenario Prediction
```

This feature demonstrates how external environmental data can be connected to scenario-based risk simulation.

---

### 7. 🧭 Sustainable Portfolio Optimizer

The optimizer randomly generates Pokémon-style portfolios and ranks them using a synthetic sustainability portfolio score.

The scoring function rewards:

- Type diversity
- Resilience
- Balance
- Adaptability
- Risk control
- Low pressure

It penalizes excessive High Risk concentration.

Users can configure:

- Team size
- Number of random portfolios
- Maximum High Risk count
- Minimum unique type count

---

### 8. 📈 Model Diagnostics

The app reports global model performance, including:

- Holdout accuracy
- Macro F1
- Recall for High Risk
- Confusion matrix
- Risk tier distribution
- Dataset preview

These diagnostics evaluate how well the model learns the synthetic risk logic.

---

## 🧮 Synthetic Sustainability Logic

The model constructs several synthetic indices.

### 🛡️ Resilience Index

Represents absorptive and defensive capacity.

```text
Resilience Index = mean(HP, Defense, Sp_Def)
```

---

### ⚡ Pressure Index

Represents pressure-generating or risk-driving force.

```text
Pressure Index = mean(Attack, Sp_Atk, Speed)
```

---

### 🔄 Adaptability Index

Represents response capability and adaptive flexibility.

```text
Adaptability Index = mean(Speed, Sp_Def, HP)
```

---

### ⚖️ Balance Index

Represents how balanced the profile is across all six stat dimensions.

A highly uneven profile receives a lower balance score.

---

### 🌈 Diversity Index

Represents whether the profile has one or two types.

```text
Dual Type = 1
Single Type = 0
```

---

## 🔥 Synthetic Risk Score

The synthetic sustainability risk score is calculated as:

```text
Risk Score =
0.40 × Pressure Index
+ 0.30 × (1 - Resilience Index)
+ 0.20 × (1 - Balance Index)
+ 0.10 × (1 - Diversity Index)
```

Higher scores indicate higher synthetic sustainability risk.

Risk tiers are divided into:

```text
Low
Medium
High
```

---

## 🤖 Machine Learning Model

The deployed app uses a **train-on-startup design**.

When the Streamlit app starts, it:

1. Loads `pokemon_dataset.csv`
2. Cleans column names
3. Constructs synthetic sustainability indices
4. Builds synthetic target labels
5. Trains a machine learning pipeline
6. Reports diagnostics
7. Re-trains on the full dataset for app-level prediction

The machine learning pipeline includes:

```text
Numeric imputation
Numeric scaling
Categorical imputation
One-hot encoding
HistGradientBoostingClassifier
```

Main input features:

```text
Type1
Type2
HP
Attack
Defense
Sp_Atk
Sp_Def
Speed
```

The strict model does not directly use:

```text
Total
Synthetic Risk Score
Proxy indices
Target label
```

This reduces shortcut learning and leakage risk.

---

## 👥 Human-in-the-Loop Decision Logic

The system does not rely only on raw model prediction.

It applies a decision layer using:

```text
Final Review Threshold = 0.75
High Risk Override Threshold = 0.35
```

Decision rules:

1. If confidence is below threshold → Human Review  
2. If model prediction conflicts with score-based tier → Human Review  
3. If P_High is high while prediction is not High → Human Review  
4. Predicted High with sufficient confidence → Auto Escalate  
5. Predicted Medium with sufficient confidence → Auto Monitor  
6. Predicted Low with sufficient confidence → Auto Clear  

Possible actions:

```text
Auto Clear as Low Risk
Auto Monitor as Medium Risk
Auto Escalate as High Risk
Human Review - Low Confidence
Human Review - Model/Score Conflict
Human Review - Possible High Risk
```

---

## 🧠 Explainability Layer

The app includes a rule-based explainability layer.

For single prediction, it explains:

- Predicted risk tier
- Confidence
- Synthetic risk score
- Main risk drivers
- Probability distribution
- Human-review decision reason

For portfolio optimization, it explains:

- Why a selected portfolio ranks highly
- Type diversity
- High-risk concentration
- Average resilience
- Average balance
- Average adaptability
- Average pressure
- Key members contributing to pressure, resilience, balance, and risk

---

## 🛡️ Governance Priority Score

Priority cases are ranked using:

```text
Governance Priority Score =
0.40 × P_High
+ 0.25 × (1 - Confidence)
+ 0.15 × Human Review Flag
+ 0.10 × Model/Score Conflict Flag
+ 0.10 × Predicted High Risk Flag
```

Higher scores indicate cases that should be reviewed earlier.

This score is used only for review prioritization inside the synthetic simulation.

---

## 🧭 Portfolio Optimizer Formula

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

This shows how case-level risk classification can be extended into portfolio-level sustainability design.

---

## 🌍 Real-World Adaptation Potential

Although this project uses Pokémon data, the system architecture can be adapted to real-world sustainability datasets.

### 🏭 Supplier Sustainability Risk

| Sandbox Variable | Real-World Supplier Variable |
|---|---|
| `Name` | Supplier name |
| `Type1 / Type2` | Industry / supplier category |
| `HP` | Operational resilience |
| `Attack` | Environmental pressure |
| `Defense` | Compliance strength |
| `Sp_Atk` | Hidden supply-chain risk |
| `Sp_Def` | Governance and disclosure quality |
| `Speed` | Response agility or volatility |
| `Risk Tier` | Supplier sustainability risk |
| `Human Review` | Audit or compliance review |

---

## 💰 Value Translation Potential

The current prototype does not estimate real financial value directly. However, the architecture can be extended into value estimation when connected to real-world data.

### Risk Reduction Value

```text
Expected Risk Exposure = P_High × Estimated Impact Value
```

```text
Risk Reduction Value =
Expected Loss Before Screening - Expected Loss After Screening
```

---

### Review Cost Optimization

```text
Review Cost = Number of Human Review Cases × Cost per Review
```

The Threshold Simulator can help evaluate how different thresholds affect review workload and cost.

---

### Portfolio Value

```text
Portfolio Value =
Base Portfolio Value
+ Sustainability Quality Premium
- Risk Concentration Penalty
```

---

### WTP-Based Revenue Potential

```text
WTP-Based Revenue Potential =
Number of Customers × Incremental WTP × Booking Probability
```

These formulas are conceptual extensions and require real economic, financial, or survey data before use.

---

## 📁 Project Structure

```text
streamlit_app.py
requirements.txt
pokemon_dataset.csv
pokemon_enriched_dataset.csv
README.md
MODEL_CARD.md
```

---

## 🧰 Requirements

```text
streamlit
pandas
numpy
scikit-learn
altair
requests
```

---

## 🖥️ How to Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

---

## 📚 Model Documentation

For detailed model explanation, including target construction, proxy variables, decision rules, governance logic, limitations, and real-world adaptation potential, see:

👉 [MODEL_CARD.md](MODEL_CARD.md)

---

## ⚠️ Limitations

This project is a synthetic simulation prototype.

Important limitations:

- Pokémon-style attributes are not real ESG indicators.
- The risk tiers are generated from a synthetic scoring rule.
- Model performance reflects the synthetic target, not real-world sustainability evidence.
- The explainability layer is rule-based and not causal.
- The governance score is for prioritization only.
- The environmental connector is not a validated weather-risk model.
- The enriched dataset may include unmatched or unavailable values.
- Results should not be used for policy, investment, supplier evaluation, conservation planning, or real ESG scoring.

---

## 📌 Disclaimer

This app is not an empirical ESG scoring system, biodiversity assessment tool, supplier evaluation model, investment tool, or conservation decision system.

It is a sandbox prototype for demonstrating how machine learning, proxy design, scenario simulation, batch screening, portfolio optimization, explainability, and governance monitoring can be combined in a sustainability-oriented data product.

This project is not affiliated with Pokémon, Nintendo, Game Freak, or The Pokémon Company. The dataset is used only for educational and methodological demonstration.

---

## 🏷️ Current Release

```text
v1.5 Environmental Scenario Connector
```

Included modules:

- 🔍 Single Risk Screening
- 📂 Batch Prediction
- 🛡️ Governance Dashboard
- 🎚️ Threshold Simulator
- 🧬 Data Enrichment Layer
- 🌦️ Environmental Scenario Connector
- 🧭 Sustainable Portfolio Optimizer
- 📈 Scenario Simulation
- 🧪 Model Diagnostics
- 👥 Human-in-the-loop Decision Logic
- 🧠 Rule-based Explainability
- 📥 CSV Export

---

## 🌟 Summary

**Pokémon Sustainability Resilience Sandbox** uses Pokémon-style data as a metaphorical testing ground to demonstrate how an AI sustainability decision-support system can be designed, deployed, explained, governed, and extended.

The project shows that a sustainability-oriented AI system should not stop at prediction. It should also include:

```text
Proxy design
Risk classification
Human oversight
Explainability
Governance monitoring
Threshold control
Scenario simulation
Portfolio optimization
External data enrichment
Environmental scenario connection
```

With real-world indicators, the same architecture can be adapted into supplier risk screening, ESG decision support, eco-accommodation analysis, conservation project screening, or sustainability portfolio planning.
