# Model Card: Pokémon Sustainability Resilience Sandbox

**Current Version:** `v1.5 Environmental Scenario Connector`  
**Model Type:** Synthetic Sustainability Risk Classification + Decision-Support Prototype  
**Deployment:** Streamlit  
**Primary Use:** Methodological sandbox for sustainability-risk analytics, AI governance, scenario simulation, and portfolio optimization  

---

## 1. Model Overview

The Pokémon Sustainability Resilience Sandbox is a synthetic machine learning and decision-support prototype that transforms Pokémon-style attributes into a sustainability-risk simulation system.

The model uses Pokémon statistics as proxy variables to simulate concepts such as:

- System resilience
- Environmental or operational pressure
- Adaptability
- Balance
- Diversity
- Risk concentration
- Human-review governance
- Scenario sensitivity

The purpose of the model is not to evaluate Pokémon in a game context. Instead, the project uses Pokémon data as a safe and interpretable sandbox for testing how a sustainability-oriented AI decision-support system could be structured.

The model classifies each profile into one of three synthetic sustainability risk tiers:

```text
Low Risk
Medium Risk
High Risk
```

The system then applies a governance layer that determines whether the prediction can be handled automatically or should be routed to human review.

---

## 2. Intended Use

This model is intended for:

- Demonstrating synthetic sustainability-risk modeling
- Testing machine learning deployment in Streamlit
- Exploring human-in-the-loop decision design
- Demonstrating governance monitoring for AI-assisted decisions
- Simulating scenario-based risk changes
- Demonstrating portfolio-level sustainability optimization
- Building a methodological prototype that can later be adapted to real-world ESG, supplier risk, conservation, or sustainability datasets

The project is educational, methodological, and experimental.

---

## 3. Non-Intended Use

This model should not be used for:

- Real ESG scoring
- Investment decisions
- Supplier approval or rejection
- Conservation planning decisions
- Environmental impact assessment
- Biodiversity valuation
- Public policy decisions
- Any real-world risk classification without replacing the synthetic variables with validated real-world indicators

The model is a synthetic proof-of-concept, not an empirical sustainability assessment tool.

---

## 4. Dataset

The base dataset contains Pokémon-style attributes, including:

```text
Name
Type1
Type2
Total
HP
Attack
Defense
Sp_Atk
Sp_Def
Speed
```

The project also includes an enriched dataset created from PokéAPI-style contextual data, including variables such as:

```text
PokeAPI_ID
Base_Experience
Height_dm
Weight_hg
Ability_Count
Hidden_Ability_Count
Move_Count
Generation
Habitat
Growth_Rate
Color
Shape
Rarity_Context_Flag
Evolution_Context_Flag
Resource_Intensity_Index
Ability_Complexity_Index
Enrichment_Context_Score
```

The enriched dataset is currently used for contextual analysis only. It does not replace the main prediction model in the current release.

---

## 5. Variable Interpretation

The model uses Pokémon variables as metaphors for sustainability-system behavior.

| Pokémon Variable | Synthetic Interpretation | Possible Real-World Equivalent |
|---|---|---|
| `Name` | Unit of analysis | Company, supplier, project, site, asset |
| `Type1` | Primary category | Industry, activity type, ecosystem class |
| `Type2` | Secondary category / diversity signal | Multi-sector exposure, hybrid function, ecological diversity |
| `HP` | Absorptive capacity | Financial buffer, carrying capacity, resilience reserve |
| `Attack` | Pressure-generating force | Carbon intensity, resource pressure, operational impact |
| `Defense` | Structural protection | Compliance system, infrastructure robustness, risk controls |
| `Sp_Atk` | Hidden or complex pressure | Supply-chain risk, reputational exposure, indirect impact |
| `Sp_Def` | Adaptive or governance protection | ESG governance, disclosure quality, institutional trust |
| `Speed` | Response agility / volatility | Crisis response, operational agility, change velocity |
| `Total` | Aggregate capacity | Aggregate score, but excluded from the strict model to reduce leakage risk |

The model intentionally avoids using `Total` in the final strict prediction feature set because it may act as an overly aggregated shortcut.

---

## 6. Synthetic Index Construction

The model constructs several synthetic indices.

### Resilience Index

Represents absorptive and defensive capacity.

```text
Resilience Index = mean(HP, Defense, Sp_Def)
```

Synthetic meaning:

- Higher resilience means the system is more capable of absorbing pressure.
- In real-world terms, this could represent operational stability, compliance strength, financial buffer, or governance maturity.

---

### Pressure Index

Represents risk-driving or pressure-generating capacity.

```text
Pressure Index = mean(Attack, Sp_Atk, Speed)
```

Synthetic meaning:

- Higher pressure means the system may generate stronger operational, environmental, or volatility-related stress.
- In real-world terms, this could represent carbon intensity, resource use, production pressure, incident frequency, or operational volatility.

---

### Adaptability Index

Represents response capacity and adaptive flexibility.

```text
Adaptability Index = mean(Speed, Sp_Def, HP)
```

Synthetic meaning:

- Higher adaptability means the system can respond better to changing conditions.
- In real-world terms, this could represent innovation agility, crisis response, supply-chain flexibility, or adaptive capacity.

---

### Balance Index

Represents how balanced the profile is across all six stat dimensions.

Synthetic meaning:

- A balanced profile is less dependent on one extreme strength.
- In real-world terms, this reflects whether an organization or project is balanced across operational, governance, and resilience dimensions.

---

### Diversity Index

Represents whether the profile has one or two types.

```text
Dual Type = 1
Single Type = 0
```

Synthetic meaning:

- Dual-type profiles are interpreted as more functionally diverse.
- In real-world terms, this may represent multi-sector capability, diversified value streams, or ecological/function diversity.

---

## 7. Target Construction

The target variable is synthetic and is not externally observed.

The system first calculates a synthetic sustainability risk score:

```text
Risk Score =
0.40 × Pressure Index
+ 0.30 × (1 - Resilience Index)
+ 0.20 × (1 - Balance Index)
+ 0.10 × (1 - Diversity Index)
```

This score increases when:

- Pressure is high
- Resilience is low
- Balance is low
- Diversity is low

The continuous risk score is then divided into three risk tiers:

```text
Low
Medium
High
```

These tiers become the target labels for the machine learning model.

---

## 8. Model Pipeline

The deployed app uses a train-on-startup design.

When the Streamlit app starts, it:

1. Loads `pokemon_dataset.csv`
2. Cleans column names
3. Constructs synthetic sustainability indices
4. Builds the synthetic target
5. Splits data for holdout diagnostics
6. Trains a machine learning pipeline
7. Re-trains on the full dataset for app-level prediction

The machine learning pipeline includes:

```text
Numeric imputation
Numeric scaling
Categorical imputation
One-hot encoding
HistGradientBoostingClassifier
```

The main input features are:

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

The final strict model does not use the synthetic score, proxy indices, target label, or `Total` as direct model inputs.

---

## 9. Prediction Output

For each profile, the model outputs:

```text
Predicted_Tier
P_Low
P_Medium
P_High
Confidence
Synthetic Risk Score
Score-Based Tier
Recommended Action
```

The model predicts one of:

```text
Low Risk
Medium Risk
High Risk
```

The probability distribution is used by the governance layer to determine whether the case should be automatically handled or sent to human review.

---

## 10. Human-in-the-Loop Decision Layer

The model includes a decision layer after prediction.

Current thresholds:

```text
Final Review Threshold = 0.75
High Risk Override Threshold = 0.35
```

Decision rules:

1. If confidence is below the review threshold, route to Human Review.
2. If model prediction conflicts with score-based tier, route to Human Review.
3. If `P_High` exceeds the high-risk override threshold while the predicted tier is not High, route to Human Review.
4. If predicted High with sufficient confidence, Auto Escalate as High Risk.
5. If predicted Medium with sufficient confidence, Auto Monitor as Medium Risk.
6. If predicted Low with sufficient confidence, Auto Clear as Low Risk.

Possible recommended actions:

```text
Auto Clear as Low Risk
Auto Monitor as Medium Risk
Auto Escalate as High Risk
Human Review - Low Confidence
Human Review - Model/Score Conflict
Human Review - Possible High Risk
```

This layer demonstrates how AI-assisted systems can include oversight rather than relying only on raw model prediction.

---

## 11. Explainability Layer

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
- Key members contributing to risk or resilience

The explainability layer is rule-based. It supports interpretation but should not be interpreted as causal explanation.

---

## 12. Governance Dashboard

The Governance Dashboard monitors AI-assisted decision outcomes.

It summarizes:

```text
Total cases
Auto decision count
Human review count
Auto decision rate
Human review rate
Predicted High Risk rate
Low-confidence cases
Model/score conflicts
Average confidence
Average P_High
Governance priority cases
```

The dashboard can analyze:

```text
Latest Batch Prediction
Base Dataset Simulation
```

This allows users to understand the behavior of the AI system at batch level, not only at single-case level.

---

## 13. Governance Priority Score

The system ranks review-priority cases using:

```text
Governance Priority Score =
0.40 × P_High
+ 0.25 × (1 - Confidence)
+ 0.15 × Human Review Flag
+ 0.10 × Model/Score Conflict Flag
+ 0.10 × Predicted High Risk Flag
```

Higher scores indicate cases that should be reviewed earlier.

This is not a real-world risk score. It is a prioritization score for the synthetic governance dashboard.

---

## 14. Threshold Simulator

The Threshold Simulator allows users to adjust:

```text
Review Threshold
High Risk Override Threshold
```

It recalculates:

```text
Human Review Rate
Auto Decision Rate
Low Confidence Count
Possible High Risk Count
Simulated Recommended Actions
Threshold sensitivity curve
```

This feature demonstrates the trade-off between automation efficiency and human oversight.

A higher review threshold usually increases human review burden.  
A lower high-risk override threshold usually increases review of possible high-risk cases.

---

## 15. Portfolio Optimizer

The Sustainable Portfolio Optimizer randomly generates teams and ranks them using a synthetic portfolio score.

The portfolio score rewards:

```text
Type diversity
Average resilience
Average balance
Average adaptability
Risk control
Low pressure
```

It penalizes:

```text
High-risk concentration
```

Portfolio score formula:

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

The optimizer supports constraints such as:

```text
Team size
Number of random portfolios
Maximum High Risk count
Minimum unique type count
```

This module demonstrates how case-level risk screening can be extended into portfolio-level sustainability design.

---

## 16. Data Enrichment Layer

The app includes a PokéAPI-enriched dataset layer.

Additional contextual variables include:

```text
Base experience
Height
Weight
Ability count
Hidden ability count
Move count
Generation
Habitat
Growth rate
Color
Shape
Rarity context
Evolution context
Resource intensity index
Ability complexity index
Enrichment context score
```

The enrichment layer is currently used for contextual analysis and quality assurance.

It does not change the main model prediction in the current release.

The Data Enrichment tab reports:

```text
Match rate
Unmatched rows
Context index explorer
Top enrichment context scores
Categorical context summary
Unmatched row QA
```

Rows marked as `Unknown / Not available` indicate missing, unavailable, or unmatched enrichment information.

---

## 17. Environmental Scenario Connector

The app includes an Environmental Scenario Connector using weather forecast data.

Users can enter:

```text
Latitude
Longitude
Forecast days
```

The connector retrieves weather variables such as:

```text
Temperature
Precipitation
Wind speed
```

These variables are transformed into synthetic stress scores:

```text
Temperature Stress
Precipitation Stress
Wind Stress
Environmental Stress Score
```

The environmental stress score is then used to adjust the current profile and compare:

```text
Baseline prediction
Weather-adjusted environmental scenario prediction
```

This connector is a demonstration layer. It is not a real climate-impact, ecological-risk, or weather-risk model.

---

## 18. Evaluation Metrics

The app reports global model diagnostics such as:

```text
Holdout Accuracy
Macro F1
Recall for High Risk
Confusion Matrix
Risk Tier Distribution
```

These metrics evaluate model performance against the synthetic target labels.

Because the target itself is synthetic, these metrics should be interpreted as evidence of whether the model learned the constructed proxy logic, not whether it predicts real-world sustainability outcomes.

---

## 19. Real-World Adaptation Potential

The architecture can be adapted to real-world sustainability datasets by replacing Pokémon-style proxy variables with validated empirical indicators.

### Supplier Sustainability Risk Example

| Sandbox Variable | Real-World Supplier Variable |
|---|---|
| `Name` | Supplier name |
| `Type1` / `Type2` | Industry / supplier category |
| `HP` | Operational resilience |
| `Attack` | Environmental pressure |
| `Defense` | Compliance strength |
| `Sp_Atk` | Hidden supply-chain risk |
| `Sp_Def` | Governance and disclosure quality |
| `Speed` | Response agility or volatility |
| `Risk Tier` | Supplier sustainability risk |
| `Human Review` | Audit or compliance review |

---

### Eco-Accommodation / ESG Tourism Example

| Sandbox Variable | Real-World Tourism Variable |
|---|---|
| `Type1` / `Type2` | Accommodation type / tourism segment |
| `HP` | Business resilience |
| `Attack` | Environmental load |
| `Defense` | Environmental management system |
| `Sp_Def` | ESG credibility / certification |
| `Speed` | Response to demand or crisis |
| `Risk Score` | Sustainability positioning risk |
| `Portfolio Score` | Destination or accommodation portfolio quality |

---

### Conservation or Botanical Garden Project Example

| Sandbox Variable | Real-World Conservation Variable |
|---|---|
| `Name` | Project, zone, or program |
| `Type1` / `Type2` | Conservation function / visitor function |
| `HP` | Ecological carrying capacity |
| `Attack` | Visitor or land-use pressure |
| `Defense` | Zoning or physical protection |
| `Sp_Def` | Scientific monitoring and governance |
| `Speed` | Operational response capacity |
| `Environmental Stress` | Weather or climate stress |
| `Portfolio Optimizer` | Balanced project portfolio planning |

---

## 20. Value Translation Potential

The current prototype does not estimate real financial value directly. However, the architecture can be extended into value estimation when connected to real-world data.

Possible value pathways include:

### Risk Reduction Value

```text
Expected Risk Exposure = P_High × Estimated Impact Value
```

```text
Risk Reduction Value =
Expected Loss Before Screening - Expected Loss After Screening
```

### Review Cost Optimization

```text
Review Cost = Number of Human Review Cases × Cost per Review
```

The Threshold Simulator can help evaluate how different thresholds affect review workload and cost.

### Portfolio Value

```text
Portfolio Value =
Base Portfolio Value
+ Sustainability Quality Premium
- Risk Concentration Penalty
```

### WTP-Based Revenue Potential

```text
WTP-Based Revenue Potential =
Number of Customers × Incremental WTP × Booking Probability
```

These formulas are conceptual extensions and require real economic, financial, or survey data before being used in practice.

---

## 21. Limitations

Important limitations:

- The dataset is synthetic for sustainability purposes.
- Pokémon variables are not real ESG or environmental indicators.
- The target label is generated from a synthetic scoring rule.
- The model learns the constructed proxy logic, not real-world sustainability behavior.
- The explainability layer is rule-based, not causal.
- The governance score is for prioritization only.
- The environmental connector is not a validated weather-risk model.
- The enriched dataset may include unmatched or unavailable values.
- The system should not be used for real policy, investment, supplier, conservation, or ESG decisions without validated data and domain review.

---

## 22. Ethical and Interpretive Cautions

The project is designed to demonstrate responsible AI system architecture.

It intentionally includes:

```text
Human review routing
Confidence-based governance
Model/score conflict detection
Threshold simulation
Governance dashboard
Explainability layer
Clear limitations
```

These features reflect the idea that AI decision-support systems should be interpretable, auditable, and governed by human oversight.

---

## 23. Current Release

```text
v1.5 Environmental Scenario Connector
```

Included modules:

```text
Single Risk Screening
Batch Prediction
Governance Dashboard
Threshold Simulator
Data Enrichment Layer
Environmental Scenario Connector
Sustainable Portfolio Optimizer
Scenario Simulation
Model Diagnostics
Decision Logic
CSV Export
Rule-based Explainability
Human-in-the-loop Decision Rules
```

---

## 24. Summary

This model is best understood as a methodological sandbox.

It uses Pokémon-style data as a metaphorical testing ground to demonstrate how an AI sustainability decision-support system can be designed, deployed, explained, governed, and extended.

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
```

With real-world indicators, the same architecture can be adapted into supplier risk screening, ESG decision support, eco-accommodation analysis, conservation project screening, or sustainability portfolio planning.
