# Pokémon Sustainability Resilience Sandbox 
| v1.0 Streamlit Prototype
v1.1 Explainability Upgrade

## Project Overview

Pokémon Sustainability Resilience Sandbox is a synthetic machine learning and decision-support prototype that transforms Pokémon-style attributes into a sustainability-risk simulation system.

The project is designed as a data product prototype for demonstrating:

- Synthetic sustainability proxy design
- Machine learning risk classification
- Human-in-the-loop decision governance
- Batch risk screening
- Scenario simulation
- Sustainable portfolio optimization

This project does not claim to measure real ESG performance, biodiversity value, or conservation outcomes. It is a methodological sandbox for demonstrating how sustainability-risk analytics and AI-assisted decision support can be designed.

---

## Live Demo

Streamlit App:

> Paste your Streamlit app link here

---

## Key Features

### 1. Single Risk Screening

Users can input one Pokémon-style profile and receive:

- Predicted sustainability risk tier
- Probability of Low, Medium, and High risk
- Synthetic sustainability risk score
- Recommended decision action
- Human review flag when confidence is low

### 2. Batch Prediction

Users can upload a CSV file and screen multiple profiles at once.

The system returns:

- Predicted risk tier for every row
- Probability scores
- Confidence
- Recommended action
- High-risk priority list
- Downloadable prediction results

### 3. Sustainable Portfolio Optimizer

The optimizer randomly generates Pokémon-style portfolios and ranks them using a synthetic sustainability score.

The scoring function rewards:

- Type diversity
- Resilience
- Balance
- Adaptability
- Risk control
- Low pressure

The optimizer penalizes excessive concentration of High Risk members.

### 4. Scenario Simulation

The app can simulate changes under different hypothetical conditions, such as:

- Resource scarcity
- Rapid disruption
- High intervention pressure
- Resilience investment
- Balanced adaptation

### 5. Model Diagnostics

The app reports global model performance, including:

- Holdout accuracy
- Macro F1
- Recall for High Risk
- Confusion matrix
- Risk tier distribution

### 6. Decision Logic

The system includes human-in-the-loop decision rules based on:

- Prediction confidence
- Probability of High Risk
- Conflict between model prediction and score-based tier
- Review threshold
- High-risk override threshold

---

## Dataset

The project uses a Pokémon-style dataset containing attributes such as:

- Type1
- Type2
- HP
- Attack
- Defense
- Sp. Attack
- Sp. Defense
- Speed

These variables are reinterpreted as synthetic indicators for a sustainability-risk simulation.

---

## Synthetic Sustainability Logic

The system constructs several synthetic indices.

### Resilience Index

Represents absorptive and defensive capacity.

Main inputs:

- HP
- Defense
- Sp. Defense

### Pressure Index

Represents disruptive or pressure-related capacity.

Main inputs:

- Attack
- Sp. Attack
- Speed

### Adaptability Index

Represents response capability and adaptive capacity.

Main inputs:

- Speed
- Sp. Defense
- HP

### Balance Index

Represents how balanced the profile is across all six stat dimensions.

### Diversity Index

Represents whether the profile has one or two types.

---

## Synthetic Risk Score

The synthetic sustainability risk score is calculated as:

```text
Risk Score =
0.40 × Pressure Index
+ 0.30 × (1 - Resilience Index)
+ 0.20 × (1 - Balance Index)
+ 0.10 × (1 - Diversity Index)
