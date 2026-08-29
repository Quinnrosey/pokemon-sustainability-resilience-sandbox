
# Pokémon Sustainability Resilience Sandbox

## Project Type
Synthetic sustainability simulation and machine learning workflow prototype.

## Purpose
This project uses Pokémon-style tabular data to simulate a sustainability risk-screening workflow.

## Important Caution
This is not empirical sustainability evidence.  
The sustainability indicators are synthetic proxy variables created for educational and workflow testing purposes.

## Final Model
HistGradientBoostingClassifier with strict input design excluding Total.

## Target
Sustainability_Risk_Tier:
- Low
- Medium
- High

## Decision Engine
The system combines:
- ML prediction
- Prediction confidence
- Probability of High Risk
- Score-based policy tier
- Human review routing

## Thresholds
- Final review threshold: 0.75
- High-risk override threshold: 0.35
- Low/Medium boundary: 0.4284875429501764
- Medium/High boundary: 0.48599213457874607

## Recommended Use
Use this as a sandbox for:
- ML workflow learning
- Sustainability proxy design
- Human-in-the-loop decision logic
- Scenario simulation
- Portfolio optimization
- Lightweight deployment through Gradio

## Not Recommended For
- Real environmental assessment
- Real ESG scoring
- Real conservation policy
- Real supplier screening
