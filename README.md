# Injury Risk Intelligence Dashboard

An interactive ML dashboard that predicts athlete injury risk and provides actionable recommendations. Combines predictive modeling, simulation, and decision support in a unified interface for data-driven athlete management.

---

## Core Capabilities

**Injury Risk Prediction**
Logistic regression model trained on player data outputs personalized injury probability scores with clear risk categorization (low/moderate/high).

**Interactive Simulations**
Adjust training load, sleep, stress, and other variables to instantly visualize impact on injury risk.

**Time-Based Projections**
Simulates multi-week risk trends accounting for fatigue accumulation and recovery decay.

**Smart Recommendations**
Generates actionable insights (reduce training load, increase sleep, manage stress, enforce warmups) based on individual player profiles.

**Multi-Page Dashboard**
Streamlit interface with:
- Overview page for at-a-glance metrics
- Player Analysis for individual performance review
- Simulation engine for scenario planning
- Insights for trend analysis and recommendations

---

## Tech Stack & Model

**Stack:** Python | Pandas & NumPy | Scikit-learn | Matplotlib | Streamlit

**Model Features:** Previous injury count, training hours/week, sleep hours/night, stress level, BMI, warmup adherence, reaction time, balance score, agility score

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

---

## Why This Project Matters

* **End-to-end workflow**: From data analysis to deployed decision tool
* **Beyond prediction**: Combines modeling, simulation, and actionable recommendations
* **Real-world focus**: Built for usability and impact, not just metrics
* **Production-ready**: Demonstrates deployment-ready application architecture

---

## Roadmap

- Upgrade to gradient boosting models (XGBoost, LightGBM)
- Add model explainability (SHAP values)
- Sensitivity analysis for optimized recommendations
- Real-time data pipeline integration

---

## Author

[Tochi Nnamdi]

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tochifootball.streamlit.app/)


