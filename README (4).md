# Hospital Readmission Risk Predictor

## Project Overview

This project develops a responsible AI system to predict 30-day hospital readmission risk for diabetic patients using machine learning, probability calibration, explainability, and fairness auditing.

The system uses XGBoost with SHAP explainability and evaluates demographic bias across race, gender, and age groups.

---

# Objectives

- Predict hospital readmission within 30 days
- Handle class imbalance effectively
- Calibrate predicted probabilities
- Explain predictions using SHAP
- Audit fairness across demographic subgroups

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Docker

---

# Project Structure

```bash
data/
models/
notebooks/
outputs/
src/
```

---

# Key Features

## Machine Learning
- Logistic Regression baseline
- XGBoost classifier
- Hyperparameter tuning

## Responsible AI
- SHAP explainability
- Bias auditing
- Calibration curve analysis
- Threshold optimization

---

# Evaluation Metrics

- ROC-AUC
- PR-AUC
- Recall
- Precision
- F1-score
- Brier Score

---

# Outputs

- Calibration Curve
- SHAP Summary Plot
- SHAP Local Explanations
- Bias Audit CSV

---

# Run Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python src/train.py
```

---

# Docker Execution

```bash
docker-compose up --build
```

---

# Author

Sai Kiran