# Model Card

# Hospital Readmission Risk Predictor

---

# Model Overview

This project develops a responsible AI system for predicting 30-day hospital readmission risk in diabetic patients using machine learning techniques. The system combines predictive modeling, probability calibration, explainability, and fairness auditing to support trustworthy healthcare AI deployment.

The final model uses XGBoost with calibrated probability outputs and SHAP-based explainability.

---

# Intended Use

The model is intended to assist healthcare professionals in identifying patients who may be at high risk of hospital readmission within 30 days after discharge.

Potential applications include:

- Clinical decision support
- Hospital discharge planning
- Patient follow-up prioritization
- Healthcare resource allocation
- Risk stratification workflows

The model should only be used as a support tool alongside professional medical judgment.

---

# Not Intended For

This model is NOT intended for:

- Autonomous medical diagnosis
- Replacing clinicians or healthcare professionals
- Emergency medical decision-making
- Determining insurance eligibility
- Denying patient treatment
- Predicting outcomes outside diabetic readmission scenarios

---

# Dataset

## Source

Diabetes 130-US Hospitals Dataset (1999–2008)

## Description

The dataset contains hospital encounter records for diabetic patients collected across multiple hospitals in the United States.

Each row represents a hospital encounter and includes:

- Demographic information
- Admission and discharge details
- Diagnoses
- Medication usage
- Laboratory results
- Readmission status

---

# Target Variable

The original target column:

- `<30`
- `>30`
- `NO`

was converted into a binary classification problem:

| Original Value | Encoded Target |
|---|---|
| `<30` | 1 |
| `>30` | 0 |
| `NO` | 0 |

Positive class (`1`) represents patients readmitted within 30 days.

---

# Features Used

The model uses a combination of:

## Demographic Features
- Race
- Gender
- Age

## Clinical Features
- Time in hospital
- Number of medications
- Number of diagnoses
- Admission type
- Laboratory results
- Diagnosis categories

## Engineered Features
ICD-9 diagnosis codes were grouped into broader clinical categories to reduce feature sparsity and improve generalization.

---

# Data Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Categorical encoding
- Numerical scaling
- ICD-9 grouping
- Leakage removal
- Pipeline serialization using Scikit-learn

Potentially leaky information such as discharge outcomes implying post-discharge knowledge was removed before training.

---

# Model Architecture

## Baseline Model
- Logistic Regression

## Final Model
- XGBoost Classifier

## Calibration
- Sigmoid probability calibration using `CalibratedClassifierCV`

---

# Evaluation Metrics

The following metrics were used:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1-score
- Brier Score

Because the dataset is imbalanced, accuracy was not used as the primary evaluation metric.

---

# Threshold Optimization

The default classification threshold of `0.5` was not used.

A lower operating threshold was selected based on precision-recall analysis in order to prioritize recall and reduce missed high-risk patients.

This decision reflects healthcare priorities where false negatives are more costly than false positives.

---

# Explainability

SHAP (SHapley Additive exPlanations) was used to improve transparency and interpretability.

The following explainability analyses were performed:

- Global SHAP summary plot
- True Positive explanation
- False Positive explanation
- False Negative explanation

SHAP helped identify important clinical factors influencing predictions.

---

# Fairness and Bias Audit

The model was audited across:

- Race groups
- Gender groups
- Age groups

Precision and recall were evaluated separately for each subgroup to identify potential disparities.

Some subgroup performance differences were observed, which may result from:

- Historical healthcare inequities
- Dataset imbalance
- Missing socioeconomic variables
- Underrepresentation of certain populations

These findings were documented transparently as part of responsible AI practice.

---

# Ethical Considerations

This system is designed as a clinical decision-support tool and must not replace human medical expertise.

Healthcare AI systems may inherit biases present in historical healthcare data. Therefore:

- Predictions should be interpreted carefully
- Human oversight is required
- Fairness audits should be performed regularly
- Clinical validation is necessary before deployment

---

# Limitations

## Dataset Limitations
- Historical dataset (1999–2008)
- Limited socioeconomic information
- Missing healthcare context variables

## Modeling Limitations
- Possible subgroup bias
- Limited generalizability beyond diabetic populations
- False positives and false negatives still occur

## Operational Limitations
- Not validated for real-world clinical deployment
- Requires periodic retraining and monitoring

---

# Reproducibility

The project includes:

- Serialized preprocessing pipeline
- Saved trained model
- Requirements file
- Docker support
- Reproducible training workflow

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Docker

---

# Author

Pavan Teja

---

# Project Type

Responsible AI / Healthcare Machine Learning / Explainable AI