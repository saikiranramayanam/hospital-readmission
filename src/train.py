import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from sklearn.calibration import (
    CalibratedClassifierCV
)

from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/cleaned_data.csv")

# Features & target
X = df.drop(columns=["target"])
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Detect columns
categorical_features = X.select_dtypes(
    include="object"
).columns

numeric_features = X.select_dtypes(
    exclude="object"
).columns

# Pipelines
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# XGBoost Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),

        ("classifier",
         XGBClassifier(
             objective="binary:logistic",
             eval_metric="logloss",
             random_state=42
         ))
    ]
)

# Train
pipeline.fit(X_train, y_train)

# Calibrate
calibrated_model = CalibratedClassifierCV(
    estimator=pipeline,
    method="sigmoid",
    cv=3
)

calibrated_model.fit(X_train, y_train)

# Predictions
y_prob = calibrated_model.predict_proba(X_test)[:,1]

# Metrics
roc_auc = roc_auc_score(y_test, y_prob)

pr_auc = average_precision_score(y_test, y_prob)

print("ROC-AUC:", roc_auc)
print("PR-AUC:", pr_auc)

# Save model
joblib.dump(
    calibrated_model,
    "models/calibrated_model.pkl"
)

print("Training Complete")