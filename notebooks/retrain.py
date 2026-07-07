# notebooks/retrain.py
"""
Headless retraining script — run by Airflow weekly.
Equivalent to selastone_kaggle_v2.ipynb but without interactive/exploratory cells.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import pickle, json, os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

print("=" * 70)
print("SELASTONE WEEKLY RETRAIN — headless run")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────────
csv_path = "notebooks/archive/Loan_Default.csv"
df = pd.read_csv(csv_path)
print(f"✓ Data loaded: {df.shape}")

# ── Clean ─────────────────────────────────────────────────────────────────────
df = df.drop(['ID'], axis=1)
missing_pct = (df.isnull().sum() / len(df) * 100)
drop_cols = missing_pct[missing_pct > 40].index.tolist()
df = df.drop(columns=drop_cols)

numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'Status' in numeric_cols:
    numeric_cols.remove('Status')

for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# ── Feature engineering ──────────────────────────────────────────────────────
df['loan_to_income']    = df['loan_amount'] / (df['income'] + 1)
df['loan_to_property']  = df['loan_amount'] / (df['property_value'] + 1)
df['credit_to_income']  = df['Credit_Score'] / (df['income'] + 1)
numeric_cols.extend(['loan_to_income', 'loan_to_property', 'credit_to_income'])

X = df[numeric_cols + categorical_cols].copy()
y = df['Status'].copy()

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

for col in numeric_cols:
    q1, q99 = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = X[col].clip(q1, q99)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# ── Train 4 models ────────────────────────────────────────────────────────────
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight, random_state=42,
    n_jobs=-1, eval_metric='logloss', verbosity=0
)
xgb_model.fit(X_train_balanced, y_train_balanced)

lgb_model = lgb.LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    random_state=42, verbosity=-1
)
lgb_model.fit(X_train, y_train)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    random_state=42, n_jobs=-1, class_weight='balanced'
)
rf_model.fit(X_train, y_train)

print("✓ All 4 models trained")

# ── Compare ───────────────────────────────────────────────────────────────────
def compute_metrics(model, X, y, label):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "model": label,
        "auc": roc_auc_score(y, y_proba),
        "f1":  f1_score(y, y_pred),
        "accuracy": accuracy_score(y, y_pred),
    }

results = [
    compute_metrics(xgb_model, X_test_scaled, y_test, "XGBoost"),
    compute_metrics(lgb_model, X_test,        y_test, "LightGBM"),
    compute_metrics(lr_model,  X_test_scaled, y_test, "LogisticRegression"),
    compute_metrics(rf_model,  X_test,        y_test, "RandomForest"),
]

best = max(results, key=lambda x: (x['auc'], x['f1']))
best_model_obj = {
    "XGBoost": xgb_model, "LightGBM": lgb_model,
    "LogisticRegression": lr_model, "RandomForest": rf_model,
}[best['model']]

print(f"🏆 Best model this run: {best['model']} — AUC {best['auc']:.4f}")

# ── Log to MLflow ─────────────────────────────────────────────────────────────
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("selastone_loan_default")

log_fns = {
    "XGBoost": mlflow.xgboost.log_model,
    "LightGBM": mlflow.lightgbm.log_model,
    "LogisticRegression": mlflow.sklearn.log_model,
    "RandomForest": mlflow.sklearn.log_model,
}

with mlflow.start_run(run_name=f"weekly_retrain_{best['model']}"):
    mlflow.log_metrics({
        "test_auc": best['auc'],
        "test_f1": best['f1'],
        "test_accuracy": best['accuracy'],
    })
    log_fns[best['model']](best_model_obj, artifact_path="model")
    run_id = mlflow.active_run().info.run_id

print(f"✓ Logged run {run_id} to MLflow — promotion handled by next DAG task")
print("=" * 70)