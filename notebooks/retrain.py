"""
notebooks/retrain.py
────────────────────────────────────────────────────────────────────────────
Headless weekly retraining script — executed by Airflow Celery worker.
Mirrors selastone_kaggle_v3.ipynb exactly, minus interactive/visual cells.

Key design decisions:
  - Loads tuned hyperparameters from models/best_hyperparams.json
    (produced by the notebook's RandomizedSearchCV cells)
  - No RandomizedSearchCV — re-tuning is a deliberate human decision,
    not something that runs every week on a schedule
  - No matplotlib plots or SHAP visualisations
  - No K-Fold cross-validation output
  - No LoanDefaultPredictor test predictions
  - MLflow uses http://mlflow:5000 (container name, not localhost)

Compliance:
  Explainable  — SHAP background sample saved for API TreeExplainer
  Auditable    — all 4 models + full metrics logged to MLflow; winner promoted
  Policy       — promotion gate enforced separately by airflow_dags/promotion.py
  Privacy      — no PII in training data; only integer app IDs referenced
────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import pickle
import json
import warnings
from pathlib import Path

# ── Project root on sys.path so shared/ is importable ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import mlflow.lightgbm

np.random.seed(42)

print("=" * 70)
print("SELASTONE WEEKLY RETRAIN — headless run")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════════
# 0. LOAD TUNED HYPERPARAMETERS  (produced by notebook Cell 50)
#    Falls back to safe defaults if no tuning file exists yet
# ════════════════════════════════════════════════════════════════════════════
OUTPUT_DIR   = Path(os.environ.get("MODEL_OUTPUT_DIR", "notebooks/models"))
PARAMS_PATH  = OUTPUT_DIR / "best_hyperparams.json"

if PARAMS_PATH.exists():
    with open(PARAMS_PATH) as f:
        saved_params = json.load(f)
    print(f"✓ Loaded tuned hyperparameters from {PARAMS_PATH}")
    for model_name, params in saved_params.items():
        print(f"  {model_name}: {params}")
else:
    print(f"⚠️  {PARAMS_PATH} not found — using conservative defaults")
    print(f"   Run the notebook first to generate tuned hyperparameters.")
    saved_params = {
        "XGBoost": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 3, "reg_lambda": 1.0, "reg_alpha": 0.0,
        },
        "LightGBM": {
            "n_estimators": 300, "num_leaves": 63, "max_depth": 7,
            "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 20, "reg_lambda": 1.0, "reg_alpha": 0.0,
        },
        "LogisticRegression": {
            "C": 0.1, "penalty": "l2", "max_iter": 1000,
        },
        "RandomForest": {
            "n_estimators": 300, "max_depth": 20, "min_samples_split": 5,
            "min_samples_leaf": 2, "max_features": "sqrt",
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA  (mirrors Cells 3–4)
# ════════════════════════════════════════════════════════════════════════════
CSV_PATH = os.environ.get(
    "TRAINING_DATA_PATH",
    "notebooks/archive/Loan_Default.csv"
)

df = pd.read_csv(CSV_PATH)
print(f"\n── Load data ──")
print(f"  ✓ {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Default rate: {df['Status'].mean():.2%}")

# ── Merge labeled feedback data collected via POST /v1/labeled-data ───────────
# Committed to DVC by the daily ingestion DAG only when PSI drift >= 0.2
FEEDBACK_CSV = Path(os.environ.get("FEEDBACK_DATA_PATH", "notebooks/archive/feedback_labeled.csv"))
if FEEDBACK_CSV.exists():
    feedback_df = pd.read_csv(FEEDBACK_CSV)
    if "actual_outcome" in feedback_df.columns:
        feedback_df = feedback_df.rename(columns={"actual_outcome": "Status"})
    shared_cols = [c for c in df.columns if c in feedback_df.columns]
    feedback_df = feedback_df[shared_cols]
    n_before = len(df)
    df = pd.concat([df, feedback_df], ignore_index=True)
    print(f"  + {len(feedback_df)} feedback rows merged  (total: {df.shape[0]:,})")
    print(f"  Updated default rate: {df['Status'].mean():.2%}")
else:
    print(f"  No feedback CSV at {FEEDBACK_CSV} — training on baseline only")


# ════════════════════════════════════════════════════════════════════════════
# 2. EXPLORE & CLEAN  (mirrors Cell 6)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Clean ──")

df = df.drop(['ID'], axis=1)

missing_pct = (df.isnull().sum() / len(df) * 100)
drop_cols   = missing_pct[missing_pct > 40].index.tolist()
df          = df.drop(columns=drop_cols)
print(f"  Dropped {len(drop_cols)} high-missing columns: {drop_cols}")

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

print(f"  ✓ Missing values imputed")


# ════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (mirrors Cell 8)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Feature engineering ──")

df['loan_to_income']   = df['loan_amount']  / (df['income']         + 1)
df['loan_to_property'] = df['loan_amount']  / (df['property_value'] + 1)
df['credit_to_income'] = df['Credit_Score'] / (df['income']         + 1)

new_features = ['loan_to_income', 'loan_to_property', 'credit_to_income']
numeric_cols.extend(new_features)
print(f"  ✓ Created {len(new_features)} derived features")


# ════════════════════════════════════════════════════════════════════════════
# 4. REMOVE LEAKY FEATURES  (mirrors Cell 10)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Remove leaky features ──")

LEAKY_FEATURES = [
    'Interest_rate_spread',  # priced per borrower risk tier → post-decision
    'rate_of_interest',      # actual rate charged → reflects risk already assessed
    'Upfront_charges',       # fees set based on risk tier → post-decision
]

df           = df.drop(columns=[f for f in LEAKY_FEATURES if f in df.columns])
numeric_cols = [c for c in numeric_cols if c not in LEAKY_FEATURES]
print(f"  ✓ Dropped: {LEAKY_FEATURES}")
print(f"  Remaining numeric features: {len(numeric_cols)}")


# ════════════════════════════════════════════════════════════════════════════
# 5. PREPARE FEATURES  (mirrors Cell 12)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Prepare features ──")

X = df[numeric_cols + categorical_cols].copy()
y = df['Status'].copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

for col in numeric_cols:
    q1, q99  = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col]   = X[col].clip(q1, q99)

feature_names = X.columns.tolist()
print(f"  ✓ {len(feature_names)} features ready")


# ════════════════════════════════════════════════════════════════════════════
# 6. TRAIN-TEST SPLIT  (mirrors Cell 14)
# ════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n── Train/test split ──")
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")


# ════════════════════════════════════════════════════════════════════════════
# 7. SCALE FEATURES  (mirrors Cell 16)
# ════════════════════════════════════════════════════════════════════════════
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"\n── Scale ──")
print(f"  ✓ StandardScaler fitted on train set")


# ════════════════════════════════════════════════════════════════════════════
# 8. SMOTE  (mirrors Cell 18 — identical settings to notebook)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── SMOTE ──")
smote = SMOTE(k_neighbors=3, sampling_strategy=0.8, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print(f"  Before: {X_train_scaled.shape[0]:,} rows  ({y_train.mean():.2%} default)")
print(f"  After : {X_train_balanced.shape[0]:,} rows  ({y_train_balanced.mean():.2%} default)")


# ════════════════════════════════════════════════════════════════════════════
# 9. TRAIN ALL 4 MODELS using tuned hyperparameters
#    Input matrix per model matches notebook exactly:
#      XGBoost          → X_train_balanced (scaled + SMOTE)
#      LightGBM         → X_train_balanced (scaled + SMOTE)
#      LogisticReg      → X_train_scaled   (scaled, no SMOTE — class_weight='balanced')
#      Random Forest    → X_train          (raw, no SMOTE  — class_weight='balanced')
# ════════════════════════════════════════════════════════════════════════════
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n── Train models (scale_pos_weight={scale_pos_weight:.2f}) ──")

# XGBoost  (mirrors Cell 22)
xgb_model = xgb.XGBClassifier(
    **saved_params["XGBoost"],
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    verbosity=0
)
xgb_model.fit(X_train_balanced, y_train_balanced)
print(f"  ✓ XGBoost  ({xgb_model.n_estimators} trees, depth {xgb_model.max_depth})")

# LightGBM  (mirrors Cell 32)
lgb_model = lgb.LGBMClassifier(
    **saved_params["LightGBM"],
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_balanced, y_train_balanced)
print(f"  ✓ LightGBM  ({lgb_model.n_estimators} trees, depth {lgb_model.max_depth})")

# Logistic Regression  (mirrors Cell 40)
lr_model = LogisticRegression(
    **saved_params["LogisticRegression"],
    solver='saga',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lr_model.fit(X_train_scaled, y_train)
print(f"  ✓ Logistic Regression")

# Random Forest  (mirrors Cell 46)
rf_model = RandomForestClassifier(
    **saved_params["RandomForest"],
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)
print(f"  ✓ Random Forest  ({rf_model.n_estimators} trees)")


# ════════════════════════════════════════════════════════════════════════════
# 10. COMPARE MODELS  (mirrors Cell 52)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Model comparison ──")

def compute_metrics(model, X, y, label):
    """Full metrics dict — matches notebook Cell 52 exactly."""
    y_pred       = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    cm_          = confusion_matrix(y, y_pred)
    tn_, fp_, fn_, tp_ = cm_.ravel()
    precision = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
    recall    = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
    return {
        "model":     label,
        "auc":       round(roc_auc_score(y, y_pred_proba), 4),
        "f1":        round(f1_score(y, y_pred),            4),
        "accuracy":  round(accuracy_score(y, y_pred),      4),
        "precision": round(precision,                       4),
        "recall":    round(recall,                          4),
        "tp":        int(tp_), "fp": int(fp_),
        "fn":        int(fn_), "tn": int(tn_),
    }

results = [
    compute_metrics(xgb_model, X_test_scaled, y_test, "XGBoost"),
    compute_metrics(lgb_model, X_test_scaled, y_test, "LightGBM"),
    compute_metrics(lr_model,  X_test_scaled, y_test, "LogisticRegression"),
    compute_metrics(rf_model,  X_test,        y_test, "RandomForest"),
]

header = f"{'Model':<22} {'AUC':>7} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
print(f"\n{header}")
print("-" * len(header))
for r in sorted(results, key=lambda x: (x['auc'], x['f1']), reverse=True):
    print(f"  {r['model']:<20} {r['auc']:>7.4f} {r['f1']:>7.4f} "
          f"{r['accuracy']:>7.4f} {r['precision']:>7.4f} {r['recall']:>7.4f}")

# Select winner — AUC primary, F1 tiebreaker (matches notebook Cell 52)
best = max(results, key=lambda x: (x['auc'], x['f1']))
best_model_obj = {
    "XGBoost":            xgb_model,
    "LightGBM":           lgb_model,
    "LogisticRegression": lr_model,
    "RandomForest":       rf_model,
}[best['model']]

print(f"\n  🏆 Winner: {best['model']}  AUC={best['auc']:.4f}  F1={best['f1']:.4f}")


# ════════════════════════════════════════════════════════════════════════════
# 11. SAVE ARTEFACTS  (mirrors Cell 54)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Save artefacts → {OUTPUT_DIR} ──")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best model
with open(OUTPUT_DIR / 'best_model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)
print(f"  ✓ best_model.pkl  ({best['model']})")

# All individual models
for name, obj in [
    ('xgboost_model.pkl', xgb_model),
    ('lgbm_model.pkl',    lgb_model),
    ('logreg_model.pkl',  lr_model),
    ('rf_model.pkl',      rf_model),
]:
    with open(OUTPUT_DIR / name, 'wb') as f:
        pickle.dump(obj, f)
print(f"  ✓ Individual model .pkl files")

# Scaler
with open(OUTPUT_DIR / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ scaler.pkl")

# Feature names
with open(OUTPUT_DIR / 'feature_names.json', 'w') as f:
    json.dump(feature_names, f, indent=2)
print(f"  ✓ feature_names.json  ({len(feature_names)} features)")

# Metadata
all_model_metrics = {
    r['model']: {k: v for k, v in r.items() if k != 'model'}
    for r in results
}
metadata = {
    'best_model':              best['model'],
    'best_model_file':         'models/best_model.pkl',
    'selection_criterion':     'test_auc_roc',
    'test_auc':                best['auc'],
    'test_f1':                 best['f1'],
    'test_accuracy':           best['accuracy'],
    'test_precision':          best['precision'],
    'test_recall':             best['recall'],
    'train_samples':           int(len(X_train)),
    'test_samples':            int(len(X_test)),
    'num_features':            len(feature_names),
    'default_rate':            float(y.mean()),
    'leaky_features_removed':  LEAKY_FEATURES,
    'hyperparams_source':      str(PARAMS_PATH),
    'all_models':              all_model_metrics,
}
with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ metadata.json")

# SHAP background sample — needed by API TreeExplainer (mirrors Cell 54)
bg_idx = np.random.default_rng(42).choice(
    X_train_scaled.shape[0],
    size=min(100, X_train_scaled.shape[0]),
    replace=False
)
shap_background = X_train_scaled[bg_idx]
with open(OUTPUT_DIR / 'shap_background.pkl', 'wb') as f:
    pickle.dump(shap_background, f)
print(f"  ✓ shap_background.pkl  ({shap_background.shape[0]} rows)")


# ════════════════════════════════════════════════════════════════════════════
# 12. FIT & SAVE FEATURE PIPELINE  (mirrors Cell 62)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── Fit FeaturePipeline ──")
from shared.features import FeaturePipeline

X_train_raw   = X_train.copy()   # unscaled — pipeline fits its own scaler
prod_pipeline = FeaturePipeline()
prod_pipeline.fit(X_train_raw, target_col='Status')

with open(OUTPUT_DIR / 'feature_pipeline.pkl', 'wb') as f:
    pickle.dump(prod_pipeline, f)
print(f"  ✓ feature_pipeline.pkl")


# ════════════════════════════════════════════════════════════════════════════
# 13. LOG ALL 4 MODELS TO MLFLOW  (mirrors Cell 60)
#     Promotion handled separately by airflow_dags/promotion.py (DAG task t3)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n── MLflow logging ──")

MLFLOW_URI      = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "selastone_loan_default"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"  Tracking URI : {MLFLOW_URI}")
print(f"  Experiment   : {EXPERIMENT_NAME}")

model_log_fns = {
    "XGBoost":            mlflow.xgboost.log_model,
    "LightGBM":           mlflow.lightgbm.log_model,
    "LogisticRegression": mlflow.sklearn.log_model,
    "RandomForest":       mlflow.sklearn.log_model,
}

model_objects = {
    "XGBoost":            xgb_model,
    "LightGBM":           lgb_model,
    "LogisticRegression": lr_model,
    "RandomForest":       rf_model,
}

model_params_log = {
    "XGBoost": {
        **saved_params["XGBoost"],
        "scale_pos_weight": float(scale_pos_weight),
        "smote":            True,
        "leaky_removed":    True,
        "input":            "X_train_balanced_scaled",
    },
    "LightGBM": {
        **saved_params["LightGBM"],
        "scale_pos_weight": float(scale_pos_weight),
        "smote":            True,
        "leaky_removed":    True,
        "input":            "X_train_balanced_scaled",
    },
    "LogisticRegression": {
        **saved_params["LogisticRegression"],
        "solver":           "saga",
        "class_weight":     "balanced",
        "smote":            False,
        "leaky_removed":    True,
        "input":            "X_train_scaled",
    },
    "RandomForest": {
        **saved_params["RandomForest"],
        "class_weight":     "balanced",
        "smote":            False,
        "leaky_removed":    True,
        "input":            "X_train_raw",
    },
}

run_ids = {}

for name, result in zip(
    ["XGBoost", "LightGBM", "LogisticRegression", "RandomForest"],
    results
):
    print(f"\n  Logging {name} ...")
    with mlflow.start_run(run_name=f"weekly_retrain_{name}"):
        # Hyperparameters
        mlflow.log_params(model_params_log[name])
        # Metrics
        mlflow.log_metrics({
            "test_auc":        result["auc"],
            "test_f1":         result["f1"],
            "test_accuracy":   result["accuracy"],
            "test_precision":  result["precision"],
            "test_recall":     result["recall"],
            "true_positives":  float(result["tp"]),
            "false_negatives": float(result["fn"]),
        })
        # Model artefact
        model_log_fns[name](model_objects[name], artifact_path="model")
        # Shared artefacts — scaler, features, metadata, SHAP background
        mlflow.log_artifact(str(OUTPUT_DIR / 'scaler.pkl'))
        mlflow.log_artifact(str(OUTPUT_DIR / 'feature_names.json'))
        mlflow.log_artifact(str(OUTPUT_DIR / 'metadata.json'))
        mlflow.log_artifact(str(OUTPUT_DIR / 'shap_background.pkl'))
        mlflow.log_artifact(str(OUTPUT_DIR / 'best_hyperparams.json'))
        run_ids[name] = mlflow.active_run().info.run_id
    print(f"    ✓ run_id={run_ids[name]}")


# ════════════════════════════════════════════════════════════════════════════
# 14. SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"SELASTONE WEEKLY RETRAIN — COMPLETE")
print(f"{'='*70}")
print(f"  Dataset        : {len(df):,} rows  |  {len(feature_names)} features")
print(f"  Default rate   : {y.mean():.2%}")
print(f"  Leaky removed  : {LEAKY_FEATURES}")
print(f"  Hyperparams    : {'tuned (from notebook)' if PARAMS_PATH.exists() else 'defaults'}")
print(f"  Winner         : {best['model']}  AUC={best['auc']:.4f}  F1={best['f1']:.4f}")
print(f"  MLflow URI     : {MLFLOW_URI}")
print(f"  Runs logged    : {len(run_ids)}")
for name, rid in run_ids.items():
    tag = "  ← winner" if name == best['model'] else ""
    print(f"    {name:<22} {rid}{tag}")
print(f"\n  Promotion      : handled by airflow_dags/promotion.py (DAG task t3)")
print(f"  Artefacts      : {OUTPUT_DIR.resolve()}")
print(f"{'='*70}")
