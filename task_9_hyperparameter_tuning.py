import os
import json
import random
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    make_scorer
)

# 1. Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def main():
    print("=== Task 9: Hyperparameter Tuning Pipeline ===")
    
    # 2. Ingestion & Stratified Split (Locked Test Set: 15%)
    raw_data = load_breast_cancer(as_frame=True)
    X, y = raw_data.data, raw_data.target
    
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    print(f"Data Split -> Dev Set: {len(X_dev)} samples, Locked Test Set: {len(X_test)} samples")
    
    # 3. Base Pipeline Architecture
    base_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=SEED, max_iter=2000))
    ])
    
    # 4. Fit Baseline Default Model
    default_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=SEED, max_iter=2000, solver="lbfgs"))
    ])
    default_model.fit(X_dev, y_dev)
    
    # 5. Define Parameter Grid & CV Scheme
    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        "classifier__penalty": ["l1", "l2"],
        "classifier__solver": ["liblinear", "saga"],
        "classifier__class_weight": [None, "balanced"]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scorer = make_scorer(recall_score, pos_label=0)
    
    # 6. Run Systematic Grid Search
    print("\nRunning GridSearchCV across parameter space...")
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_dev, y_dev)
    
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    print(f"Best CV Recall (Class 0): {best_cv_score:.4f}")
    print(f"Best Config: {best_params}")
    
    # 7. Evaluate on Held-Out Test Set (Unseen Confirmation)
    tuned_model = grid_search.best_estimator_
    
    # Predictions
    y_test_pred_def = default_model.predict(X_test)
    y_test_prob_def = default_model.predict_proba(X_test)[:, 1]
    
    y_test_pred_tune = tuned_model.predict(X_test)
    y_test_prob_tune = tuned_model.predict_proba(X_test)[:, 1]
    
    cm_def = confusion_matrix(y_test, y_test_pred_def)
    cm_tune = confusion_matrix(y_test, y_test_pred_tune)
    
    results = {
        "default_model": {
            "test_accuracy": float(accuracy_score(y_test, y_test_pred_def)),
            "test_recall_malignant": float(recall_score(y_test, y_test_pred_def, pos_label=0)),
            "test_precision_malignant": float(precision_score(y_test, y_test_pred_def, pos_label=0)),
            "test_f1_weighted": float(f1_score(y_test, y_test_pred_def, average="weighted")),
            "test_roc_auc": float(roc_auc_score(y_test, y_test_prob_def)),
            "false_negatives": int(cm_def[0, 1]),
            "false_positives": int(cm_def[1, 0])
        },
        "tuned_model": {
            "best_cv_recall": float(best_cv_score),
            "best_params": best_params,
            "test_accuracy": float(accuracy_score(y_test, y_test_pred_tune)),
            "test_recall_malignant": float(recall_score(y_test, y_test_pred_tune, pos_label=0)),
            "test_precision_malignant": float(precision_score(y_test, y_test_pred_tune, pos_label=0)),
            "test_f1_weighted": float(f1_score(y_test, y_test_pred_tune, average="weighted")),
            "test_roc_auc": float(roc_auc_score(y_test, y_test_prob_tune)),
            "false_negatives": int(cm_tune[0, 1]),
            "false_positives": int(cm_tune[1, 0])
        }
    }
    
    print("\n=== Test Set Performance Comparison ===")
    comp_df = pd.DataFrame([results["default_model"], results["tuned_model"]], index=["Default", "Tuned"])
    print(comp_df[["test_accuracy", "test_recall_malignant", "test_precision_malignant", "test_roc_auc", "false_negatives"]])
    
    # 8. Save Artifacts
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(tuned_model, "artifacts/tuned_best_model.joblib")
    with open("artifacts/task9_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nSaved tuned pipeline to: artifacts/tuned_best_model.joblib")
    print("Saved tuning results log to: artifacts/task9_tuning_results.json")

if __name__ == "__main__":
    main()