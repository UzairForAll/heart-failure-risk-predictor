"""Train, tune, and explain heart failure risk models."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import FEATURES

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT.parent / "data" / "heart_disease.csv"
MODEL_PATH = ROOT / "model.pkl"
META_PATH = ROOT / "model_meta.json"

CV_FOLDS = 5
RANDOM_STATE = 42


def build_candidates() -> list[tuple[str, object]]:
    return [
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6)),
        ("Logistic Regression", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ("Naive Bayes", GaussianNB()),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
        ),
        (
            "Neural Network",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def score_predictions(y_true, probabilities, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
    }


def cross_validate_model(pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(pipeline, x, y, cv=cv, method="predict_proba")[:, 1]
    metrics = score_predictions(y, probabilities, threshold=0.5)
    return {
        "cv_folds": CV_FOLDS,
        "cv_accuracy": metrics["accuracy"],
        "cv_f1": metrics["f1"],
        "cv_recall": metrics["recall"],
        "cv_roc_auc": metrics["roc_auc"],
    }


def tune_threshold(y_true, probabilities) -> tuple[float, dict]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    best_threshold = 0.5
    best_f1 = -1.0

    for idx, threshold in enumerate(thresholds):
        f1 = f1_score(y_true, (probabilities >= threshold).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    tuned_metrics = score_predictions(y_true, probabilities, best_threshold)
    return best_threshold, tuned_metrics


def tune_random_forest(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[Pipeline, dict]:
    base = make_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )
    )
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions={
            "model__n_estimators": [200, 300, 400, 500],
            "model__max_depth": [4, 6, 8, 10, None],
            "model__min_samples_split": [2, 4, 6],
            "model__min_samples_leaf": [1, 2, 3],
            "model__max_features": ["sqrt", "log2", None],
        },
        n_iter=24,
        scoring="f1",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    print(f"Tuned Random Forest params: {search.best_params_}")
    print(f"Tuned CV F1: {search.best_score_:.2%}")
    clean_params = {
        key.replace("model__", ""): value for key, value in search.best_params_.items()
    }
    return search.best_estimator_, clean_params


def build_feature_metadata(df: pd.DataFrame) -> dict:
    correlations = {
        feature: round(float(df[feature].corr(df["DEATH_EVENT"])), 4) for feature in FEATURES
    }
    stats = {}
    for feature in FEATURES:
        stats[feature] = {
            "median": round(float(df[feature].median()), 4),
            "mean": round(float(df[feature].mean()), 4),
        }
    return {"feature_correlations": correlations, "feature_stats": stats}


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=FEATURES + ["DEATH_EVENT"])

    x = df[FEATURES]
    y = df["DEATH_EVENT"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results: list[dict] = []
    best_name = ""
    best_model: Pipeline | None = None
    best_ranking = (-1.0, -1.0, -1.0)

    print("Cross-validated model comparison:")
    for name, estimator in build_candidates():
        pipeline = make_pipeline(estimator)
        cv_metrics = cross_validate_model(pipeline, x_train, y_train)
        pipeline.fit(x_train, y_train)
        holdout_probs = pipeline.predict_proba(x_test)[:, 1]
        holdout_metrics = score_predictions(y_test, holdout_probs, threshold=0.5)

        entry = {"name": name, **cv_metrics, **{f"holdout_{k}": v for k, v in holdout_metrics.items() if k != "threshold"}}
        results.append(entry)

        ranking_score = (cv_metrics["cv_f1"], cv_metrics["cv_roc_auc"], cv_metrics["cv_accuracy"])
        if ranking_score > best_ranking:
            best_ranking = ranking_score
            best_name = name
            best_model = pipeline

        print(
            f"{name:22} | cv_f1={cv_metrics['cv_f1']:.2%} "
            f"cv_recall={cv_metrics['cv_recall']:.2%} holdout_f1={holdout_metrics['f1']:.2%}"
        )

    if best_model is None:
        raise RuntimeError("No model was trained.")

    # Hyper-tune the winning family when it is Random Forest.
    tuned_params = None
    if best_name == "Random Forest":
        best_model, tuned_params = tune_random_forest(x_train, y_train)

    train_probs = best_model.predict_proba(x_train)[:, 1]
    threshold, tuned_train_metrics = tune_threshold(y_train, train_probs)

    test_probs = best_model.predict_proba(x_test)[:, 1]
    tuned_test_metrics = score_predictions(y_test, test_probs, threshold)

    feature_metadata = build_feature_metadata(df)

    if hasattr(best_model.named_steps["model"], "feature_importances_"):
        importances = best_model.named_steps["model"].feature_importances_
        feature_metadata["feature_importances"] = {
            feature: round(float(score), 4)
            for feature, score in zip(FEATURES, importances)
        }

    results.sort(key=lambda item: (item["cv_f1"], item["cv_roc_auc"], item["cv_accuracy"]), reverse=True)

    joblib.dump(best_model, MODEL_PATH)

    metadata = {
        "selected_model": best_name,
        "selection_rule": "Highest 5-fold CV F1, then ROC-AUC, then accuracy",
        "threshold_rule": "Threshold tuned on training split to maximize F1",
        "decision_threshold": threshold,
        "test_samples": int(len(y_test)),
        "feature_count": len(FEATURES),
        "tuned_params": tuned_params,
        "tuned_train_metrics": tuned_train_metrics,
        "tuned_test_metrics": tuned_test_metrics,
        "models": results,
        **feature_metadata,
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"Best model: {best_name}")
    print(f"Decision threshold: {threshold:.3f}")
    print(
        f"Tuned test metrics: acc={tuned_test_metrics['accuracy']:.2%} "
        f"f1={tuned_test_metrics['f1']:.2%} recall={tuned_test_metrics['recall']:.2%}"
    )
    print(f"Saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}")


if __name__ == "__main__":
    main()
