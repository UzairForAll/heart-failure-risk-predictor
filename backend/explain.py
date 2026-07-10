from pathlib import Path

import numpy as np

from features import FEATURE_LABELS, FEATURES

ROOT = Path(__file__).resolve().parent


def _direction_label(feature: str, value: float, typical: float, correlation: float) -> str:
    if value > typical:
        comparison = "higher than typical"
    elif value < typical:
        comparison = "lower than typical"
    else:
        comparison = "close to typical"

    # Positive correlation means higher values tend to increase risk.
    if correlation >= 0:
        risk_effect = "increase" if value > typical else "reduce"
    else:
        risk_effect = "increase" if value < typical else "reduce"

    return f"{FEATURE_LABELS[feature]} is {comparison}, which tends to {risk_effect} estimated risk."


def explain_prediction(model, metadata: dict, payload: dict) -> list[dict]:
    row = np.array([[payload[feature] for feature in FEATURES]])
    scaler = model.named_steps["scaler"]
    estimator = model.named_steps["model"]

    scaled = scaler.transform(row)[0]
    correlations = metadata.get("feature_correlations", {})
    stats = metadata.get("feature_stats", {})
    importances = metadata.get("feature_importances", {})

    contributions: list[dict] = []
    for index, feature in enumerate(FEATURES):
        correlation = float(correlations.get(feature, 0.0))
        importance = float(importances.get(feature, abs(correlation)))
        z_score = float(scaled[index])
        impact = importance * z_score * (1 if correlation >= 0 else -1)

        typical = float(stats.get(feature, {}).get("median", 0.0))
        value = float(payload[feature])

        contributions.append(
            {
                "feature": FEATURE_LABELS[feature],
                "feature_key": feature,
                "value": value,
                "typical_value": typical,
                "impact_score": round(impact, 4),
                "direction": "increases_risk" if impact > 0 else "reduces_risk",
                "summary": _direction_label(feature, value, typical, correlation),
            }
        )

    contributions.sort(key=lambda item: abs(item["impact_score"]), reverse=True)
    return contributions[:3]
