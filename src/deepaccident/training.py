"""Training utilities and calibrated metrics for DeepAccident risk models."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def risk_ttc_loss(
    output: Dict[str, torch.Tensor],
    risk_target: torch.Tensor,
    ttc_target_s: torch.Tensor,
    ttc_mask: torch.Tensor,
    positive_weight: torch.Tensor,
    prediction_horizon_s: float,
    ttc_scale: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    risk_loss = F.binary_cross_entropy_with_logits(
        output["risk_logit"], risk_target, pos_weight=positive_weight
    )
    mask_count = ttc_mask.sum()
    if float(mask_count.detach().cpu()) > 0.0:
        ttc_loss = (
            F.smooth_l1_loss(
                output["ttc_s"] / prediction_horizon_s,
                ttc_target_s / prediction_horizon_s,
                reduction="none",
            )
            * ttc_mask
        ).sum() / mask_count
    else:
        ttc_loss = output["ttc_s"].sum() * 0.0
    total = risk_loss + float(ttc_scale) * ttc_loss
    return total, {
        "total": float(total.detach().cpu()),
        "risk_bce": float(risk_loss.detach().cpu()),
        "ttc_smooth_l1": float(ttc_loss.detach().cpu()),
    }


def _best_f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    if thresholds.size == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1.0e-12
    )
    return float(thresholds[int(np.nanargmax(f1))])


def binary_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if target.shape != probability.shape or target.size == 0:
        raise ValueError("target/probability must be non-empty arrays with equal shape")
    threshold = _best_f1_threshold(target, probability) if threshold is None else float(threshold)
    prediction = (probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(target, prediction, labels=(0, 1))
    tn, fp, fn, tp = (int(value) for value in matrix.reshape(-1))
    metrics = {
        "threshold": threshold,
        "average_precision": float(average_precision_score(target, probability)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "brier": float(brier_score_loss(target, probability)),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
        "positive_rate": float(target.mean()),
        "predicted_positive_rate": float(prediction.mean()),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(target, probability))
        if np.unique(target).size > 1
        else float("nan")
    )
    return metrics


def promotion_gate(
    evaluation: Mapping[str, Any],
    min_source_groups: int = 30,
    min_auc: float = 0.65,
    min_ap_lift: float = 0.05,
    min_recall: float = 0.50,
    max_false_positive_rate: float = 0.25,
) -> Dict[str, Any]:
    """Apply a conservative offline gate before any CARLA integration.

    Passing this gate only makes an encoder eligible for paired closed-loop
    tests. It never promotes the model directly into the driving runtime.
    """

    validation = dict(evaluation["validation"])
    testing = dict(evaluation["test"])
    source_groups = int(evaluation["dataset_audit"].get("source_scenario_groups", 0))

    def _recall(metrics: Mapping[str, Any]) -> float:
        true_positive = float(metrics.get("true_positive", 0.0))
        false_negative = float(metrics.get("false_negative", 0.0))
        return true_positive / max(1.0, true_positive + false_negative)

    def _false_positive_rate(metrics: Mapping[str, Any]) -> float:
        false_positive = float(metrics.get("false_positive", 0.0))
        true_negative = float(metrics.get("true_negative", 0.0))
        return false_positive / max(1.0, false_positive + true_negative)

    validation_ap_lift = float(validation["average_precision"]) - float(
        validation["positive_rate"]
    )
    test_ap_lift = float(testing["average_precision"]) - float(testing["positive_rate"])
    test_recall = _recall(testing)
    validation_false_positive_rate = _false_positive_rate(validation)
    checks = {
        "enough_independent_source_groups": source_groups >= int(min_source_groups),
        "validation_auc": float(validation["roc_auc"]) >= float(min_auc),
        "validation_ap_lift": validation_ap_lift >= float(min_ap_lift),
        "validation_false_positive_rate": (
            validation_false_positive_rate <= float(max_false_positive_rate)
        ),
        "test_auc": float(testing["roc_auc"]) >= float(min_auc),
        "test_ap_lift": test_ap_lift >= float(min_ap_lift),
        "test_recall": test_recall >= float(min_recall),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "eligible_for_paired_carla" if passed else "hold_not_promoted",
        "checks": checks,
        "observed": {
            "source_scenario_groups": source_groups,
            "validation_ap_lift": validation_ap_lift,
            "validation_false_positive_rate": validation_false_positive_rate,
            "test_ap_lift": test_ap_lift,
            "test_recall": test_recall,
        },
        "thresholds": {
            "min_source_groups": int(min_source_groups),
            "min_auc": float(min_auc),
            "min_ap_lift": float(min_ap_lift),
            "min_recall": float(min_recall),
            "max_false_positive_rate": float(max_false_positive_rate),
        },
        "meaning": (
            "Passing permits paired CARLA evaluation only; it does not install "
            "or activate the encoder in SimLingo."
        ),
    }
