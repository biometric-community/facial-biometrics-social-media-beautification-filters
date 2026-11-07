"""Verification and SSIM helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def cosine_scores(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    return np.sum(emb_a * emb_b, axis=1)


def best_accuracy_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return (accuracy, threshold) via sorted score sweep (LFW-style)."""
    labels = labels.astype(np.int32)
    order = np.argsort(scores)
    scores_s = scores[order]
    labels_s = labels[order]
    # Threshold midway between consecutive scores; also -inf / +inf ends
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    # Start: all predicted different (score < -inf) → TP=0 FP=0
    tp = 0
    fp = 0
    best_acc = max(n_pos, n_neg) / len(labels)  # predict all same or all different
    best_thr = float(scores_s[0] - 1.0) if len(scores_s) else 0.0

    # Sweep: classify score >= thr as same
    # Equivalent: as thr decreases, more pairs become "same"
    # Iterate from high to low
    order_desc = np.argsort(-scores)
    scores_d = scores[order_desc]
    labels_d = labels[order_desc]
    tp = 0
    fp = 0
    for i, (s, y) in enumerate(zip(scores_d, labels_d)):
        if y == 1:
            tp += 1
        else:
            fp += 1
        tn = n_neg - fp
        acc = (tp + tn) / len(labels)
        thr = float(s) if i + 1 == len(scores_d) else float((s + scores_d[i + 1]) / 2.0)
        if i + 1 < len(scores_d):
            thr = float((s + scores_d[i + 1]) / 2.0)
        else:
            thr = float(s - 1e-6)
        if acc >= best_acc:
            best_acc = acc
            best_thr = thr
    return float(best_acc), float(best_thr)


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    labels = labels.astype(np.int32)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(neg) == 0 or len(pos) == 0:
        return 0.0
    thr = np.quantile(neg, 1.0 - target_fpr)
    return float((pos >= thr).mean())


def roc_curve_points(
    scores: np.ndarray, labels: np.ndarray, n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int32)
    thresholds = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, n_points)
    fprs, tprs = [], []
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    for thr in thresholds:
        pred = scores >= thr
        tpr = float(((pred) & (labels == 1)).sum() / n_pos)
        fpr = float(((pred) & (labels == 0)).sum() / n_neg)
        tprs.append(tpr)
        fprs.append(fpr)
    return np.asarray(fprs), np.asarray(tprs)


def auc_trapezoid(fpr: np.ndarray, tpr: np.ndarray) -> float:
    order = np.lexsort((tpr, fpr))  # ties in FPR must rise in TPR
    x, y = fpr[order], tpr[order]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def ssim_gray(a: np.ndarray, b: np.ndarray, window: int = 8) -> float:
    """SSIM (paper Eq. 1) on grayscale images, from local statistics in sliding windows."""
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    if a.ndim == 3:
        a = np.mean(a.astype(np.float64), axis=2)
        b = np.mean(b.astype(np.float64), axis=2)
    else:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    def pool(x: np.ndarray) -> np.ndarray:
        return cv2.blur(x, (window, window))[window:-window, window:-window]

    mu_a, mu_b = pool(a), pool(b)
    sigma_a = pool(a * a) - mu_a**2
    sigma_b = pool(b * b) - mu_b**2
    sigma_ab = pool(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return float((num / den).mean())


def summarize_verification(scores: np.ndarray, labels: np.ndarray) -> Dict:
    acc, thr = best_accuracy_threshold(scores, labels)
    fpr, tpr = roc_curve_points(scores, labels)
    return {
        "accuracy": acc,
        "threshold": thr,
        "auc": auc_trapezoid(fpr, tpr),
        "tpr_at_fpr_0.01": tpr_at_fpr(scores, labels, 0.01),
        "tpr_at_fpr_0.001": tpr_at_fpr(scores, labels, 0.001),
        "n_pairs": int(len(labels)),
        "n_same": int((labels == 1).sum()),
        "n_diff": int((labels == 0).sum()),
        "mean_score_same": float(scores[labels == 1].mean()) if (labels == 1).any() else None,
        "mean_score_diff": float(scores[labels == 0].mean()) if (labels == 0).any() else None,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }
