"""
scripts/train_simple_classifier.py
baseline_scores.jsonl 에서 추출한 영상별 통계 피처로
LogisticRegression 분류기를 학습한다.

피처: [overall_score, avg_similarity, min_similarity,
       anomaly_count, n_frames]
타깃: label_true == "fake"  (binary)

학습 데이터가 ~100개로 작으므로 70/30 분리 + 5-fold CV 로 평가.

실행:
    python scripts/run_baseline_eval.py     # 먼저 baseline 결과를 만든 뒤
    python scripts/train_simple_classifier.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


FEATURE_KEYS = [
    "overall_score",
    "avg_similarity",
    "min_similarity",
    "anomaly_count",
    "n_frames",
]


def load_baseline(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(
            f"baseline 결과 파일이 없습니다: {path}\n"
            "먼저 `python scripts/run_baseline_eval.py` 를 실행하세요."
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_xy(rows: list):
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=np.float32)
    y = np.array([1 if r["label_true"] == "fake" else 0 for r in rows], dtype=np.int64)
    return X, y


def cross_validate(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = {"acc": [], "f1": [], "prec": [], "rec": [], "auc": []}
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), start=1):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)

        clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xtr_s, ytr)
        yp = clf.predict(Xte_s)
        yp_prob = clf.predict_proba(Xte_s)[:, 1]

        scores["acc"].append((yp == yte).mean())
        scores["f1"].append(f1_score(yte, yp, zero_division=0))
        scores["prec"].append(precision_score(yte, yp, zero_division=0))
        scores["rec"].append(recall_score(yte, yp, zero_division=0))
        try:
            scores["auc"].append(roc_auc_score(yte, yp_prob))
        except ValueError:
            scores["auc"].append(float("nan"))

    return scores


def final_fit(X, y, test_size=0.3, seed=42):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(Xtr), ytr)
    yp = clf.predict(scaler.transform(Xte))
    yp_prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
    return clf, scaler, Xte, yte, yp, yp_prob


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("data/eval/baseline_scores.jsonl"),
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_baseline(args.baseline)
    print(f"[train] baseline rows : {len(rows)}")
    X, y = build_xy(rows)
    print(f"[train] X shape       : {X.shape}")
    print(f"[train] class balance : normal={int((y==0).sum())}, fake={int((y==1).sum())}")
    print(f"[train] features      : {FEATURE_KEYS}")

    # ── 5-fold CV ──
    print(f"\n=== 5-fold Cross-Validation ===")
    cv = cross_validate(X, y, n_splits=5, seed=args.seed)
    for k, vals in cv.items():
        arr = np.array(vals)
        print(f"  {k:<5}: mean={arr.mean():.3f}  std={arr.std():.3f}")

    # ── 70/30 hold-out ──
    print(f"\n=== 70/30 Hold-out ===")
    clf, scaler, Xte, yte, yp, yp_prob = final_fit(X, y, test_size=0.3, seed=args.seed)
    print(classification_report(yte, yp, target_names=["normal", "fake"], digits=3))
    print("confusion matrix [rows=true, cols=pred]:")
    print(f"  {confusion_matrix(yte, yp)}")
    try:
        print(f"  ROC-AUC: {roc_auc_score(yte, yp_prob):.3f}")
    except ValueError:
        pass

    # 학습된 회귀계수 — 각 피처가 'fake' 판단에 얼마나 기여하는지
    print(f"\n=== 학습된 가중치 (스케일 적용 후) ===")
    coefs = clf.coef_.flatten()
    for k, c in zip(FEATURE_KEYS, coefs):
        sign = "+" if c >= 0 else "-"
        print(f"  {k:<18}: {sign}{abs(c):.3f}")
    print(f"  {'(intercept)':<18}: {clf.intercept_[0]:+.3f}")


if __name__ == "__main__":
    main()
