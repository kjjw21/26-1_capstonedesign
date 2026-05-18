"""
src/classifier.py
학습된 Phase 1 분류기 (StandardScaler + LogisticRegression on 3 similarity
features) 를 로드해 새 영상의 점수를 받아 예측한다.

자산 파일: data/app/classifier.joblib  (scripts/export_app_assets.py 가 생성)
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np


_ASSET_PATH = Path("data/app/classifier.joblib")
_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        if not _ASSET_PATH.exists():
            raise FileNotFoundError(
                f"분류기 자산이 없습니다: {_ASSET_PATH}\n"
                "먼저 `python scripts/export_app_assets.py` 를 실행하세요."
            )
        _bundle = joblib.load(_ASSET_PATH)
    return _bundle


def predict(
    overall_score: float,
    avg_similarity: float,
    min_similarity: float,
) -> dict:
    """3개 통계 feature 로부터 위조 확률을 추정한다.

    Returns
    -------
    dict
        prob_fake : 0.0 ~ 1.0
        label     : 'normal' | 'fake'
        confidence: |prob_fake - 0.5| * 2  (0 = 모름, 1 = 확신)
    """
    b = _load()
    feats = np.array([[overall_score, avg_similarity, min_similarity]], dtype=np.float32)
    feats_s = b["scaler"].transform(feats)
    prob_fake = float(b["clf"].predict_proba(feats_s)[0, 1])
    return {
        "prob_fake": prob_fake,
        "label": "fake" if prob_fake >= 0.5 else "normal",
        "confidence": abs(prob_fake - 0.5) * 2,
    }


def info() -> dict:
    """학습 시 메타데이터 반환 (UI 표시용)."""
    b = _load()
    return {
        "feature_keys": b["feature_keys"],
        "n_train": b["n_train"],
        "class_balance": b["class_balance"],
        "coefficients": dict(zip(b["feature_keys"], b["clf"].coef_.flatten().tolist())),
        "intercept": float(b["clf"].intercept_[0]),
    }
