"""
score.py
불일치 탐지 및 스코어링 모듈.

- 프레임별 유사도 → 불일치 스코어 (0~100)
- 시간적 이상치 탐지 (temporal anomaly)
- 3단계 판정 레이블
- Suspicious 구간 마킹
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class FrameScore:
    index: int
    timestamp: float
    similarity: float         # CLIP 코사인 유사도 [-1, 1]
    mismatch_score: float     # 불일치 점수 [0, 100] — 높을수록 위험
    is_anomaly: bool = False  # 시간적 이상치 여부


@dataclass
class VideoScore:
    uid: str
    overall_score: float              # 전체 불일치 점수 [0, 100]
    label: str                        # "정상" | "의심" | "높은 위험"
    label_en: str                     # "normal" | "suspicious" | "high_risk"
    confidence: float                 # 판정 신뢰도 [0, 1]
    frame_scores: List[FrameScore]
    suspicious_intervals: List[dict]  # [{'start': float, 'end': float, 'score': float}]
    avg_similarity: float
    min_similarity: float
    anomaly_count: int
    summary: str                      # 자연어 요약


# ──────────────────────────────────────────────
# 유사도 → 불일치 스코어 변환
# ──────────────────────────────────────────────

def similarity_to_mismatch(sim: float) -> float:
    """
    CLIP 코사인 유사도 [-1, 1] → 불일치 점수 [0, 100].
    
    CLIP zero-shot 에서 영상-텍스트 유사도는 보통 0.1~0.4 범위.
    - sim >= 0.30 : 잘 일치  → 낮은 불일치 점수
    - sim  0.15~0.30 : 모호  → 중간 점수
    - sim <  0.15 : 불일치  → 높은 점수
    
    선형 역변환 후 클리핑.
    """
    # 경험적 범위: [0.05, 0.40] → [100, 0] 선형 매핑
    low, high = 0.05, 0.40
    score = (high - sim) / (high - low) * 100
    return float(np.clip(score, 0, 100))


# ──────────────────────────────────────────────
# 시간적 이상치 탐지
# ──────────────────────────────────────────────

def detect_temporal_anomalies(
    similarities: np.ndarray,
    window: int = 3,
    z_thresh: float = 1.5,
) -> np.ndarray:
    """
    슬라이딩 윈도우 내 유사도 변화율로 이상 구간을 탐지한다.
    
    갑작스러운 유사도 하락 = 장면 전환 후 맥락 불일치 구간.

    Parameters
    ----------
    similarities : (N,) 프레임별 유사도
    window       : 이동 평균 윈도우 크기
    z_thresh     : 이상치 판단 z-score 임계값

    Returns
    -------
    bool 배열 (N,) — True 인 인덱스가 이상치
    """
    if len(similarities) < 3:
        return np.zeros(len(similarities), dtype=bool)

    # 이동 평균과의 차이
    from numpy.lib.stride_tricks import sliding_window_view
    pad = window // 2
    padded = np.pad(similarities, pad, mode="edge")
    rolling_mean = np.convolve(padded, np.ones(window) / window, mode="valid")[: len(similarities)]
    diff = similarities - rolling_mean

    # z-score 기반 이상치
    mean_diff = np.mean(diff)
    std_diff = np.std(diff) + 1e-8
    z_scores = (diff - mean_diff) / std_diff
    anomalies = z_scores < -z_thresh  # 평균보다 크게 낮은 구간

    return anomalies


# ──────────────────────────────────────────────
# 의심 구간 병합
# ──────────────────────────────────────────────

def get_suspicious_intervals(
    frame_scores: List[FrameScore],
    score_thresh: float = 55.0,
    merge_gap: float = 3.0,
) -> List[dict]:
    """
    불일치 점수가 임계값 이상인 프레임들을 시간 구간으로 병합한다.

    Parameters
    ----------
    score_thresh : 의심 프레임으로 분류하는 불일치 점수 임계값
    merge_gap    : 이 시간(초) 이내 인접 의심 구간은 병합

    Returns
    -------
    List[dict]: [{'start': float, 'end': float, 'peak_score': float}]
    """
    suspicious = [fs for fs in frame_scores if fs.mismatch_score >= score_thresh]
    if not suspicious:
        return []

    intervals = []
    cur_start = suspicious[0].timestamp
    cur_end = suspicious[0].timestamp
    cur_peak = suspicious[0].mismatch_score

    for fs in suspicious[1:]:
        if fs.timestamp - cur_end <= merge_gap:
            cur_end = fs.timestamp
            cur_peak = max(cur_peak, fs.mismatch_score)
        else:
            intervals.append({"start": cur_start, "end": cur_end, "peak_score": cur_peak})
            cur_start = fs.timestamp
            cur_end = fs.timestamp
            cur_peak = fs.mismatch_score

    intervals.append({"start": cur_start, "end": cur_end, "peak_score": cur_peak})
    return intervals


# ──────────────────────────────────────────────
# 종합 점수 및 판정
# ──────────────────────────────────────────────

def compute_overall_score(
    mismatch_scores: np.ndarray,
    anomaly_flags: np.ndarray,
) -> tuple:
    """
    프레임별 불일치 점수들을 종합하여 단일 점수로 환산.
    이상치 프레임에 가중치를 높인다.

    Returns
    -------
    (overall_score: float, confidence: float)
    """
    if len(mismatch_scores) == 0:
        return 50.0, 0.0

    weights = np.where(anomaly_flags, 2.0, 1.0)
    overall = np.average(mismatch_scores, weights=weights)

    # 신뢰도: 표준편차가 낮고 프레임 수가 많을수록 높음
    std = np.std(mismatch_scores)
    n = len(mismatch_scores)
    confidence = float(np.clip(1.0 - (std / 50.0) + min(n / 20.0, 0.3), 0.3, 1.0))

    return float(overall), confidence


def label_score(score: float) -> tuple:
    """
    점수 → (한국어 레이블, 영어 레이블)
    
    임계값은 뉴스 도메인 경험치 기반:
    - 0~40  : 정상 (영상과 텍스트 맥락 일치)
    - 40~65 : 의심 (일부 불일치 가능성)
    - 65~100: 높은 위험 (명확한 맥락 불일치)
    """
    if score < 40:
        return "정상", "normal"
    elif score < 65:
        return "의심", "suspicious"
    else:
        return "높은 위험", "high_risk"


def generate_summary(
    label: str,
    overall_score: float,
    avg_sim: float,
    anomaly_count: int,
    suspicious_intervals: List[dict],
) -> str:
    """자연어 분석 요약 생성."""
    if label == "정상":
        return (
            f"영상과 텍스트 설명 사이의 맥락이 전반적으로 일치합니다. "
            f"평균 시각-텍스트 유사도 {avg_sim:.3f}, 불일치 점수 {overall_score:.1f}/100."
        )
    elif label == "의심":
        interval_desc = ""
        if suspicious_intervals:
            ts = suspicious_intervals[0]
            interval_desc = f" 특히 {ts['start']:.1f}~{ts['end']:.1f}초 구간({ts['peak_score']:.0f}점)이 주목됩니다."
        return (
            f"영상의 일부 구간에서 텍스트 맥락과의 불일치가 감지되었습니다. "
            f"불일치 점수 {overall_score:.1f}/100, 이상 프레임 {anomaly_count}개.{interval_desc}"
        )
    else:  # 높은 위험
        n_intervals = len(suspicious_intervals)
        return (
            f"영상과 텍스트 설명 사이에 명확한 맥락 불일치가 탐지되었습니다. "
            f"불일치 점수 {overall_score:.1f}/100, 의심 구간 {n_intervals}개, "
            f"이상 프레임 {anomaly_count}개. 원본 영상 확인을 권장합니다."
        )


# ──────────────────────────────────────────────
# 메인 스코어링 함수
# ──────────────────────────────────────────────

def score_video(
    uid: str,
    frame_scores_raw: list,         # preprocess Keyframe 리스트
    image_embeddings: np.ndarray,   # (N, D)
    text_embedding: np.ndarray,     # (D,)
    score_thresh: float = 55.0,
) -> VideoScore:
    """
    키프레임 임베딩과 텍스트 임베딩을 받아 VideoScore 를 계산한다.

    Parameters
    ----------
    uid                : 영상 고유 ID
    frame_scores_raw   : Keyframe 객체 리스트 (timestamp 정보용)
    image_embeddings   : embed.embed_images() 결과
    text_embedding     : embed.embed_texts([combined_text])[0] 결과
    score_thresh       : 의심 프레임 임계값
    """
    from .embed import frame_text_similarities

    # 1. 프레임별 유사도 계산
    similarities = frame_text_similarities(image_embeddings, text_embedding)

    # 2. 불일치 스코어 변환
    mismatch_scores = np.array([similarity_to_mismatch(s) for s in similarities])

    # 3. 시간적 이상치 탐지
    anomalies = detect_temporal_anomalies(similarities)

    # 4. FrameScore 리스트 구성
    frame_scores = []
    for i, (kf, sim, ms, anom) in enumerate(
        zip(frame_scores_raw, similarities, mismatch_scores, anomalies)
    ):
        frame_scores.append(
            FrameScore(
                index=i,
                timestamp=kf.timestamp,
                similarity=float(sim),
                mismatch_score=float(ms),
                is_anomaly=bool(anom),
            )
        )

    # 5. 종합 점수
    overall, confidence = compute_overall_score(mismatch_scores, anomalies)

    # 6. 판정
    label_ko, label_en = label_score(overall)

    # 7. 의심 구간
    intervals = get_suspicious_intervals(frame_scores, score_thresh=score_thresh)

    # 8. 요약
    summary = generate_summary(
        label=label_ko,
        overall_score=overall,
        avg_sim=float(np.mean(similarities)),
        anomaly_count=int(np.sum(anomalies)),
        suspicious_intervals=intervals,
    )

    return VideoScore(
        uid=uid,
        overall_score=overall,
        label=label_ko,
        label_en=label_en,
        confidence=confidence,
        frame_scores=frame_scores,
        suspicious_intervals=intervals,
        avg_similarity=float(np.mean(similarities)),
        min_similarity=float(np.min(similarities)),
        anomaly_count=int(np.sum(anomalies)),
        summary=summary,
    )
