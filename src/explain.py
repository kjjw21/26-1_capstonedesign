"""
explain.py
Claude API 를 활용한 불일치 근거 자연어 설명 생성 모듈.

분석 결과(VideoScore, PreprocessResult)를 받아
사용자가 이해하기 쉬운 한국어 설명을 생성한다.
"""

import json
import urllib.request
import urllib.error
from .score import VideoScore
from .preprocess import PreprocessResult


import os

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
# 모델 이름은 환경변수로 override 가능. 기본값은 안정성/품질 균형이 좋은 Sonnet 4.5.
CLAUDE_MODEL = os.environ.get("CHEAPFAKE_CLAUDE_MODEL", "claude-sonnet-4-5-20250929")


def build_prompt(video_score: VideoScore, prep: PreprocessResult) -> str:
    """
    분석 결과를 Claude 에게 전달할 프롬프트로 변환한다.
    """
    # 의심 구간 요약
    interval_desc = ""
    if video_score.suspicious_intervals:
        parts = []
        for iv in video_score.suspicious_intervals[:3]:
            parts.append(f"{iv['start']:.0f}~{iv['end']:.0f}초 (최고점수 {iv['peak_score']:.0f})")
        interval_desc = ", ".join(parts)
    else:
        interval_desc = "없음"

    # 불일치 점수 높은 프레임 상위 3개
    top_frames = sorted(
        video_score.frame_scores,
        key=lambda x: x.mismatch_score,
        reverse=True
    )[:3]

    frame_desc = ""
    for fs in top_frames:
        kf_map = {kf.index: kf for kf in prep.keyframes}
        kf = kf_map.get(fs.index)
        ocr = kf.ocr_text[:80] if kf and kf.ocr_text else "(텍스트 없음)"
        frame_desc += f"  - {fs.timestamp:.1f}초: 불일치점수 {fs.mismatch_score:.0f}, 화면텍스트: {ocr}\n"

    stt_preview = prep.stt_text[:300] if prep.stt_text else "(음성 없음)"

    prompt = f"""당신은 미디어 팩트체킹 전문가입니다.
AI 시스템이 숏폼 영상을 분석한 결과를 바탕으로, 일반 사용자가 이해하기 쉬운 한국어로 구체적인 설명을 작성해주세요.

## 분석 결과

- 전체 불일치 점수: {video_score.overall_score:.1f} / 100 (0=완전일치, 100=완전불일치)
- 판정: {video_score.label}
- 영상 길이: {prep.duration:.0f}초
- 분석 프레임 수: {len(video_score.frame_scores)}개
- 이상 프레임 수: {video_score.anomaly_count}개
- 의심 구간: {interval_desc}
- 평균 시각-텍스트 유사도: {video_score.avg_similarity:.3f} (1.0에 가까울수록 일치)
- 최저 유사도 구간: {video_score.min_similarity:.3f}

## 주요 의심 프레임 (불일치 점수 높은 순)
{frame_desc}

## 음성 인식 내용 (STT)
{stt_preview}

## 요청사항

위 데이터를 바탕으로 아래 형식으로 작성해주세요. 반드시 실제 데이터(시간대, 화면 텍스트, 음성 내용)를 구체적으로 언급해야 합니다.

**[판정 요약]**
한 문장으로 이 영상의 신뢰도를 평가해주세요.

**[불일치 근거]**
- 어느 시간대(몇 초 구간)에서 문제가 발생했는지
- 그 구간의 화면에 어떤 내용이 보이는지
- 그 구간의 음성/자막은 어떤 내용인지
- 화면과 음성이 왜 맞지 않는지
위 4가지를 구체적으로 3~4문장으로 설명해주세요.

**[신뢰도 평가]**
이 영상의 전반적인 신뢰도와 시청자가 취해야 할 행동을 1~2문장으로 설명해주세요.

전문 용어 없이 중학생도 이해할 수 있는 쉬운 말로 작성해주세요.
"""
    return prompt


def generate_explanation(
    video_score: VideoScore,
    prep: PreprocessResult,
    api_key: str,
) -> str:
    """
    Claude API 를 호출하여 불일치 근거 설명을 생성한다.

    Parameters
    ----------
    video_score : 스코어링 결과
    prep        : 전처리 결과
    api_key     : Anthropic API 키

    Returns
    -------
    str — 자연어 설명 (마크다운 형식)
    """
    prompt = build_prompt(video_score, prep)

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 2000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }).encode("utf-8")

    req = urllib.request.Request(
        CLAUDE_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["content"][0]["text"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return f"API 오류 ({e.code}): {body[:200]}"
    except Exception as e:
        return f"설명 생성 실패: {str(e)}"


def generate_explanation_fallback(video_score: VideoScore, prep: PreprocessResult) -> str:
    """
    API 키 없을 때 규칙 기반으로 설명 생성 (폴백).
    """
    label = video_score.label
    score = video_score.overall_score
    intervals = video_score.suspicious_intervals
    avg_sim = video_score.avg_similarity
    min_sim = video_score.min_similarity

    # 판정 요약
    if label == "정상":
        summary = "영상과 음성 내용이 전반적으로 일치합니다."
    elif label == "의심":
        summary = "일부 구간에서 영상과 음성 내용이 맞지 않습니다."
    else:
        summary = "영상과 음성 내용 사이에 심각한 불일치가 발견됐습니다."

    # 불일치 근거 (완성된 문장)
    evidence_parts = []
    evidence_parts.append(f"이 영상의 전체 불일치 점수는 {score:.0f}점(100점 만점)이며, 평균 시각-텍스트 유사도는 {avg_sim:.3f}입니다.")

    if intervals:
        ts = intervals[0]
        evidence_parts.append(
            f"특히 {ts['start']:.0f}초~{ts['end']:.0f}초 구간에서 불일치 점수가 최대 {ts['peak_score']:.0f}점으로 가장 높게 나타났습니다."
        )
        if len(intervals) > 1:
            evidence_parts.append(f"총 {len(intervals)}개의 의심 구간이 감지됐습니다.")
    elif video_score.anomaly_count > 0:
        evidence_parts.append(f"총 {video_score.anomaly_count}개 프레임에서 시각-텍스트 불일치가 탐지됐습니다.")
    else:
        evidence_parts.append(f"전체 구간에서 유사도가 고르게 유지되어 특별한 이상 구간은 발견되지 않았습니다.")

    if min_sim < avg_sim * 0.85:
        evidence_parts.append(f"최저 유사도({min_sim:.3f})가 평균({avg_sim:.3f})보다 크게 낮은 구간이 존재합니다.")

    evidence = " ".join(evidence_parts)

    # 권고
    if label == "정상":
        advice = "이 영상은 시각 정보와 음성 정보가 일치하며 신뢰할 수 있는 것으로 판단됩니다."
    elif label == "의심":
        advice = "원본 출처를 확인하고 다른 신뢰할 수 있는 뉴스 소스와 비교해보시기 바랍니다."
    else:
        advice = "이 영상의 공유를 자제하고, 반드시 공신력 있는 언론사의 원본 영상을 확인하세요."

    return f"""**[판정 요약]**
{summary}

**[불일치 근거]**
{evidence}

**[신뢰도 평가]**
💡 {advice}"""