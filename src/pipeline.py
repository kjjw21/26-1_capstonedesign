"""
pipeline.py
전체 파이프라인 통합 실행기.
ingest → preprocess → embed → score → (visualize) 를 순서대로 실행.
Gradio app 과 CLI 양쪽에서 호출 가능하도록 설계.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .ingest import ingest
from .preprocess import preprocess, PreprocessResult
from .embed import embed_images, embed_texts
from .score import score_video, VideoScore


@dataclass
class PipelineResult:
    ingest_meta: dict
    preprocess_result: PreprocessResult
    video_score: VideoScore
    image_embeddings: np.ndarray
    text_embedding: np.ndarray
    # v2 단계에서 채워지는 필드 (multilingual + classifier + retrieval)
    classifier_result: Optional[dict] = None
    audio_source_candidates: List[dict] = field(default_factory=list)
    video_source_candidates: List[dict] = field(default_factory=list)
    backend: str = "openai-clip"


def run_pipeline(
    source: str,
    whisper_model: str = "base",
    scene_threshold: float = 27.0,
    clip_model: str = "ViT-L/14",
    status_cb: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """
    영상 소스(파일 경로 또는 URL)를 받아 불일치 분석 결과를 반환한다.

    Parameters
    ----------
    source          : 로컬 파일 경로 또는 영상 URL
    whisper_model   : 'tiny' | 'base' | 'small' | 'medium' | 'large'
    scene_threshold : PySceneDetect 민감도 (낮을수록 더 많은 장면 감지)
    clip_model      : 'ViT-B/32' (빠름) | 'ViT-L/14' (정확)
    status_cb       : 진행 상태 메시지 콜백 fn(message: str)

    Returns
    -------
    PipelineResult
    """
    def status(msg: str):
        if status_cb:
            status_cb(msg)
        else:
            print(f"[pipeline] {msg}")

    # ── Step 1: Ingest ──
    status("입력 처리 중...")
    meta = ingest(source)
    status(f"입력 완료: {meta['title']}")

    # ── Step 2: Preprocess ──
    prep_result = preprocess(
        uid=meta["uid"],
        video_path=meta["video_path"],
        whisper_model=whisper_model,
        scene_threshold=scene_threshold,
        status_cb=status,
    )

    # ── Step 3: Embed ──
    status("CLIP 임베딩 생성 중...")
    image_paths = [kf.path for kf in prep_result.keyframes]
    image_embs = embed_images(image_paths, model_name=clip_model)

    combined_text = prep_result.combined_text or "영상"
    text_emb = embed_texts([combined_text], model_name=clip_model)[0]
    status(f"임베딩 완료 — {len(image_embs)}개 프레임")

    # ── Step 4: Score ──
    status("불일치 분석 중...")
    vs = score_video(
        uid=meta["uid"],
        frame_scores_raw=prep_result.keyframes,
        image_embeddings=image_embs,
        text_embedding=text_emb,
    )
    status(f"분석 완료 — {vs.label} ({vs.overall_score:.1f}/100)")

    return PipelineResult(
        ingest_meta=meta,
        preprocess_result=prep_result,
        video_score=vs,
        image_embeddings=image_embs,
        text_embedding=text_emb,
    )


def run_pipeline_v2(
    source: str,
    whisper_model: str = "tiny",
    scene_threshold: float = 27.0,
    top_k: int = 5,
    status_cb: Optional[Callable[[str], None]] = None,
) -> PipelineResult:
    """
    v2 흐름: 다국어 CLIP (xlm-roberta-base-ViT-B-32) + Phase 1 분류기 +
    원본 검색(Phase C). app.py 가 이 함수를 호출한다.

    Parameters
    ----------
    source          : 로컬 파일 경로 또는 영상 URL
    whisper_model   : 'tiny' 권장 (CPU)
    scene_threshold : PySceneDetect 민감도
    top_k           : 원본 후보 영상 개수
    status_cb       : 진행 상태 콜백
    """
    from . import mclip
    from . import classifier
    from . import retrieval

    def status(msg: str):
        if status_cb:
            status_cb(msg)
        else:
            print(f"[pipeline-v2] {msg}")

    status("입력 처리 중...")
    meta = ingest(source)
    status(f"입력 완료: {meta['title']}")

    prep_result = preprocess(
        uid=meta["uid"],
        video_path=meta["video_path"],
        whisper_model=whisper_model,
        scene_threshold=scene_threshold,
        status_cb=status,
    )

    status("다국어 CLIP 임베딩 생성 중...")
    image_paths = [kf.path for kf in prep_result.keyframes]
    image_embs = mclip.embed_images(image_paths)
    combined_text = prep_result.combined_text or "영상"
    text_emb = mclip.embed_text(combined_text)
    status(f"임베딩 완료 — {len(image_embs)}개 프레임 (xlm-roberta-base-ViT-B-32)")

    status("불일치 분석 중...")
    vs = score_video(
        uid=meta["uid"],
        frame_scores_raw=prep_result.keyframes,
        image_embeddings=image_embs,
        text_embedding=text_emb,
    )
    status(f"점수 산출 완료 — {vs.label} ({vs.overall_score:.1f}/100)")

    # ── Phase 1 분류기 ──
    try:
        clf_out = classifier.predict(
            overall_score=vs.overall_score,
            avg_similarity=vs.avg_similarity,
            min_similarity=vs.min_similarity,
        )
        status(
            f"분류기 결과 — {clf_out['label']} "
            f"(P(fake)={clf_out['prob_fake']:.2f}, confidence={clf_out['confidence']:.2f})"
        )
    except Exception as e:
        status(f"분류기 호출 실패: {e}")
        clf_out = None

    # ── Phase C 원본 검색 ──
    audio_cands: List[dict] = []
    video_cands: List[dict] = []
    try:
        image_mean = image_embs.mean(axis=0)
        audio_cands = retrieval.search_by_text(text_emb, k=top_k)
        video_cands = retrieval.search_by_image(image_mean, k=top_k)
        status(
            f"원본 검색 완료 — 오디오 후보 {len(audio_cands)}건, "
            f"영상 후보 {len(video_cands)}건"
        )
    except Exception as e:
        status(f"원본 검색 실패: {e}")

    return PipelineResult(
        ingest_meta=meta,
        preprocess_result=prep_result,
        video_score=vs,
        image_embeddings=image_embs,
        text_embedding=text_emb,
        classifier_result=clf_out,
        audio_source_candidates=audio_cands,
        video_source_candidates=video_cands,
        backend="xlm-roberta-base-ViT-B-32",
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python pipeline.py <file_or_url>")
        sys.exit(1)

    result = run_pipeline(sys.argv[1])
    vs = result.video_score
    print(f"\n{'='*40}")
    print(f"판정      : {vs.label} ({vs.label_en})")
    print(f"점수      : {vs.overall_score:.1f} / 100")
    print(f"신뢰도    : {vs.confidence:.2f}")
    print(f"의심 구간 : {len(vs.suspicious_intervals)}개")
    print(f"이상 프레임: {vs.anomaly_count}개")
    print(f"\n요약:\n{vs.summary}")
