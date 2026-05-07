"""
pipeline.py
전체 파이프라인 통합 실행기.
ingest → preprocess → embed → score → (visualize) 를 순서대로 실행.
Gradio app 과 CLI 양쪽에서 호출 가능하도록 설계.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional

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
