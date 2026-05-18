"""
preprocess.py
전처리 파이프라인:
  1. 키프레임 추출  (PySceneDetect + FFmpeg)
  2. STT           (OpenAI Whisper)
  3. OCR           (EasyOCR — 화면 자막·CG 텍스트)

결과는 data/processed/<uid>/ 디렉토리에 저장된다.
"""

import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

import cv2
import whisper
import easyocr
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image


PROCESSED_DIR = Path("C:/cheapfake_data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# EasyOCR reader는 모델 로드 비용이 크므로 모듈 수준에서 지연 초기화
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        # 한국어+영어 동시 지원
        _ocr_reader = easyocr.Reader(["ko", "en"], gpu=_has_gpu())
    return _ocr_reader


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ──────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────

@dataclass
class Keyframe:
    index: int
    timestamp: float          # 초 단위
    path: str                 # 저장된 이미지 경로
    ocr_text: str = ""        # 해당 프레임에서 추출된 화면 텍스트


@dataclass
class PreprocessResult:
    uid: str
    video_path: str
    duration: float           # 초
    fps: float
    keyframes: list           # List[Keyframe]
    stt_text: str             # 전체 STT 결과
    stt_segments: list        # List[dict] — 타임스탬프 포함 세그먼트
    combined_text: str        # STT + OCR 합성 텍스트 (임베딩 입력용)
    out_dir: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ──────────────────────────────────────────────
# 1. 키프레임 추출
# ──────────────────────────────────────────────

def extract_keyframes(
    video_path: str,
    out_dir: Path,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    max_frames: int = 40,
) -> list:
    """
    PySceneDetect ContentDetector 로 장면 전환을 감지하고
    각 장면의 대표 프레임을 JPEG 로 저장한다.

    Parameters
    ----------
    threshold     : ContentDetector 민감도 (낮을수록 더 많은 장면 감지)
    min_scene_len : 최소 장면 길이 (프레임 수)
    max_frames    : 최대 키프레임 수 (너무 많으면 잘라냄)

    Returns
    -------
    List[Keyframe]
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len)
    )
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    # 장면이 너무 적으면 균등 샘플링으로 보완
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    if len(scene_list) == 0:
        # 장면 감지 실패 시 5초 간격 균등 샘플링
        timestamps = list(range(0, int(duration), 5))[:max_frames]
        frame_indices = [int(t * fps) for t in timestamps]
    else:
        frame_indices = []
        for start_tc, end_tc in scene_list:
            mid = (start_tc.get_frames() + end_tc.get_frames()) // 2
            frame_indices.append(mid)
        frame_indices = frame_indices[:max_frames]

    keyframes = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img_path = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(img_path), frame)
        ts = frame_idx / fps
        keyframes.append(Keyframe(index=i, timestamp=ts, path=str(img_path)))

    cap.release()
    return keyframes, fps, duration


# ──────────────────────────────────────────────
# 2. STT — Whisper
# ──────────────────────────────────────────────

def run_stt(video_path: str, model_size: str = "base") -> tuple:
    """
    Whisper 로 음성을 텍스트로 변환한다.

    Parameters
    ----------
    model_size : 'tiny' | 'base' | 'small' | 'medium' | 'large'
                 GPU 없을 때는 'base' 권장 (속도 ↔ 정확도 균형)

    Returns
    -------
    (full_text: str, segments: List[dict])
    segments 각 원소: {'start': float, 'end': float, 'text': str}
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(
        video_path,
        language=None,          # 자동 언어 감지
        task="transcribe",
        verbose=False,
        fp16=_has_gpu(),
    )
    full_text = result["text"].strip()
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]
    return full_text, segments


# ──────────────────────────────────────────────
# 3. OCR — EasyOCR
# ──────────────────────────────────────────────

def run_ocr_on_keyframes(keyframes: list) -> list:
    """
    각 키프레임 이미지에 OCR 을 적용하고 Keyframe.ocr_text 를 채운다.
    화면 자막, 뉴스 CG, 날짜/장소 텍스트 등을 추출.

    Returns
    -------
    ocr_text 가 채워진 Keyframe 리스트 (in-place 수정 후 반환)
    """
    reader = _get_ocr_reader()
    for kf in keyframes:
        try:
            results = reader.readtext(kf.path, detail=0, paragraph=True)
            kf.ocr_text = " ".join(results).strip()
        except Exception:
            kf.ocr_text = ""
    return keyframes


# ──────────────────────────────────────────────
# 텍스트 합성
# ──────────────────────────────────────────────

def combine_text(stt_text: str, keyframes: list) -> str:
    """
    STT 전체 텍스트 + 키프레임별 OCR 텍스트를 합성하여
    CLIP 텍스트 인코더 입력용 단일 문자열로 만든다.
    중복 제거 후 최대 300 토큰 수준으로 자른다.
    """
    parts = [stt_text] if stt_text else []
    seen = set()
    for kf in keyframes:
        if kf.ocr_text and kf.ocr_text not in seen:
            parts.append(kf.ocr_text)
            seen.add(kf.ocr_text)

    combined = " | ".join(parts)
    # CLIP 텍스트 토큰 한계(77 토큰) 고려 — 단어 기준 약 200자
    words = combined.split()
    if len(words) > 200:
        combined = " ".join(words[:200]) + "..."
    return combined


# ──────────────────────────────────────────────
# 통합 파이프라인
# ──────────────────────────────────────────────

def _load_cached_result(uid: str, video_path: str, status) -> Optional["PreprocessResult"]:
    """
    이전 실행에서 저장된 preprocess_result.json 을 가능한 한 안전하게 복원한다.
    캐시가 없거나, 키프레임 이미지 파일 중 하나라도 사라졌으면 None 을 반환해
    재계산으로 폴백한다.
    """
    cache_path = PROCESSED_DIR / uid / "preprocess_result.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None

    # 키프레임 이미지가 실제 디스크에 있는지 모두 확인 (앱이 OCR/Plotly 에 그대로 사용)
    kf_data = d.get("keyframes") or []
    keyframes = []
    for k in kf_data:
        p = k.get("path", "")
        if not p or not os.path.exists(p):
            return None
        keyframes.append(
            Keyframe(
                index=int(k["index"]),
                timestamp=float(k["timestamp"]),
                path=p,
                ocr_text=k.get("ocr_text", "") or "",
            )
        )
    if not keyframes:
        return None

    status(f"캐시 hit — preprocess 결과 재사용 ({len(keyframes)} keyframes)")
    return PreprocessResult(
        uid=d["uid"],
        video_path=d.get("video_path", video_path),
        duration=float(d.get("duration", 0.0)),
        fps=float(d.get("fps", 0.0)),
        keyframes=keyframes,
        stt_text=d.get("stt_text", "") or "",
        stt_segments=d.get("stt_segments", []) or [],
        combined_text=d.get("combined_text", "") or "",
        out_dir=d.get("out_dir", str(PROCESSED_DIR / uid)),
    )


def preprocess(
    uid: str,
    video_path: str,
    whisper_model: str = "base",
    scene_threshold: float = 27.0,
    status_cb=None,
    use_cache: bool = True,
) -> PreprocessResult:
    """
    전체 전처리 파이프라인을 실행한다.

    Parameters
    ----------
    uid           : ingest 에서 받은 고유 ID
    video_path    : 영상 파일 경로
    whisper_model : STT 모델 크기
    scene_threshold : 장면 감지 민감도
    status_cb     : 진행 상황 콜백 fn(message: str)
    use_cache     : True 면 preprocess_result.json 이 있을 때 재사용

    Returns
    -------
    PreprocessResult
    """
    out_dir = PROCESSED_DIR / uid
    out_dir.mkdir(exist_ok=True)

    def status(msg):
        if status_cb:
            status_cb(msg)

    # Step 0: 캐시 hit 시 즉시 반환 (Whisper/OCR/키프레임 추출 전부 건너뜀)
    if use_cache:
        cached = _load_cached_result(uid, video_path, status)
        if cached is not None:
            return cached

    # Step 1: 키프레임 추출
    status("키프레임 추출 중...")
    keyframes, fps, duration = extract_keyframes(
        video_path, out_dir, threshold=scene_threshold
    )
    status(f"키프레임 {len(keyframes)}개 추출 완료 ({duration:.1f}초 영상)")

    # Step 2: STT
    status(f"음성 인식 중 (Whisper {whisper_model})...")
    stt_text, stt_segments = run_stt(video_path, model_size=whisper_model)
    status(f"STT 완료 — {len(stt_text)}자")

    # Step 3: OCR
    status("화면 텍스트 인식 중 (OCR)...")
    keyframes = run_ocr_on_keyframes(keyframes)
    ocr_count = sum(1 for kf in keyframes if kf.ocr_text)
    status(f"OCR 완료 — {ocr_count}개 프레임에서 텍스트 발견")

    # 텍스트 합성
    combined = combine_text(stt_text, keyframes)

    result = PreprocessResult(
        uid=uid,
        video_path=video_path,
        duration=duration,
        fps=fps,
        keyframes=keyframes,
        stt_text=stt_text,
        stt_segments=stt_segments,
        combined_text=combined,
        out_dir=str(out_dir),
    )

    # 결과 캐싱
    cache_path = out_dir / "preprocess_result.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

    return result


if __name__ == "__main__":
    import sys
    vid = sys.argv[1] if len(sys.argv) > 1 else None
    if not vid:
        print("usage: python preprocess.py <video_path>")
        sys.exit(1)
    res = preprocess(uid="test_001", video_path=vid, status_cb=print)
    print(f"\n=== 결과 ===")
    print(f"키프레임: {len(res.keyframes)}개")
    print(f"STT: {res.stt_text[:200]}")
    print(f"합성 텍스트: {res.combined_text[:200]}")
