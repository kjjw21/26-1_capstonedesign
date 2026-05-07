"""
ingest.py
입력 수신 모듈: 로컬 파일 또는 URL로부터 영상을 받아 data/raw/ 에 저장하고
기본 메타데이터를 반환한다.
"""

import os
import shutil
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime

import yt_dlp


RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _make_uid(source: str) -> str:
    """소스 문자열(경로 또는 URL) 기반 짧은 고유 ID 생성."""
    h = hashlib.md5(source.encode()).hexdigest()[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{h}"


def from_file(filepath: str) -> dict:
    """
    로컬 영상 파일을 data/raw/ 로 복사한다.

    Returns
    -------
    dict with keys:
        video_path  : str  — data/raw/ 내 복사된 경로
        uid         : str  — 고유 식별자
        source_type : 'file'
        original    : str  — 원본 경로
        title       : str  — 파일명 (확장자 제외)
    """
    src = Path(filepath)
    if not src.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")

    uid = _make_uid(str(src))
    dst = RAW_DIR / f"{uid}{src.suffix}"
    shutil.copy2(src, dst)

    return {
        "video_path": str(dst),
        "uid": uid,
        "source_type": "file",
        "original": str(src),
        "title": src.stem,
        "url": None,
        "description": "",
        "upload_date": None,
        "channel": None,
    }


def from_url(url: str, progress_cb=None) -> dict:
    """
    yt-dlp 로 URL에서 영상을 다운로드한다.
    YouTube, TikTok, Instagram Reels, Twitter/X 등 지원.

    Parameters
    ----------
    url         : 다운로드할 영상 URL
    progress_cb : 진행률 콜백 (optional). dict 인자를 받는 callable.

    Returns
    -------
    dict (from_file 과 동일 키 + 'url', 'description', 'upload_date', 'channel')
    """
    uid = _make_uid(url)
    out_template = str(RAW_DIR / f"{uid}.%(ext)s")

    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    if progress_cb:
        ydl_opts["progress_hooks"] = [progress_cb]

    meta = {}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        meta = {
            "title": info.get("title", uid),
            "description": info.get("description", ""),
            "upload_date": info.get("upload_date"),
            "channel": info.get("channel") or info.get("uploader"),
        }

    # 다운로드된 파일 탐색 (확장자가 yt-dlp 에 의해 결정됨)
    candidates = list(RAW_DIR.glob(f"{uid}.*"))
    if not candidates:
        raise RuntimeError(f"다운로드 실패: {url}")

    video_path = str(candidates[0])
    return {
        "video_path": video_path,
        "uid": uid,
        "source_type": "url",
        "original": url,
        "url": url,
        **meta,
    }


def ingest(source: str, progress_cb=None) -> dict:
    """
    source 가 http/https 로 시작하면 URL, 아니면 파일로 처리한다.
    편의 래퍼 함수.
    """
    if source.startswith("http://") or source.startswith("https://"):
        return from_url(source, progress_cb=progress_cb)
    return from_file(source)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python ingest.py <file_or_url>")
        sys.exit(1)
    result = ingest(sys.argv[1])
    print(result)
