"""
scripts/crawl_factcheck.py
팩트체크 채널/플레이리스트에서 정상 영상을 자동 수집한다.

기본 소스:
    JTBC '뉴스룸ㅣ팩트체크' 플레이리스트

사용 예:
    # 기본 소스에서 60~300초 영상 최대 50개 수집
    python scripts/crawl_factcheck.py

    # 추가 채널/플레이리스트도 같이 크롤
    python scripts/crawl_factcheck.py \
        --source "https://www.youtube.com/playlist?list=PL3Eb1N33oAXgQrRBThE4TPSOIR8ZgfSug" \
                 "https://www.youtube.com/channel/UCj3_t5p4L4aFsvdW3uHjnnw" \
        --max-per-source 30

수집 결과:
    data/raw/normal/<video_id>.mp4
    data/manifest/normal.jsonl   (한 줄에 한 영상 메타)
    data/manifest/errors.log     (실패 기록)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yt_dlp
from tqdm import tqdm


# Windows 콘솔에서 한글이 cp949 로 깨지지 않도록 보강
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


DEFAULT_SOURCES = [
    # JTBC 뉴스룸ㅣ팩트체크
    "https://www.youtube.com/playlist?list=PL3Eb1N33oAXgQrRBThE4TPSOIR8ZgfSug",
]


# ──────────────────────────────────────────────
# Manifest I/O
# ──────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> set:
    """기존 manifest 에서 이미 받은 video_id 집합을 반환."""
    if not manifest_path.exists():
        return set()
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                seen.add(entry["video_id"])
            except Exception:
                continue
    return seen


def append_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_error(error_log: Path, vid: str, reason: str) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{vid}\t{reason}\n")


# ──────────────────────────────────────────────
# yt-dlp 호출
# ──────────────────────────────────────────────

def list_videos_in_source(url: str) -> list:
    """채널/플레이리스트의 비디오 메타(flat) 리스트 추출. 다운로드는 안 한다."""
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    entries = info.get("entries") or []
    items = []
    for e in entries:
        if not e:
            continue
        vid = e.get("id")
        if not vid:
            continue
        items.append({
            "video_id": vid,
            "title": e.get("title", ""),
            "url": e.get("url") or f"https://www.youtube.com/watch?v={vid}",
            "duration_hint": e.get("duration"),
        })
    return items


def fetch_full_meta(video_url: str) -> Optional[dict]:
    """단일 영상의 풀 메타데이터(다운로드 X). 길이 확정·필터링용."""
    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(video_url, download=False)
    except Exception:
        return None


def download_video(video_url: str, out_dir: Path, video_id: str) -> Optional[Path]:
    """영상 다운로드. 저장된 파일 경로 반환 (실패 시 None)."""
    out_template = str(out_dir / f"{video_id}.%(ext)s")
    ydl_opts = {
        "outtmpl": out_template,
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "merge_output_format": "mp4",
        "retries": 3,
        "fragment_retries": 3,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(video_url, download=True)
    candidates = list(out_dir.glob(f"{video_id}.*"))
    return candidates[0] if candidates else None


# ──────────────────────────────────────────────
# 메인 크롤 루프
# ──────────────────────────────────────────────

def crawl(
    sources: list,
    max_per_source: int,
    min_sec: float,
    max_sec: float,
    out_dir: Path,
    manifest_path: Path,
    error_log: Path,
    label: str,
    sleep_between: float,
) -> dict:
    seen = load_manifest(manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"downloaded": 0, "skipped_dup": 0, "skipped_filter": 0, "failed": 0}

    for source_url in sources:
        print(f"\n[source] {source_url}")
        try:
            videos = list_videos_in_source(source_url)
        except Exception as e:
            print(f"  ! 소스 메타 추출 실패: {e}")
            continue

        print(f"  {len(videos)}개 항목 발견")
        downloaded_here = 0
        pbar = tqdm(videos, desc="crawl", unit="vid")
        for v in pbar:
            if downloaded_here >= max_per_source:
                break

            vid = v["video_id"]
            if vid in seen:
                stats["skipped_dup"] += 1
                continue

            meta = fetch_full_meta(v["url"])
            if meta is None:
                stats["failed"] += 1
                log_error(error_log, vid, "meta_fetch_failed")
                continue

            duration = meta.get("duration") or 0
            if duration < min_sec or duration > max_sec:
                stats["skipped_filter"] += 1
                continue

            try:
                file_path = download_video(v["url"], out_dir, vid)
                if file_path is None:
                    raise RuntimeError("file not found after download")
            except Exception as e:
                stats["failed"] += 1
                log_error(error_log, vid, f"download_failed: {e}")
                continue

            entry = {
                "video_id": vid,
                "url": v["url"],
                "title": meta.get("title", ""),
                "channel": meta.get("channel") or meta.get("uploader"),
                "upload_date": meta.get("upload_date"),
                "duration": float(duration),
                "file_path": str(file_path),
                "source_url": source_url,
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
                "label": label,
            }
            append_manifest(manifest_path, entry)
            seen.add(vid)
            stats["downloaded"] += 1
            downloaded_here += 1
            pbar.set_postfix_str(f"got={stats['downloaded']}")
            time.sleep(sleep_between)

        pbar.close()

    return stats


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="팩트체크 채널 자동 크롤러")
    p.add_argument(
        "--source",
        nargs="+",
        default=DEFAULT_SOURCES,
        help="채널 또는 플레이리스트 URL. 여러 개 입력 가능.",
    )
    p.add_argument("--max-per-source", type=int, default=50)
    p.add_argument("--min-sec", type=float, default=60.0)
    p.add_argument("--max-sec", type=float, default=300.0)
    p.add_argument("--label", default="normal", help="manifest 에 기록할 라벨")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/normal"))
    p.add_argument("--manifest", type=Path, default=Path("data/manifest/normal.jsonl"))
    p.add_argument("--error-log", type=Path, default=Path("data/manifest/errors.log"))
    p.add_argument("--sleep", type=float, default=1.0, help="영상 사이 대기 시간(초)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[crawl] sources       : {len(args.source)}")
    for s in args.source:
        print(f"   - {s}")
    print(f"[crawl] duration범위  : {args.min_sec:.0f}~{args.max_sec:.0f}초")
    print(f"[crawl] max/source    : {args.max_per_source}")
    print(f"[crawl] out dir       : {args.out_dir}")
    print(f"[crawl] manifest      : {args.manifest}")
    print(f"[crawl] label         : {args.label}")

    stats = crawl(
        sources=args.source,
        max_per_source=args.max_per_source,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        error_log=args.error_log,
        label=args.label,
        sleep_between=args.sleep,
    )

    print("\n=== 결과 ===")
    for k, v in stats.items():
        print(f"  {k:<16}: {v}")
    print(f"\nmanifest: {args.manifest}")
    print(f"errors  : {args.error_log}")


if __name__ == "__main__":
    main()
