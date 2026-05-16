"""
scripts/make_cheapfake.py
정상 영상 두 개에서 비디오·오디오를 교차 결합해 Cheapfake 학습용 데이터를 생성한다.

생성되는 결과:
    data/raw/fake/<fake_id>.mp4   — A의 비디오 + B의 오디오
    data/manifest/fake.jsonl      — 한 줄에 한 fake 메타 (소스 영상 id 포함)

사용 예:
    python scripts/make_cheapfake.py --n 50 --max-uses 4

요건:
    PATH 에 ffmpeg 가 있어야 함. 또는 환경변수 FFMPEG_BIN 으로 절대경로 지정.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ──────────────────────────────────────────────
# I/O 유틸
# ──────────────────────────────────────────────

def load_normal_manifest(manifest_path: Path) -> list:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 없음: {manifest_path}")
    items = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def resolve_ffmpeg() -> str:
    """ffmpeg 실행 파일 경로를 찾는다. FFMPEG_BIN → PATH → CONTEXT 기재 경로 순."""
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    bin_path = shutil.which("ffmpeg")
    if bin_path:
        return bin_path
    fallback = Path(r"C:\ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(
        "ffmpeg 실행 파일을 찾을 수 없습니다. "
        "PATH 에 추가하거나 FFMPEG_BIN 환경변수를 설정하세요."
    )


# ──────────────────────────────────────────────
# 합성
# ──────────────────────────────────────────────

def make_one(
    ffmpeg_bin: str,
    video_src: Path,
    audio_src: Path,
    out_path: Path,
) -> None:
    """A의 비디오 + B의 오디오 → out_path 로 합성."""
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(video_src),
        "-i", str(audio_src),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",          # 비디오는 그대로 (빠름)
        "-c:a", "aac",           # 오디오 AAC 재인코딩 (호환성)
        "-shortest",
        "-loglevel", "error",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# ──────────────────────────────────────────────
# 페어 샘플링
# ──────────────────────────────────────────────

def sample_pairs(
    items: list,
    n: int,
    max_uses: int,
    rng: random.Random,
) -> list:
    """
    L1 (random) — (video_src, audio_src) 페어 n개를 뽑는다.
    - 같은 페어 중복 방지
    - 영상 1개의 사용 횟수가 max_uses 를 넘지 않게 제한
    - 같은 채널 페어는 50% 확률로 reroll (다양성 위해)
    """
    pairs = []
    usage = defaultdict(int)
    seen = set()
    max_attempts = max(n * 20, 200)
    attempts = 0

    while len(pairs) < n and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(items, 2)
        key = (a["video_id"], b["video_id"])
        if key in seen:
            continue
        if usage[a["video_id"]] >= max_uses or usage[b["video_id"]] >= max_uses:
            continue
        if a.get("channel") == b.get("channel") and rng.random() < 0.5:
            continue

        seen.add(key)
        usage[a["video_id"]] += 1
        usage[b["video_id"]] += 1
        pairs.append((a, b))

    return pairs


def sample_pairs_topic(
    items: list,
    n: int,
    max_uses: int,
    rng: random.Random,
    embeddings_dir: Path,
    min_sim: float,
) -> list:
    """
    L2 (same-topic) — combined_text 임베딩 유사도가 min_sim 이상인 페어만 사용.
    의미적으로 비슷한 두 영상을 audio-swap → 실제 SNS cheapfake 와 더 가까운 분포.

    items 의 각 영상에 대응하는 임베딩이 `<embeddings_dir>/<video_id>.npz` 에
    있어야 한다. 페어를 (sim 내림차순) 정렬한 뒤 max_uses 제약을 두고 n개 선택.
    """
    import numpy as np

    # 1) 사용 가능한 영상의 text_emb 로드
    enriched = []
    for it in items:
        vid = it.get("video_id")
        npz = embeddings_dir / f"{vid}.npz"
        if not npz.exists():
            continue
        try:
            d = np.load(npz, allow_pickle=True)
            text_emb = d["text_emb"]
        except Exception:
            continue
        enriched.append((it, text_emb))

    if len(enriched) < 2:
        print(f"  ! 임베딩이 부족합니다 ({len(enriched)}개). "
              f"먼저 scripts/extract_clip_embeddings.py 를 돌리세요.")
        return []

    # 2) 페어 sim 행렬 (대칭, 상삼각만)
    embs = np.stack([e for _, e in enriched]).astype(np.float32)
    sim_matrix = embs @ embs.T   # 이미 L2 정규화된 임베딩

    candidates = []
    nE = len(enriched)
    for i in range(nE):
        for j in range(i + 1, nE):
            s = float(sim_matrix[i, j])
            if s >= min_sim:
                candidates.append((s, i, j))

    # 3) sim 내림차순으로 정렬 + 약간의 무작위성(jitter)
    candidates.sort(key=lambda x: (-x[0], rng.random()))

    # 4) max_uses 제약하에 페어 n개 선택. (i->j) 와 (j->i) 양방향을 고려
    pairs = []
    usage = defaultdict(int)
    seen = set()
    for s, i, j in candidates:
        if len(pairs) >= n:
            break
        # 무작위로 (i->j) 또는 (j->i) 선택 — 어느 쪽이 비디오 source 인지
        if rng.random() < 0.5:
            a, b = enriched[i][0], enriched[j][0]
        else:
            a, b = enriched[j][0], enriched[i][0]

        key = (a["video_id"], b["video_id"])
        if key in seen:
            continue
        if usage[a["video_id"]] >= max_uses or usage[b["video_id"]] >= max_uses:
            continue

        seen.add(key)
        usage[a["video_id"]] += 1
        usage[b["video_id"]] += 1
        # sim 정보를 entry 에 실어두기 위해 a 사본에 첨부 (manifest 에 기록됨)
        pairs.append((a, b, s))

    return pairs


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="FFmpeg 기반 Cheapfake 자동 생성기")
    p.add_argument(
        "--mode",
        default="random",
        choices=["random", "topic"],
        help="random=L1 무작위 페어, topic=L2 임베딩 유사도 기반 같은 주제 페어",
    )
    p.add_argument("--n", type=int, default=50, help="만들 cheapfake 개수")
    p.add_argument("--max-uses", type=int, default=4, help="영상 1개의 최대 사용 횟수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-sim", type=float, default=0.78,
                   help="topic 모드에서 페어로 인정할 최소 코사인 유사도")
    p.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="topic 모드에서 사용할 video_id.npz 임베딩 디렉토리",
    )
    p.add_argument(
        "--manifest-in",
        type=Path,
        default=Path("data/manifest/normal.jsonl"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/fake"))
    p.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("data/manifest/fake.jsonl"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    ffmpeg_bin = resolve_ffmpeg()
    print(f"[fake] mode       : {args.mode}")
    print(f"[fake] ffmpeg     : {ffmpeg_bin}")

    items = load_normal_manifest(args.manifest_in)
    print(f"[fake] 정상 영상 풀: {len(items)}개")
    if len(items) < 2:
        print("정상 영상이 2개 미만이라 합성할 수 없습니다. 먼저 크롤러를 돌리세요.")
        return

    # 페어 샘플링 — mode 에 따라 분기
    if args.mode == "random":
        raw_pairs = sample_pairs(items, args.n, args.max_uses, rng)
        pairs = [(a, b, None) for a, b in raw_pairs]
        kind = "audio_swap_random"     # 기존 'audio_swap' 과 구분되게 명시
    else:  # topic
        pairs = sample_pairs_topic(
            items, args.n, args.max_uses, rng,
            embeddings_dir=args.embeddings_dir, min_sim=args.min_sim,
        )
        kind = "audio_swap_topic"

    print(f"[fake] 생성 페어   : {len(pairs)}개 (목표 {args.n})")
    if not pairs:
        print("페어가 없습니다. 종료.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0
    with args.manifest_out.open("a", encoding="utf-8") as mf:
        for a, b, sim in tqdm(pairs, desc="compose", unit="vid"):
            fake_id = f"fake_{kind}_{a['video_id']}_{b['video_id']}"
            out_path = args.out_dir / f"{fake_id}.mp4"
            if out_path.exists():
                skipped += 1
                continue

            video_src = Path(a["file_path"])
            audio_src = Path(b["file_path"])
            if not video_src.exists() or not audio_src.exists():
                failed += 1
                continue

            try:
                make_one(ffmpeg_bin, video_src, audio_src, out_path)
            except subprocess.CalledProcessError:
                failed += 1
                continue

            entry = {
                "fake_id": fake_id,
                "video_source": a["video_id"],
                "video_source_title": a.get("title", ""),
                "audio_source": b["video_id"],
                "audio_source_title": b.get("title", ""),
                "file_path": str(out_path),
                "duration": min(a.get("duration", 0), b.get("duration", 0)),
                "channel_video": a.get("channel"),
                "channel_audio": b.get("channel"),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "label": "fake",
                "kind": kind,
                "pair_similarity": float(sim) if sim is not None else None,
            }
            mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            succeeded += 1

    print(f"\n=== 결과 ===")
    print(f"  succeeded : {succeeded}")
    print(f"  failed    : {failed}")
    print(f"  skipped   : {skipped}  (out 파일 이미 존재)")
    print(f"  out dir   : {args.out_dir}")
    print(f"  manifest  : {args.manifest_out}")


if __name__ == "__main__":
    main()
