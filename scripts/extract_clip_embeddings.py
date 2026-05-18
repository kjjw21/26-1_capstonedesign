"""
scripts/extract_clip_embeddings.py
정상·위조 모든 영상에 대해 다국어 CLIP(xlm-roberta-base-ViT-B-32)
임베딩을 추출하여 영상별 npz 로 저장한다.

preprocess 캐시(키프레임·STT·OCR)는 재활용한다.

저장 포맷:
    data/embeddings/<video_id>.npz
        image_embs : (N, 512) — 키프레임별 이미지 임베딩 (L2 정규화)
        text_emb   : (512,)   — combined_text 임베딩 (L2 정규화)
        label      : "normal" or "fake"
        n_frames   : int
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run_baseline_multilingual import (
    embed_images_mclip,
    embed_text_mclip,
    load_preprocess_cache,
    make_uid,
)
from src.preprocess import preprocess
from run_baseline_eval import load_jsonl


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def extract_for_video(file_path: str, scene_threshold: float = 27.0):
    uid = make_uid(file_path)
    cached = load_preprocess_cache(uid)
    if cached:
        keyframes, combined_text = cached
    else:
        prep = preprocess(uid=uid, video_path=file_path, whisper_model="tiny", scene_threshold=scene_threshold)
        keyframes = prep.keyframes
        combined_text = prep.combined_text

    image_paths = [kf.path for kf in keyframes]
    if not image_paths:
        raise RuntimeError("키프레임 0개")

    image_embs = embed_images_mclip(image_paths)
    text_emb = embed_text_mclip(combined_text or "영상")
    return image_embs, text_emb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--normal-manifest", type=Path, default=Path("data/manifest/normal.jsonl"))
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--errors", type=Path, default=Path("data/embeddings/errors.log"))
    p.add_argument("--scene-threshold", type=float, default=27.0)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    normal_items = load_jsonl(args.normal_manifest)
    fake_items = load_jsonl(args.fake_manifest)
    tasks = (
        [(it, "normal", it.get("video_id")) for it in normal_items]
        + [(it, "fake", it.get("fake_id")) for it in fake_items]
    )
    print(f"[extract] 정상: {len(normal_items)}  위조: {len(fake_items)}  합계: {len(tasks)}")

    skipped = 0
    saved = 0
    failed = 0
    for item, label, vid in tqdm(tasks, desc="extract", unit="vid"):
        out = args.out_dir / f"{vid}.npz"
        if out.exists():
            skipped += 1
            continue

        file_path = item.get("file_path") or ""
        if not file_path or not Path(file_path).exists():
            failed += 1
            continue

        try:
            image_embs, text_emb = extract_for_video(file_path, args.scene_threshold)
        except Exception as e:
            failed += 1
            with args.errors.open("a", encoding="utf-8") as ef:
                ef.write(f"{vid}\t{type(e).__name__}\t{e}\n")
            continue

        np.savez(
            out,
            image_embs=image_embs.astype(np.float32),
            text_emb=text_emb.astype(np.float32),
            label=np.array(label),
            n_frames=np.array(image_embs.shape[0]),
        )
        saved += 1

    print(f"\n=== 결과 ===")
    print(f"  saved   : {saved}")
    print(f"  skipped : {skipped}  (이미 존재)")
    print(f"  failed  : {failed}")
    print(f"  out dir : {args.out_dir}")


if __name__ == "__main__":
    main()
