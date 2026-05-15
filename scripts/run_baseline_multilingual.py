"""
scripts/run_baseline_multilingual.py
다국어 CLIP (xlm-roberta-base-ViT-B-32, laion5b 가중치)로 baseline 재평가.

기존 baseline 의 preprocess 캐시(키프레임·STT·OCR)를 재활용하여
임베딩 단계만 다국어 CLIP 으로 다시 계산한다. 영상당 ~10초, 102개 ~20분.

결과:
    data/eval/baseline_mclip_scores.jsonl
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# scripts/, src/ 둘 다 import 가능하게
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import open_clip

from src.preprocess import preprocess, Keyframe
from src.score import score_video

from run_baseline_eval import analyze, load_jsonl, load_done


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


PROCESSED_DIR = Path("C:/cheapfake_data/processed")
MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"


# ──────────────────────────────────────────────
# preprocess 캐시 활용
# ──────────────────────────────────────────────

def make_uid(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()[:12]


def load_preprocess_cache(uid: str):
    """캐시가 있으면 (keyframes, combined_text) 반환, 없으면 None."""
    cache = PROCESSED_DIR / uid / "preprocess_result.json"
    if not cache.exists():
        return None
    with cache.open("r", encoding="utf-8") as f:
        d = json.load(f)
    keyframes = [
        Keyframe(
            index=k["index"],
            timestamp=k["timestamp"],
            path=k["path"],
            ocr_text=k.get("ocr_text", ""),
        )
        for k in d.get("keyframes", [])
    ]
    return keyframes, d.get("combined_text", "")


# ──────────────────────────────────────────────
# 다국어 CLIP (싱글턴)
# ──────────────────────────────────────────────

_mclip = None


def get_mclip():
    global _mclip
    if _mclip is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess_fn = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval().to(device)
        _mclip = (model, preprocess_fn, tokenizer, device)
        print(f"[mclip] {MODEL_NAME} ({PRETRAINED}) on {device}")
    return _mclip


def embed_images_mclip(image_paths, batch_size: int = 8) -> np.ndarray:
    model, preprocess_fn, _, device = get_mclip()
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess_fn(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(preprocess_fn(Image.new("RGB", (224, 224))))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def embed_text_mclip(text: str) -> np.ndarray:
    model, _, tokenizer, device = get_mclip()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]


# ──────────────────────────────────────────────
# 영상 단위 평가
# ──────────────────────────────────────────────

def evaluate_video_mclip(file_path: str, scene_threshold: float = 27.0):
    uid = make_uid(file_path)
    cached = load_preprocess_cache(uid)
    if cached:
        keyframes, combined_text = cached
    else:
        prep = preprocess(
            uid=uid,
            video_path=file_path,
            whisper_model="tiny",
            scene_threshold=scene_threshold,
        )
        keyframes = prep.keyframes
        combined_text = prep.combined_text

    image_paths = [kf.path for kf in keyframes]
    if not image_paths:
        raise RuntimeError("키프레임 0개")
    image_embs = embed_images_mclip(image_paths)
    text_emb = embed_text_mclip(combined_text or "영상")
    vs = score_video(uid, keyframes, image_embs, text_emb)
    return vs


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="다국어 CLIP baseline 재평가")
    p.add_argument("--normal-manifest", type=Path, default=Path("data/manifest/normal.jsonl"))
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/eval/baseline_mclip_scores.jsonl"))
    p.add_argument("--errors", type=Path, default=Path("data/eval/errors_mclip.log"))
    p.add_argument("--scene-threshold", type=float, default=27.0)
    p.add_argument("--analyze-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.analyze_only:
        analyze(args.out)
        return

    normal_items = load_jsonl(args.normal_manifest)
    fake_items = load_jsonl(args.fake_manifest)
    tasks = (
        [(it, "normal", it.get("video_id")) for it in normal_items]
        + [(it, "fake", it.get("fake_id")) for it in fake_items]
    )
    print(f"[mclip] 정상: {len(normal_items)}  위조: {len(fake_items)}  합계: {len(tasks)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(args.out)
    print(f"[mclip] 이미 처리됨: {len(done)} — skip")

    with args.out.open("a", encoding="utf-8") as out_f:
        for item, label, vid in tqdm(tasks, desc="mclip", unit="vid"):
            if vid in done:
                continue
            file_path = item.get("file_path") or ""
            if not file_path or not Path(file_path).exists():
                continue

            try:
                vs = evaluate_video_mclip(file_path, args.scene_threshold)
            except Exception as e:
                with args.errors.open("a", encoding="utf-8") as ef:
                    ef.write(f"{vid}\t{type(e).__name__}\t{e}\n")
                continue

            row = {
                "video_id": vid,
                "file_path": file_path,
                "label_true": label,
                "overall_score": float(vs.overall_score),
                "label_pred": vs.label_en,
                "avg_similarity": float(vs.avg_similarity),
                "min_similarity": float(vs.min_similarity),
                "anomaly_count": int(vs.anomaly_count),
                "n_frames": len(vs.frame_scores),
                "backend": MODEL_NAME,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    analyze(args.out)


if __name__ == "__main__":
    main()
