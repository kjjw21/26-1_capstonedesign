"""
scripts/recompute_baseline_from_embeddings.py
score.py 의 캘리브레이션이 바뀌었을 때 baseline_mclip_scores.jsonl 을
다시 굽는다. raw 임베딩 (data/embeddings/*.npz) 에서 frame-level
similarity 를 계산하고 score_video 를 거쳐 영상별 점수와 통계를 산출.

기존 run_baseline_multilingual.py 가 Whisper/OCR/CLIP encoding 까지 다시
도는 것과 달리, 이 스크립트는 이미 추출된 임베딩만 사용하므로 152 영상에
약 2초 걸린다.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.score import score_video
from src.preprocess import Keyframe


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


EMB_DIR = Path("data/embeddings")
NORMAL_MANIFEST = Path("data/manifest/normal.jsonl")
FAKE_MANIFEST = Path("data/manifest/fake.jsonl")
OUT = Path("data/eval/baseline_mclip_scores.jsonl")


def load_jsonl(p: Path):
    items = []
    if not p.exists():
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def main():
    # label 매핑 (vid -> 'normal'|'fake') + manifest 의 file_path
    label_map = {}
    fp_map = {}
    for it in load_jsonl(NORMAL_MANIFEST):
        vid = it.get("video_id")
        if vid:
            label_map[vid] = "normal"
            fp_map[vid] = it.get("file_path", "")
    for it in load_jsonl(FAKE_MANIFEST):
        vid = it.get("fake_id")
        if vid:
            label_map[vid] = "fake"
            fp_map[vid] = it.get("file_path", "")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with OUT.open("w", encoding="utf-8") as out_f:
        for npz in sorted(EMB_DIR.glob("*.npz")):
            vid = npz.stem
            label = label_map.get(vid, str(np.load(npz, allow_pickle=True)["label"]))
            d = np.load(npz, allow_pickle=True)
            img = d["image_embs"].astype(np.float32)
            txt = d["text_emb"].astype(np.float32)

            # 임베딩은 이미 L2 정규화 상태로 저장되었지만 안전을 위해 한 번 더
            img_n = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-12)
            txt_n = txt / (np.linalg.norm(txt) + 1e-12)

            # score_video 에 넘기기 위한 가짜 keyframe 리스트
            # timestamp 는 균등 간격으로 가정 (분포·anomaly 신호 그대로 유지됨)
            n_frames = img_n.shape[0]
            keyframes = [Keyframe(index=i, timestamp=float(i), path="") for i in range(n_frames)]

            vs = score_video(
                uid=vid,
                frame_scores_raw=keyframes,
                image_embeddings=img_n,
                text_embedding=txt_n,
            )

            row = {
                "video_id": vid,
                "file_path": fp_map.get(vid, ""),
                "label_true": label,
                "overall_score": float(vs.overall_score),
                "label_pred": vs.label_en,
                "avg_similarity": float(vs.avg_similarity),
                "min_similarity": float(vs.min_similarity),
                "anomaly_count": int(vs.anomaly_count),
                "n_frames": int(n_frames),
                "whisper": "(cached)",
                "clip": "xlm-roberta-base-ViT-B-32",
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[recompute] {n_written} rows -> {OUT}")


if __name__ == "__main__":
    main()
