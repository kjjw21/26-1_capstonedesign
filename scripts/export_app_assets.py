"""
scripts/export_app_assets.py
Gradio 앱에서 실시간으로 쓰기 위한 두 가지 자산을 저장한다.

1. 학습된 Phase 1 분류기 (StandardScaler + LogisticRegression)
   - 입력: baseline_mclip_scores.jsonl 의 'overall_score','avg_similarity','min_similarity'
   - 출력: data/app/classifier.joblib

2. 정상 corpus 검색 인덱스
   - data/embeddings/*.npz 중 label=='normal' 만 모아
   - vid 리스트 + L2 정규화된 (image_mean, text_emb) 행렬을 .npz 로 저장
   - 영상 메타데이터(title, url, channel)는 별도 jsonl 로
   - 출력: data/app/retrieval_index.npz, data/app/retrieval_meta.jsonl
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


OUT_DIR = Path("data/app")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_KEYS = ["overall_score", "avg_similarity", "min_similarity"]


# ──────────────────────────────────────────────
# 1. 분류기 학습 + 저장
# ──────────────────────────────────────────────

def train_classifier():
    rows = []
    with open("data/eval/baseline_mclip_scores.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    X = np.array([[r[k] for k in FEATURE_KEYS] for r in rows], dtype=np.float32)
    y = np.array([1 if r["label_true"] == "fake" else 0 for r in rows], dtype=np.int64)

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(X), y)
    bundle = {
        "scaler": scaler,
        "clf": clf,
        "feature_keys": FEATURE_KEYS,
        "n_train": len(y),
        "class_balance": {"normal": int((y == 0).sum()), "fake": int((y == 1).sum())},
    }
    out = OUT_DIR / "classifier.joblib"
    joblib.dump(bundle, out)
    print(f"[clf] saved -> {out}")
    print(f"[clf] coefficients: {dict(zip(FEATURE_KEYS, clf.coef_.flatten().tolist()))}")
    print(f"[clf] intercept   : {float(clf.intercept_[0]):.3f}")


# ──────────────────────────────────────────────
# 2. 정상 corpus 검색 인덱스 저장
# ──────────────────────────────────────────────

def build_retrieval_index():
    emb_dir = Path("data/embeddings")
    norm_keys, img_vecs, txt_vecs = [], [], []
    for npz in sorted(emb_dir.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        if str(d["label"]) != "normal":
            continue
        norm_keys.append(npz.stem)
        img_vecs.append(d["image_embs"].mean(axis=0).astype(np.float32))
        txt_vecs.append(d["text_emb"].astype(np.float32))

    img_mat = np.stack(img_vecs)
    img_mat = img_mat / (np.linalg.norm(img_mat, axis=1, keepdims=True) + 1e-12)
    txt_mat = np.stack(txt_vecs)
    txt_mat = txt_mat / (np.linalg.norm(txt_mat, axis=1, keepdims=True) + 1e-12)

    out_npz = OUT_DIR / "retrieval_index.npz"
    np.savez(
        out_npz,
        keys=np.array(norm_keys),
        img_mat=img_mat,
        txt_mat=txt_mat,
    )
    print(f"[idx] saved -> {out_npz}  ({len(norm_keys)} normal videos)")

    # 메타데이터 (title, url, channel) — normal.jsonl 에서 추출
    meta_by_id = {}
    with open("data/manifest/normal.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            vid = it.get("video_id")
            if vid:
                meta_by_id[vid] = {
                    "video_id": vid,
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "channel": it.get("channel"),
                    "upload_date": it.get("upload_date"),
                    "duration": it.get("duration"),
                }

    out_meta = OUT_DIR / "retrieval_meta.jsonl"
    with open(out_meta, "w", encoding="utf-8") as f:
        for vid in norm_keys:
            m = meta_by_id.get(vid, {"video_id": vid})
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"[idx] meta  -> {out_meta}")


if __name__ == "__main__":
    train_classifier()
    build_retrieval_index()
    print("\n자산 저장 완료. src/retrieval.py / src/classifier.py 가 이 파일들을 로드합니다.")
