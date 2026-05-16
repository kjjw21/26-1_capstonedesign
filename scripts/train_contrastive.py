"""
scripts/train_contrastive.py
Phase B-1: Frozen multilingual CLIP + 학습 가능한 projection MLP.

학습 방식:
  - 영상별 입력: image_mean (512), text_emb (512)  — 모두 다국어 CLIP 임베딩
  - DualEncoder: 두 모달리티를 각각 256차원으로 projection (L2 정규화)
  - 학습된 공간에서 cosine similarity = "정상이라는 확신도"
  - Loss: BCE on (sim * temperature)
       * 정상 페어 → sim 높게
       * 위조 페어 → sim 낮게

평가:
  - 5-fold StratifiedKFold (label_true 기준)
  - 각 fold 에서 전체 AUC, L1 only AUC, L2 only AUC 각각 측정
  - L2 hold-out AUC 가 baseline 0.522 에서 얼마나 오르는지가 핵심 지표

실행:
    python scripts/train_contrastive.py
    python scripts/train_contrastive.py --epochs 200 --lr 5e-4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run_baseline_eval import load_jsonl

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim=512, out_dim=256, hidden=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)


class DualEncoder(nn.Module):
    """이미지·텍스트 임베딩을 공통 공간으로 매핑하고 cos similarity 반환."""
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        self.img_proj = ProjectionMLP(in_dim, out_dim)
        self.txt_proj = ProjectionMLP(in_dim, out_dim)

    def forward(self, img_emb, txt_emb):
        img_p = self.img_proj(img_emb)
        txt_p = self.txt_proj(txt_emb)
        # L2 정규화된 두 벡터의 내적 = cosine similarity
        return (img_p * txt_p).sum(dim=-1)


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_dataset(embeddings_dir: Path, fake_manifest: Path):
    """모든 영상의 (image_mean, text_emb, label, kind, id) 로드."""
    kind_map = {}
    for it in load_jsonl(fake_manifest):
        fid = it.get("fake_id")
        kind = it.get("kind") or "audio_swap"
        if kind == "audio_swap":
            kind = "audio_swap_random"
        if fid:
            kind_map[fid] = kind

    img_feats, txt_feats, labels, kinds, ids = [], [], [], [], []
    for npz in sorted(embeddings_dir.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        img_mean = d["image_embs"].mean(axis=0)
        txt = d["text_emb"]
        label = str(d["label"])
        vid = npz.stem
        img_feats.append(img_mean.astype(np.float32))
        txt_feats.append(txt.astype(np.float32))
        labels.append(1 if label == "normal" else 0)
        kinds.append("normal" if label == "normal" else kind_map.get(vid, "audio_swap_random"))
        ids.append(vid)

    return (
        np.stack(img_feats),
        np.stack(txt_feats),
        np.array(labels, dtype=np.int64),
        kinds,
        ids,
    )


# ──────────────────────────────────────────────
# 학습/평가
# ──────────────────────────────────────────────

def train_one_fold(X_img, X_txt, y, tr_idx, te_idx,
                   epochs=200, lr=1e-3, batch_size=16,
                   temperature=10.0, weight_decay=1e-4,
                   device="cpu", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DualEncoder(in_dim=X_img.shape[1], out_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xi_tr = torch.from_numpy(X_img[tr_idx]).to(device)
    Xt_tr = torch.from_numpy(X_txt[tr_idx]).to(device)
    y_tr = torch.from_numpy(y[tr_idx]).float().to(device)  # 1=normal

    n = len(tr_idx)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        model.train()
        for i in range(0, n, batch_size):
            sel = perm[i: i + batch_size]
            opt.zero_grad()
            sim = model(Xi_tr[sel], Xt_tr[sel])     # in [-1, 1]
            logits = sim * temperature
            loss = F.binary_cross_entropy_with_logits(logits, y_tr[sel])
            loss.backward()
            opt.step()

    model.eval()
    Xi_te = torch.from_numpy(X_img[te_idx]).to(device)
    Xt_te = torch.from_numpy(X_txt[te_idx]).to(device)
    with torch.no_grad():
        sim_te = model(Xi_te, Xt_te).cpu().numpy()  # 정상에서 높음
    # 평가 시 fake = 1 기준으로 점수: -sim 이 크면 fake
    return -sim_te


def evaluate_per_kind(scores, y, kinds):
    """scores 가 클수록 fake. label y=1 이면 normal."""
    out = {}
    fake_target = 1 - y  # 1 = fake
    try:
        out["all"] = roc_auc_score(fake_target, scores)
    except ValueError:
        out["all"] = float("nan")

    kinds = np.array(kinds)
    normal_mask = kinds == "normal"
    for kname, key in [("L1", "audio_swap_random"), ("L2", "audio_swap_topic")]:
        mask_k = kinds == key
        if mask_k.sum() < 2 or normal_mask.sum() < 2:
            out[kname] = float("nan")
            continue
        sub_y = np.concatenate([np.ones(mask_k.sum()), np.zeros(normal_mask.sum())])
        sub_s = np.concatenate([scores[mask_k], scores[normal_mask]])
        try:
            out[kname] = roc_auc_score(sub_y, sub_s)
        except ValueError:
            out[kname] = float("nan")
    return out


def cv_evaluate(X_img, X_txt, y, kinds,
                n_splits=5, seed=42, epochs=200, lr=1e-3, device="cpu"):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    for fold, (tr, te) in enumerate(skf.split(X_img, y), 1):
        scores = train_one_fold(
            X_img, X_txt, y, tr, te,
            epochs=epochs, lr=lr, device=device, seed=seed + fold,
        )
        kinds_te = [kinds[i] for i in te]
        m = evaluate_per_kind(scores, y[te], kinds_te)
        m["fold"] = fold
        fold_results.append(m)
        print(f"  fold {fold}: AUC_all={m['all']:.3f}  AUC_L1={m['L1']:.3f}  AUC_L2={m['L2']:.3f}")
    return fold_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}")

    X_img, X_txt, y, kinds, ids = load_dataset(args.embeddings_dir, args.fake_manifest)
    print(f"[train] n total : {len(y)}")
    print(f"[train] normal  : {(y==1).sum()}")
    print(f"[train] fake    : {(y==0).sum()}")
    from collections import Counter
    print(f"[train] kinds   : {dict(Counter(kinds))}")

    print(f"\n=== {args.n_splits}-fold CV (epochs={args.epochs}, lr={args.lr}) ===")
    folds = cv_evaluate(
        X_img, X_txt, y, kinds,
        n_splits=args.n_splits, seed=args.seed,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    print(f"\n=== 평균 ===")
    for k in ["all", "L1", "L2"]:
        vals = np.array([f[k] for f in folds])
        print(f"  AUC_{k:<3}: mean={np.nanmean(vals):.3f}  std={np.nanstd(vals):.3f}")

    # 비교 — baseline (zero-shot multilingual CLIP)
    print(f"\n=== Baseline 대비 (zero-shot multilingual CLIP) ===")
    print(f"  Baseline  : AUC_all=0.623  AUC_L1=0.722  AUC_L2=0.522")
    L2_mean = float(np.nanmean([f['L2'] for f in folds]))
    delta_L2 = L2_mean - 0.522
    arrow = "↑" if delta_L2 > 0 else "↓"
    print(f"  Ours      : AUC_L2={L2_mean:.3f}  ({arrow} {abs(delta_L2):+.3f})  ← 핵심 지표")


if __name__ == "__main__":
    main()
