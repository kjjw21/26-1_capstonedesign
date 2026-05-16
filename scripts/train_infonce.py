"""
scripts/train_infonce.py
Stage B-1 (재시도): 정통 contrastive learning (InfoNCE) 으로
image-text projection MLP 학습.

학습 방식:
  - 정상 영상만 사용 (positive=일치, negative=in-batch 다른 영상)
  - CLIP 원래 학습 방식 (symmetric cross-entropy across diagonals)
  - 위조는 평가 단계에서만 활용 (이게 zero-shot test)

평가:
  - 5-fold split (정상만 기준 stratified):
      4 fold = train, 1 fold = test (정상 ~10개)
  - test fold 정상 + 모든 위조에 대해 (img_proj, txt_proj) sim 계산
  - sim 작을수록 fake → 1-sim 을 fake score 로 사용
  - L1/L2/L3 ROC-AUC 분리

실행:
    python scripts/train_infonce.py
    python scripts/train_infonce.py --epochs 300 --batch-size 32 --temperature 0.07
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
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run_baseline_eval import load_jsonl
from train_contrastive import DualEncoder, load_dataset


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ──────────────────────────────────────────────
# InfoNCE 학습
# ──────────────────────────────────────────────

def info_nce_loss(img_p: torch.Tensor, txt_p: torch.Tensor, temperature: float):
    """CLIP 원래 학습 방식: diagonal 이 positive, off-diagonal 이 negative.
    img_p, txt_p 는 이미 L2 정규화된 (B, D) 텐서."""
    logits = img_p @ txt_p.T / temperature   # (B, B)
    B = img_p.shape[0]
    targets = torch.arange(B, device=img_p.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_i2t + loss_t2i)


def train_one_fold(
    X_img_train, X_txt_train,
    X_img_test_normal, X_txt_test_normal,
    X_img_fake, X_txt_fake,
    fake_kinds,
    epochs: int = 300,
    lr: float = 1e-3,
    batch_size: int = 32,
    temperature: float = 0.07,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DualEncoder(in_dim=X_img_train.shape[1], out_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xi_tr = torch.from_numpy(X_img_train).to(device)
    Xt_tr = torch.from_numpy(X_txt_train).to(device)
    n = len(Xi_tr)

    for ep in range(epochs):
        perm = np.random.permutation(n)
        model.train()
        for i in range(0, n, batch_size):
            sel = perm[i: i + batch_size]
            if len(sel) < 4:   # 너무 작은 batch 는 contrastive 의미 약함, skip
                continue
            sel_t = torch.from_numpy(sel).long().to(device)
            opt.zero_grad()
            img_p = model.img_proj(Xi_tr[sel_t])
            txt_p = model.txt_proj(Xt_tr[sel_t])
            loss = info_nce_loss(img_p, txt_p, temperature)
            loss.backward()
            opt.step()

    # 평가: test fold 정상 + 모든 위조에 대해 sim 계산
    model.eval()
    with torch.no_grad():
        # 정상 test fold
        img_p_n = model.img_proj(torch.from_numpy(X_img_test_normal).to(device))
        txt_p_n = model.txt_proj(torch.from_numpy(X_txt_test_normal).to(device))
        sim_normal = (img_p_n * txt_p_n).sum(dim=-1).cpu().numpy()
        # fake
        img_p_f = model.img_proj(torch.from_numpy(X_img_fake).to(device))
        txt_p_f = model.txt_proj(torch.from_numpy(X_txt_fake).to(device))
        sim_fake = (img_p_f * txt_p_f).sum(dim=-1).cpu().numpy()

    # fake score = -sim (작을수록 더 위조)
    n_fake = len(sim_fake)
    n_normal = len(sim_normal)
    fake_target = np.concatenate([np.ones(n_fake), np.zeros(n_normal)])
    fake_score = np.concatenate([-sim_fake, -sim_normal])

    # 전체 AUC
    try:
        auc_all = roc_auc_score(fake_target, fake_score)
    except ValueError:
        auc_all = float("nan")

    # kind별 AUC
    out = {"all": auc_all}
    fake_kinds = np.array(fake_kinds)
    for kname, key in [("L1", "audio_swap_random"),
                       ("L2", "audio_swap_topic"),
                       ("L3", "audio_swap_topic_strong")]:
        mask = fake_kinds == key
        if mask.sum() < 2 or n_normal < 2:
            out[kname] = float("nan")
            continue
        sub_y = np.concatenate([np.ones(mask.sum()), np.zeros(n_normal)])
        sub_s = np.concatenate([-sim_fake[mask], -sim_normal])
        try:
            out[kname] = roc_auc_score(sub_y, sub_s)
        except ValueError:
            out[kname] = float("nan")
    return out


# ──────────────────────────────────────────────
# CV
# ──────────────────────────────────────────────

def cv_evaluate(X_img, X_txt, y, kinds,
                n_splits=5, seed=42, epochs=300, lr=1e-3,
                batch_size=32, temperature=0.07, device="cpu"):
    is_normal = y == 1
    normal_idx = np.where(is_normal)[0]
    fake_idx = np.where(~is_normal)[0]
    fake_kinds_full = [kinds[i] for i in fake_idx]
    X_img_fake_all = X_img[fake_idx]
    X_txt_fake_all = X_txt[fake_idx]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_results = []
    for fold, (tr_n, te_n) in enumerate(kf.split(normal_idx), 1):
        tr_global = normal_idx[tr_n]
        te_global = normal_idx[te_n]
        r = train_one_fold(
            X_img[tr_global], X_txt[tr_global],
            X_img[te_global], X_txt[te_global],
            X_img_fake_all, X_txt_fake_all, fake_kinds_full,
            epochs=epochs, lr=lr, batch_size=batch_size,
            temperature=temperature, device=device, seed=seed + fold,
        )
        r["fold"] = fold
        fold_results.append(r)
        print(f"  fold {fold}: AUC_all={r['all']:.3f}  L1={r['L1']:.3f}  L2={r['L2']:.3f}  L3={r['L3']:.3f}")
    return fold_results


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature (CLIP 기본값 0.07)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[infonce] device      : {device}")
    print(f"[infonce] epochs      : {args.epochs}")
    print(f"[infonce] batch_size  : {args.batch_size}")
    print(f"[infonce] temperature : {args.temperature}")

    X_img, X_txt, y, kinds, ids = load_dataset(args.embeddings_dir, args.fake_manifest)
    print(f"[infonce] n total : {len(y)}  (normal={(y==1).sum()}, fake={(y==0).sum()})")
    from collections import Counter
    print(f"[infonce] kinds   : {dict(Counter(kinds))}")

    print(f"\n=== {args.n_splits}-fold CV (정상만 학습, 정상 test fold + 전체 위조 평가) ===")
    folds = cv_evaluate(
        X_img, X_txt, y, kinds,
        n_splits=args.n_splits, seed=args.seed,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, temperature=args.temperature,
        device=device,
    )

    print(f"\n=== 평균 ===")
    for k in ["all", "L1", "L2", "L3"]:
        vals = np.array([f[k] for f in folds])
        print(f"  AUC_{k:<3}: mean={np.nanmean(vals):.3f}  std={np.nanstd(vals):.3f}")

    print(f"\n=== Baseline 대비 (zero-shot multilingual CLIP) ===")
    print(f"  Baseline  : AUC_L1=0.722  AUC_L2=0.522  AUC_L3=0.405")
    L1 = float(np.nanmean([f['L1'] for f in folds]))
    L2 = float(np.nanmean([f['L2'] for f in folds]))
    L3 = float(np.nanmean([f['L3'] for f in folds]))
    print(f"  InfoNCE   : AUC_L1={L1:.3f}  AUC_L2={L2:.3f}  AUC_L3={L3:.3f}")
    print(f"  Δ         : "
          f"{L1-0.722:+.3f} / {L2-0.522:+.3f} / {L3-0.405:+.3f}")


if __name__ == "__main__":
    main()
