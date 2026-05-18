"""
scripts/train_mlp_classifier.py
다국어 CLIP 임베딩 위에 작은 MLP 를 학습한다.

Feature 구성 (영상당):
    [image_mean (512), text_emb (512), image_mean ⊙ text_emb (512)]  = 1536 차원

MLP:
    1536 -> 128 -> 1 (sigmoid)
    ReLU + Dropout 0.3, BCE loss, Adam(lr=1e-3, weight_decay=1e-4)

평가:
    5-fold StratifiedKFold cross-validation
    + 70/30 hold-out final report

전제: scripts/extract_clip_embeddings.py 가 먼저 실행되어
data/embeddings/<video_id>.npz 가 채워져 있어야 한다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_video_features(embeddings_dir: Path, feature_mode: str):
    """모든 영상 npz 를 읽어 (X, y, ids) 로 반환."""
    X = []
    y = []
    ids = []
    for npz_path in sorted(embeddings_dir.glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        image_embs = d["image_embs"]      # (N, 512)
        text_emb = d["text_emb"]          # (512,)
        label = str(d["label"])
        vid = npz_path.stem

        img_mean = image_embs.mean(axis=0)
        if feature_mode == "img_only":
            feat = img_mean
        elif feature_mode == "text_only":
            feat = text_emb
        elif feature_mode == "concat":
            feat = np.concatenate([img_mean, text_emb])
        elif feature_mode == "full":
            prod = img_mean * text_emb
            feat = np.concatenate([img_mean, text_emb, prod])
        else:
            raise ValueError(feature_mode)

        X.append(feat.astype(np.float32))
        y.append(1 if label == "fake" else 0)
        ids.append(vid)

    return np.stack(X), np.array(y, dtype=np.int64), ids


# ──────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logit


def train_one_fold(
    Xtr, ytr, Xte, yte,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    seed: int = 42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = Xtr.shape[1]
    model = MLP(in_dim).cpu()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).float()
    Xte_t = torch.from_numpy(Xte).float()

    n = len(Xtr_t)
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        model.train()
        for i in range(0, n, batch_size):
            sel = idx[i : i + batch_size]
            opt.zero_grad()
            logits = model(Xtr_t[sel])
            loss = loss_fn(logits, ytr_t[sel])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(Xte_t)).numpy()
    pred = (prob >= 0.5).astype(np.int64)
    return prob, pred


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def cv_evaluate(X, y, feature_mode: str, n_splits: int = 5, seed: int = 42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics = {"auc": [], "f1": [], "prec": [], "rec": [], "acc": []}
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        prob, pred = train_one_fold(X[tr], y[tr], X[te], y[te], seed=seed + fold)
        metrics["auc"].append(roc_auc_score(y[te], prob))
        metrics["f1"].append(f1_score(y[te], pred, zero_division=0))
        metrics["prec"].append(precision_score(y[te], pred, zero_division=0))
        metrics["rec"].append(recall_score(y[te], pred, zero_division=0))
        metrics["acc"].append((pred == y[te]).mean())
    return metrics


def holdout_evaluate(X, y, seed: int = 42):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    prob, pred = train_one_fold(Xtr, ytr, Xte, yte, seed=seed)
    return Xtr, Xte, ytr, yte, prob, pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument(
        "--feature-mode",
        default="full",
        choices=["img_only", "text_only", "concat", "full"],
        help="full = [img_mean, text, img_mean*text]  (default)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-ablation", action="store_true",
                   help="기본 feature_mode 만 학습. 켜져있으면 4가지 ablation 동시 수행 (skip ablation if set)")
    return p.parse_args()


def main():
    args = parse_args()

    # Ablation: 4가지 모드 모두 학습해 비교 표를 만든다 (기본 동작).
    modes = [args.feature_mode] if args.no_ablation else ["img_only", "text_only", "concat", "full"]

    print(f"{'feature_mode':<14}  {'in_dim':>7}  {'AUC':>14}  {'F1':>14}  {'Prec':>14}  {'Rec':>14}")
    print("-" * 84)
    results = {}
    for mode in modes:
        X, y, ids = load_video_features(args.embeddings_dir, mode)
        m = cv_evaluate(X, y, feature_mode=mode, seed=args.seed)
        results[mode] = (X, y, m)
        print(
            f"{mode:<14}  {X.shape[1]:>7}  "
            f"{np.mean(m['auc']):.3f} ± {np.std(m['auc']):.3f}  "
            f"{np.mean(m['f1']):.3f} ± {np.std(m['f1']):.3f}  "
            f"{np.mean(m['prec']):.3f} ± {np.std(m['prec']):.3f}  "
            f"{np.mean(m['rec']):.3f} ± {np.std(m['rec']):.3f}"
        )

    # 최종: 가장 좋은 mode 로 70/30 hold-out
    best_mode = max(results, key=lambda m: np.mean(results[m][2]["auc"]))
    X, y, _ = results[best_mode]
    print(f"\n=== Hold-out (70/30) on best mode: {best_mode} ===")
    Xtr, Xte, ytr, yte, prob, pred = holdout_evaluate(X, y, seed=args.seed)
    print(classification_report(yte, pred, target_names=["normal", "fake"], digits=3))
    print(f"confusion matrix:\n{confusion_matrix(yte, pred)}")
    print(f"ROC-AUC: {roc_auc_score(yte, prob):.3f}")


if __name__ == "__main__":
    main()
