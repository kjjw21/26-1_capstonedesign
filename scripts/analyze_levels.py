"""
scripts/analyze_levels.py
baseline 평가 결과를 위조 난이도 레벨(L1/L2/...)별로 분석한다.

수행 분석:
  1. kind 별 점수 분포 (mean/std/median/range)
  2. 분류 성능 (normal vs L1 only / vs L2 only / vs all fakes)
       - 임계값 sweep, 최적 F1, ROC-AUC
  3. L1 과 L2 점수 비교 (Mann-Whitney U test)

실행:
    python scripts/analyze_levels.py                  # 기본: mclip 결과
    python scripts/analyze_levels.py --eval data/eval/baseline_scores.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_jsonl(path: Path):
    items = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def build_kind_map(fake_manifest: Path) -> dict:
    """fake_id → kind 매핑. 기존 'audio_swap' 은 'audio_swap_random' 으로 보정."""
    mapping = {}
    for it in load_jsonl(fake_manifest):
        fid = it.get("fake_id")
        kind = it.get("kind") or "audio_swap"
        if kind == "audio_swap":
            kind = "audio_swap_random"
        if fid:
            mapping[fid] = kind
    return mapping


def stats(name, arr):
    if not arr:
        print(f"  {name:<22}: (empty)")
        return
    a = np.array(arr, dtype=np.float64)
    print(
        f"  {name:<22}: n={len(arr):>3}  mean={a.mean():6.2f}  std={a.std():5.2f}  "
        f"median={np.median(a):6.2f}  range=[{a.min():.2f}, {a.max():.2f}]"
    )


def sweep_thresholds(normal_scores, fake_scores, thresholds=None):
    if thresholds is None:
        thresholds = [40, 45, 50, 55, 60, 65, 70]
    print(f"  {'thresh':>6}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'acc':>6}")
    best = None
    for thresh in thresholds:
        tp = sum(1 for s in fake_scores if s >= thresh)
        fn = len(fake_scores) - tp
        fp = sum(1 for s in normal_scores if s >= thresh)
        tn = len(normal_scores) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        print(f"  {thresh:>6}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {acc:>6.3f}")
        if best is None or f1 > best[1]:
            best = (thresh, f1, prec, rec, acc)
    return best


def compute_auc(normal_scores, fake_scores):
    try:
        from sklearn.metrics import roc_auc_score
        y_true = [0] * len(normal_scores) + [1] * len(fake_scores)
        y_score = list(normal_scores) + list(fake_scores)
        return roc_auc_score(y_true, y_score)
    except Exception:
        return float("nan")


def mann_whitney(a, b):
    """L1 과 L2 점수 분포가 통계적으로 다른지 검정."""
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return u, p
    except ImportError:
        return None, None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--eval",
        type=Path,
        default=Path("data/eval/baseline_mclip_scores.jsonl"),
    )
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    return p.parse_args()


def main():
    args = parse_args()
    rows = load_jsonl(args.eval)
    if not rows:
        print(f"평가 결과 없음: {args.eval}")
        return

    kind_map = build_kind_map(args.fake_manifest)

    # row 에 kind 부여
    for r in rows:
        if r["label_true"] == "normal":
            r["_kind"] = "normal"
        else:
            r["_kind"] = kind_map.get(r["video_id"]) or "audio_swap_random"

    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["_kind"], []).append(r["overall_score"])

    print(f"\n=== 평가 파일: {args.eval} ===")
    print(f"총 영상: {len(rows)}개\n")

    print(f"[점수 분포 (kind별)]")
    for kind in ["normal", "audio_swap_random", "audio_swap_topic"]:
        stats(kind, by_kind.get(kind, []))

    normal_scores = by_kind.get("normal", [])
    l1_scores = by_kind.get("audio_swap_random", [])
    l2_scores = by_kind.get("audio_swap_topic", [])
    all_fake = l1_scores + l2_scores

    # 시나리오별 분류 성능
    for scenario_name, fakes in [
        ("normal vs L1 (random)", l1_scores),
        ("normal vs L2 (topic)", l2_scores),
        ("normal vs all fakes", all_fake),
    ]:
        if not fakes:
            continue
        print(f"\n[{scenario_name}]  n_normal={len(normal_scores)}  n_fake={len(fakes)}")
        best = sweep_thresholds(normal_scores, fakes)
        auc = compute_auc(normal_scores, fakes)
        if best:
            print(
                f"  최적 F1 thresh={best[0]}  f1={best[1]:.3f}  "
                f"prec={best[2]:.3f}  rec={best[3]:.3f}  acc={best[4]:.3f}"
            )
        print(f"  ROC-AUC = {auc:.3f}")

    # L1 vs L2 차이 검정
    if l1_scores and l2_scores:
        print(f"\n[L1 vs L2 점수 분포 차이]")
        l1 = np.array(l1_scores)
        l2 = np.array(l2_scores)
        print(f"  L1 mean = {l1.mean():.2f},  L2 mean = {l2.mean():.2f},  차이 = {l1.mean() - l2.mean():+.2f}")
        u, p = mann_whitney(l1_scores, l2_scores)
        if u is not None:
            print(f"  Mann-Whitney U test: U={u:.1f}, p={p:.4f}")
            if p < 0.05:
                print(f"  ⇒ 두 분포는 통계적으로 유의미하게 다름 (p<0.05)")
            else:
                print(f"  ⇒ 두 분포의 차이가 통계적으로 유의미하지 않음 (p>=0.05)")


if __name__ == "__main__":
    main()
