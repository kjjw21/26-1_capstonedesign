"""
scripts/run_baseline_eval.py
정상·위조 데이터셋을 기존 zero-shot CLIP 파이프라인으로 평가하여
점수 분포와 분류 성능(Precision/Recall/F1/ROC-AUC)을 측정한다.

기본 설정: Whisper tiny + CLIP ViT-B/32 (CPU 환경 속도 우선)

실행:
    python scripts/run_baseline_eval.py
    python scripts/run_baseline_eval.py --whisper base --clip ViT-L/14    # 정확 설정

결과:
    data/eval/baseline_scores.jsonl    — 영상별 점수 (append, 중복 skip)
    data/eval/errors.log               — 처리 실패 영상 기록
    + 콘솔에 분석 통계 출력
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# src 패키지 임포트 경로
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import preprocess
from src.embed import embed_images, embed_texts
from src.score import score_video


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ──────────────────────────────────────────────
# 영상 단위 평가
# ──────────────────────────────────────────────

def make_uid(file_path: str) -> str:
    return hashlib.md5(file_path.encode()).hexdigest()[:12]


def evaluate_video(file_path: str, whisper_model: str, clip_model: str, scene_threshold: float):
    """pipeline 전체 단계를 ingest 없이 호출 (영상 복사 우회)."""
    uid = make_uid(file_path)
    prep = preprocess(
        uid=uid,
        video_path=file_path,
        whisper_model=whisper_model,
        scene_threshold=scene_threshold,
    )
    image_paths = [kf.path for kf in prep.keyframes]
    if not image_paths:
        raise RuntimeError("키프레임 0개")
    image_embs = embed_images(image_paths, model_name=clip_model)
    text_emb = embed_texts([prep.combined_text or "영상"], model_name=clip_model)[0]
    vs = score_video(uid, prep.keyframes, image_embs, text_emb)
    return vs


# ──────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
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


def load_done(path: Path) -> set:
    """이미 평가된 영상의 video_id 집합."""
    return {
        r.get("video_id")
        for r in load_jsonl(path)
        if r.get("video_id")
    }


# ──────────────────────────────────────────────
# 분석
# ──────────────────────────────────────────────

def analyze(out_path: Path) -> None:
    rows = load_jsonl(out_path)
    if not rows:
        print("\n결과 파일이 비어 있습니다.")
        return

    print(f"\n{'='*48}")
    print(f"=== Baseline 분석 ({len(rows)}개) ===")
    print(f"{'='*48}")

    normal_scores = [r["overall_score"] for r in rows if r["label_true"] == "normal"]
    fake_scores = [r["overall_score"] for r in rows if r["label_true"] == "fake"]

    def stats(name, arr):
        if not arr:
            print(f"  {name:<10}: (empty)")
            return
        a = np.array(arr)
        print(
            f"  {name:<10}: n={len(arr)}  mean={a.mean():.2f}  std={a.std():.2f}  "
            f"median={np.median(a):.2f}  min={a.min():.2f}  max={a.max():.2f}"
        )

    print("\n[점수 분포]")
    stats("normal", normal_scores)
    stats("fake", fake_scores)

    print("\n[임계값별 분류 성능]  (점수 >= thresh 면 fake 로 예측)")
    print(f"  {'thresh':>8}  {'prec':>8}  {'recall':>8}  {'f1':>8}  {'acc':>8}")

    best = None
    for thresh in [30, 35, 40, 45, 50, 55, 60, 65]:
        tp = sum(1 for s in fake_scores if s >= thresh)
        fn = len(fake_scores) - tp
        fp = sum(1 for s in normal_scores if s >= thresh)
        tn = len(normal_scores) - fp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
        print(f"  {thresh:>8}  {prec:>8.3f}  {rec:>8.3f}  {f1:>8.3f}  {acc:>8.3f}")
        if best is None or f1 > best[1]:
            best = (thresh, f1, prec, rec, acc)

    if best:
        print(
            f"\n[최적 F1 임계값] thresh={best[0]}  f1={best[1]:.3f}  "
            f"prec={best[2]:.3f}  rec={best[3]:.3f}  acc={best[4]:.3f}"
        )

    try:
        from sklearn.metrics import roc_auc_score
        y_true = [1] * len(fake_scores) + [0] * len(normal_scores)
        y_score = fake_scores + normal_scores
        auc = roc_auc_score(y_true, y_score)
        print(f"[ROC-AUC]      {auc:.3f}")
    except Exception as e:
        print(f"ROC-AUC 계산 실패: {e}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline zero-shot 평가")
    p.add_argument("--normal-manifest", type=Path, default=Path("data/manifest/normal.jsonl"))
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/eval/baseline_scores.jsonl"))
    p.add_argument("--errors", type=Path, default=Path("data/eval/errors.log"))
    p.add_argument("--whisper", default="tiny", choices=["tiny", "base", "small", "medium"])
    p.add_argument("--clip", default="ViT-B/32", choices=["ViT-B/32", "ViT-L/14"])
    p.add_argument("--scene-threshold", type=float, default=27.0)
    p.add_argument("--analyze-only", action="store_true",
                   help="평가는 안 하고 기존 결과 파일로 분석만 출력")
    return p.parse_args()


def main():
    args = parse_args()

    if args.analyze_only:
        analyze(args.out)
        return

    print(f"[eval] whisper      : {args.whisper}")
    print(f"[eval] clip         : {args.clip}")

    normal_items = load_jsonl(args.normal_manifest)
    fake_items = load_jsonl(args.fake_manifest)
    tasks = (
        [(it, "normal", it.get("video_id")) for it in normal_items]
        + [(it, "fake", it.get("fake_id")) for it in fake_items]
    )
    print(f"[eval] 정상         : {len(normal_items)}개")
    print(f"[eval] 위조         : {len(fake_items)}개")
    print(f"[eval] 합계         : {len(tasks)}개")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.errors.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(args.out)
    print(f"[eval] 이미 처리됨  : {len(done)}개 — skip")

    with args.out.open("a", encoding="utf-8") as out_f:
        for item, label, vid in tqdm(tasks, desc="evaluate", unit="vid"):
            if vid in done:
                continue
            file_path = item.get("file_path") or ""
            if not file_path or not Path(file_path).exists():
                continue

            try:
                vs = evaluate_video(
                    file_path=file_path,
                    whisper_model=args.whisper,
                    clip_model=args.clip,
                    scene_threshold=args.scene_threshold,
                )
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
                "whisper": args.whisper,
                "clip": args.clip,
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            done.add(vid)

    analyze(args.out)


if __name__ == "__main__":
    main()
