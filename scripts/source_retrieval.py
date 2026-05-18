"""
scripts/source_retrieval.py
Phase C — 원본 검색 시스템 (Source Retrieval).

각 위조 영상에 대해 정상 풀에서 (a) 오디오 출처 (b) 영상 출처를 찾는다.
ground truth 는 합성 시 manifest 에 기록된 audio_source / video_source.

평가 메트릭:
    Recall@1, Recall@5, Recall@10, MRR (Mean Reciprocal Rank)

분리 분석:
    L1 (random) / L2 (topic) / L3 (topic strong) 별 결과
    검색 모드별: text 기반 audio source, image 기반 video source, cross-modal

실행:
    python scripts/source_retrieval.py
    python scripts/source_retrieval.py --top-k 1 5 10 20
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from run_baseline_eval import load_jsonl


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_embeddings(embeddings_dir: Path):
    """모든 npz 를 로드하여 (vid -> {img, txt, label}) 딕셔너리 반환."""
    data = {}
    for npz in sorted(embeddings_dir.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        data[npz.stem] = {
            "img": d["image_embs"].mean(axis=0).astype(np.float32),
            "txt": d["text_emb"].astype(np.float32),
            "label": str(d["label"]),
        }
    return data


def load_fake_manifest_map(fake_manifest: Path):
    """fake_id -> {audio_source, video_source, kind}."""
    out = {}
    for it in load_jsonl(fake_manifest):
        fid = it.get("fake_id")
        if not fid:
            continue
        kind = it.get("kind") or "audio_swap"
        if kind == "audio_swap":
            kind = "audio_swap_random"
        out[fid] = {
            "audio_source": it.get("audio_source"),
            "video_source": it.get("video_source"),
            "kind": kind,
        }
    return out


# ──────────────────────────────────────────────
# 검색
# ──────────────────────────────────────────────

def cosine_topk(query: np.ndarray, corpus_keys: list, corpus_mat: np.ndarray, k: int):
    """query (D,) — corpus_mat (N, D) 이미 L2 정규화 가정.
    반환: 유사도 내림차순 정렬된 (key, score) 리스트 길이 k."""
    q = query / (np.linalg.norm(query) + 1e-12)
    sims = corpus_mat @ q                # (N,)
    order = np.argsort(-sims)[:k]
    return [(corpus_keys[i], float(sims[i])) for i in order]


def rank_of(target: str, ranked: list) -> int:
    """target 이 ranked 의 몇 번째에 있나 (1-based). 없으면 None."""
    for i, (k, _) in enumerate(ranked, start=1):
        if k == target:
            return i
    return None


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def evaluate_mode(
    fake_items: list,           # [{vid, kind, gt, query_mode}]
    corpus_keys: list,
    corpus_mat: np.ndarray,
    top_ks: list,
):
    """평가 결과 dict 반환: 전체 + kind 별 Recall@K, MRR."""
    by_kind = defaultdict(list)
    all_ranks = []

    for fi in fake_items:
        gt = fi["gt"]
        if not gt or gt not in corpus_keys:
            continue
        ranked = cosine_topk(fi["query"], corpus_keys, corpus_mat, k=len(corpus_keys))
        r = rank_of(gt, ranked)
        all_ranks.append((fi["kind"], r))
        by_kind[fi["kind"]].append(r)

    def summarize(ranks):
        n = len(ranks)
        if n == 0:
            return {f"R@{k}": float("nan") for k in top_ks} | {"MRR": float("nan"), "n": 0}
        out = {}
        for k in top_ks:
            out[f"R@{k}"] = sum(1 for r in ranks if r is not None and r <= k) / n
        rr = [1.0 / r for r in ranks if r is not None]
        out["MRR"] = sum(rr) / n
        out["n"] = n
        return out

    result = {
        "all": summarize([r for _, r in all_ranks]),
        "by_kind": {k: summarize(rs) for k, rs in by_kind.items()},
    }
    return result


def stack_corpus(data: dict, modality: str):
    """data 에서 label=='normal' 인 (key, normalized vec) 만 추려서 (keys, mat) 반환.
    modality: 'img' or 'txt'"""
    keys, vecs = [], []
    for k, v in data.items():
        if v["label"] != "normal":
            continue
        keys.append(k)
        vecs.append(v[modality])
    mat = np.stack(vecs).astype(np.float32)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return keys, mat


def build_fake_queries(data: dict, fake_map: dict, query_modality: str, target_field: str):
    """위조 영상마다 query/gt 페어 생성.
    query_modality: 'img' or 'txt'
    target_field: 'audio_source' or 'video_source'"""
    out = []
    for vid, v in data.items():
        if v["label"] != "fake":
            continue
        meta = fake_map.get(vid)
        if not meta:
            continue
        out.append({
            "vid": vid,
            "kind": meta["kind"],
            "query": v[query_modality],
            "gt": meta.get(target_field),
        })
    return out


# ──────────────────────────────────────────────
# 출력
# ──────────────────────────────────────────────

def print_result(label: str, result: dict, top_ks: list):
    print(f"\n[{label}]")
    a = result["all"]
    rs = "  ".join([f"R@{k}={a[f'R@{k}']:.3f}" for k in top_ks])
    print(f"  ALL (n={a['n']:>3}): {rs}  MRR={a['MRR']:.3f}")
    print(f"  --- by kind ---")
    label_map = {
        "audio_swap_random": "L1 random",
        "audio_swap_topic": "L2 topic",
        "audio_swap_topic_strong": "L3 strong",
    }
    for kind_key in ["audio_swap_random", "audio_swap_topic", "audio_swap_topic_strong"]:
        m = result["by_kind"].get(kind_key)
        if not m or m["n"] == 0:
            continue
        rs = "  ".join([f"R@{k}={m[f'R@{k}']:.3f}" for k in top_ks])
        print(f"  {label_map[kind_key]:<10} (n={m['n']:>2}): {rs}  MRR={m['MRR']:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--fake-manifest", type=Path, default=Path("data/manifest/fake.jsonl"))
    p.add_argument("--top-k", type=int, nargs="+", default=[1, 5, 10])
    return p.parse_args()


def main():
    args = parse_args()
    data = load_embeddings(args.embeddings_dir)
    fake_map = load_fake_manifest_map(args.fake_manifest)
    n_normal = sum(1 for v in data.values() if v["label"] == "normal")
    n_fake = sum(1 for v in data.values() if v["label"] == "fake")
    print(f"[retr] normal corpus: {n_normal}")
    print(f"[retr] fake queries : {n_fake}")
    print(f"[retr] kinds        : {{ {', '.join(sorted({m['kind'] for m in fake_map.values()}))} }}")

    # 정상 풀 임베딩 행렬
    norm_keys_txt, norm_mat_txt = stack_corpus(data, "txt")
    norm_keys_img, norm_mat_img = stack_corpus(data, "img")

    # 모드 1: text → text, gt = audio_source
    q1 = build_fake_queries(data, fake_map, "txt", "audio_source")
    r1 = evaluate_mode(q1, norm_keys_txt, norm_mat_txt, args.top_k)
    print_result("(a) audio source retrieval — fake.txt → normal.txt", r1, args.top_k)

    # 모드 2: image → image, gt = video_source
    q2 = build_fake_queries(data, fake_map, "img", "video_source")
    r2 = evaluate_mode(q2, norm_keys_img, norm_mat_img, args.top_k)
    print_result("(b) video source retrieval — fake.img → normal.img", r2, args.top_k)

    # 모드 3a: cross-modal — fake.txt → normal.img (다른 모달리티 매칭)
    q3a = build_fake_queries(data, fake_map, "txt", "audio_source")
    r3a = evaluate_mode(q3a, norm_keys_img, norm_mat_img, args.top_k)
    print_result("(c1) cross-modal — fake.txt → normal.img (target audio_source)", r3a, args.top_k)

    # 모드 3b: cross-modal — fake.img → normal.txt
    q3b = build_fake_queries(data, fake_map, "img", "video_source")
    r3b = evaluate_mode(q3b, norm_keys_txt, norm_mat_txt, args.top_k)
    print_result("(c2) cross-modal — fake.img → normal.txt (target video_source)", r3b, args.top_k)


if __name__ == "__main__":
    main()
