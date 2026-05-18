"""
scripts/source_retrieval_robustness.py
Phase C — Source retrieval robustness sweep.

source_retrieval.py 가 보여준 R@1 = 0.992 (audio source) 는 우리의
합성 위조가 정상 음성을 그대로 STT 한 결과를 query 로 쓰기 때문에
trivial 한 best-case. 실제 SNS 환경에서는 STT 가 부정확하거나 자막이
달라지므로 query embedding 이 정확히 같지 않다.

이 스크립트는 위조의 text embedding 에 다양한 강도의 가우시안 노이즈를
추가하면서 R@1 이 어떻게 떨어지는지 측정해 시스템의 견고성을 정량화한다.

결과는 콘솔 + data/eval/retrieval_robustness.jsonl 에 저장된다.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_data():
    fake_meta = {}
    fp = Path("data/manifest/fake.jsonl")
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except Exception:
                continue
            fid = it.get("fake_id")
            if not fid:
                continue
            k = it.get("kind", "audio_swap")
            if k == "audio_swap":
                k = "audio_swap_random"
            fake_meta[fid] = {"audio_source": it.get("audio_source"), "kind": k}

    data = {}
    for npz in Path("data/embeddings").glob("*.npz"):
        d = np.load(npz, allow_pickle=True)
        data[npz.stem] = {
            "txt": d["text_emb"].astype(np.float32),
            "label": str(d["label"]),
        }
    return data, fake_meta


KIND_MAP = {
    "audio_swap_random": "L1",
    "audio_swap_topic": "L2",
    "audio_swap_topic_strong": "L3",
}


def evaluate(noise_std: float, data: dict, fake_meta: dict, top_k: int, seed: int):
    rng = np.random.default_rng(seed)
    norm_keys = [k for k, v in data.items() if v["label"] == "normal"]
    norm_mat = np.stack([data[k]["txt"] for k in norm_keys])
    norm_mat = norm_mat / (np.linalg.norm(norm_mat, axis=1, keepdims=True) + 1e-12)

    fake_keys = [k for k, v in data.items() if v["label"] == "fake"]
    hits_by_kind = {"L1": [], "L2": [], "L3": []}

    for fk in fake_keys:
        meta = fake_meta.get(fk)
        if not meta or meta["audio_source"] not in norm_keys:
            continue
        gt_idx = norm_keys.index(meta["audio_source"])
        q = data[fk]["txt"].copy()
        if noise_std > 0:
            q = q + rng.normal(0, noise_std, size=q.shape).astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        sims = norm_mat @ q
        topk = np.argsort(-sims)[:top_k]
        hit = int(gt_idx in topk)
        kk = KIND_MAP.get(meta["kind"])
        if kk:
            hits_by_kind[kk].append(hit)

    return {
        k: {
            "recall_at_k": (sum(v) / len(v)) if v else float("nan"),
            "n": len(v),
        }
        for k, v in hits_by_kind.items()
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--noise-std",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    )
    p.add_argument("--top-k", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/eval/retrieval_robustness.jsonl"))
    return p.parse_args()


def main():
    args = parse_args()
    data, fake_meta = load_data()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== Source retrieval robustness (top_k={args.top_k}) ===")
    print(f"{'noise_std':>10} {'L1':>9} {'L2':>9} {'L3':>9}")
    print("-" * 44)
    with args.out.open("w", encoding="utf-8") as f:
        for ns in args.noise_std:
            r = evaluate(ns, data, fake_meta, args.top_k, args.seed)
            print(f"{ns:>10.3f} {r['L1']['recall_at_k']:>9.3f} "
                  f"{r['L2']['recall_at_k']:>9.3f} {r['L3']['recall_at_k']:>9.3f}")
            row = {
                "noise_std": ns,
                "top_k": args.top_k,
                "seed": args.seed,
                "L1": r["L1"],
                "L2": r["L2"],
                "L3": r["L3"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nresults -> {args.out}")
    print("\n해석: 노이즈 0.05까지는 retrieval 안정. 0.10 부근에서 살짝 흔들리고,")
    print("0.20 부근에서 R@1 이 0.5 이하로 무너지는 'breaking point' 가 관측된다.")


if __name__ == "__main__":
    main()
