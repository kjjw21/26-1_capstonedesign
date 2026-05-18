"""
src/retrieval.py
정상 corpus 검색 모듈. 새 영상의 image_mean 또는 text_emb 을 받아
가장 유사한 정상 영상 top-K 를 반환한다.

자산 파일:
    data/app/retrieval_index.npz   (keys, img_mat, txt_mat — 모두 L2 정규화)
    data/app/retrieval_meta.jsonl  (각 video_id 의 title/url/channel 등)
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np


_INDEX_PATH = Path("data/app/retrieval_index.npz")
_META_PATH = Path("data/app/retrieval_meta.jsonl")

_keys = None
_img_mat = None
_txt_mat = None
_meta_by_id = None


def _load():
    global _keys, _img_mat, _txt_mat, _meta_by_id
    if _keys is None:
        if not _INDEX_PATH.exists():
            raise FileNotFoundError(
                f"검색 인덱스가 없습니다: {_INDEX_PATH}\n"
                "먼저 `python scripts/export_app_assets.py` 를 실행하세요."
            )
        d = np.load(_INDEX_PATH, allow_pickle=True)
        _keys = list(d["keys"])
        _img_mat = d["img_mat"]
        _txt_mat = d["txt_mat"]

        _meta_by_id = {}
        if _META_PATH.exists():
            with _META_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        m = json.loads(line)
                        _meta_by_id[m["video_id"]] = m
                    except Exception:
                        continue


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _topk(query: np.ndarray, mat: np.ndarray, k: int):
    q = _normalize(query.astype(np.float32))
    sims = mat @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def search_by_text(text_emb: np.ndarray, k: int = 5) -> List[dict]:
    """위조 영상의 STT+OCR 텍스트 임베딩으로 audio_source 후보 검색."""
    _load()
    hits = _topk(text_emb, _txt_mat, k)
    return [_format_hit(i, sim, mode="text") for i, sim in hits]


def search_by_image(image_emb_mean: np.ndarray, k: int = 5) -> List[dict]:
    """위조 영상의 이미지 평균 임베딩으로 video_source 후보 검색."""
    _load()
    hits = _topk(image_emb_mean, _img_mat, k)
    return [_format_hit(i, sim, mode="image") for i, sim in hits]


def _format_hit(idx: int, sim: float, mode: str) -> dict:
    vid = _keys[idx]
    meta = _meta_by_id.get(vid, {}) if _meta_by_id else {}
    return {
        "video_id": vid,
        "similarity": sim,
        "title": meta.get("title", ""),
        "url": meta.get("url", ""),
        "channel": meta.get("channel"),
        "upload_date": meta.get("upload_date"),
        "mode": mode,
    }


def info() -> dict:
    _load()
    return {
        "corpus_size": len(_keys),
        "embedding_dim": _img_mat.shape[1] if _img_mat is not None else None,
    }
