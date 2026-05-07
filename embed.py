"""
embed.py
CLIP ViT-L/14 기반 멀티모달 임베딩 모듈.

- 키프레임 이미지 배치 임베딩
- 텍스트 임베딩
- 코사인 유사도 계산
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List

import clip
from PIL import Image


# ──────────────────────────────────────────────
# 모델 로드 (싱글턴)
# ──────────────────────────────────────────────

_model = None
_preprocess_fn = None
_device = None


def load_model(model_name: str = "ViT-L/14") -> tuple:
    """
    CLIP 모델을 로드한다. 최초 1회만 로드하고 이후는 캐시된 인스턴스 반환.
    GPU 가 있으면 CUDA, 없으면 CPU 사용.

    Returns
    -------
    (model, preprocess_fn, device)
    """
    global _model, _preprocess_fn, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model, _preprocess_fn = clip.load(model_name, device=_device)
        _model.eval()
        print(f"[embed] CLIP {model_name} 로드 완료 — device: {_device}")
    return _model, _preprocess_fn, _device


# ──────────────────────────────────────────────
# 이미지 임베딩
# ──────────────────────────────────────────────

def embed_images(
    image_paths: List[str],
    batch_size: int = 8,
    model_name: str = "ViT-L/14",
) -> np.ndarray:
    """
    키프레임 이미지 리스트를 CLIP 이미지 임베딩으로 변환한다.

    Parameters
    ----------
    image_paths : 이미지 파일 경로 리스트
    batch_size  : GPU 메모리에 맞게 조정 (4GB VRAM → 8, CPU → 4)

    Returns
    -------
    np.ndarray, shape (N, D) — L2 정규화된 임베딩
    """
    model, preprocess_fn, device = load_model(model_name)

    all_embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(preprocess_fn(img))
            except Exception as e:
                # 손상된 이미지는 검은색 이미지로 대체
                print(f"[embed] 이미지 로드 실패 {p}: {e}")
                dummy = Image.new("RGB", (224, 224), color=(0, 0, 0))
                images.append(preprocess_fn(dummy))

        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 정규화
        all_embeddings.append(feats.cpu().numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ──────────────────────────────────────────────
# 텍스트 임베딩
# ──────────────────────────────────────────────

def embed_texts(
    texts: List[str],
    model_name: str = "ViT-L/14",
) -> np.ndarray:
    """
    텍스트 리스트를 CLIP 텍스트 임베딩으로 변환한다.
    각 텍스트는 CLIP 토크나이저 기준 77 토큰으로 자동 절단된다.

    Returns
    -------
    np.ndarray, shape (N, D) — L2 정규화된 임베딩
    """
    model, _, device = load_model(model_name)

    # CLIP 토크나이저는 77 토큰 초과 시 자동 truncate
    tokens = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    return feats.cpu().numpy().astype(np.float32)


# ──────────────────────────────────────────────
# 유사도 계산
# ──────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 벡터 또는 배열 간 코사인 유사도 계산.
    입력이 이미 L2 정규화된 경우 내적(dot product)과 동일.

    Parameters
    ----------
    a : (D,) 또는 (N, D)
    b : (D,) 또는 (N, D)

    Returns
    -------
    스칼라 또는 (N,) 배열
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.ndim == 1 and b.ndim == 1:
        return float(np.dot(a, b))
    # 배치 연산: 각 쌍의 내적
    return np.einsum("id,id->i", a, b)


def frame_text_similarities(
    image_embeddings: np.ndarray,
    text_embedding: np.ndarray,
) -> np.ndarray:
    """
    키프레임별로 텍스트 임베딩과의 코사인 유사도를 계산한다.

    Parameters
    ----------
    image_embeddings : (N, D)
    text_embedding   : (D,) — 단일 텍스트 벡터

    Returns
    -------
    np.ndarray shape (N,) — 각 프레임의 유사도 [-1, 1]
    """
    text_emb = text_embedding.flatten()
    sims = []
    for img_emb in image_embeddings:
        sims.append(cosine_similarity(img_emb, text_emb))
    return np.array(sims, dtype=np.float32)


if __name__ == "__main__":
    # 간단한 동작 테스트
    print("CLIP 모델 로드 테스트...")
    model, prep, device = load_model("ViT-B/32")  # 테스트엔 작은 모델
    print(f"device: {device}")

    dummy_texts = ["뉴스 영상", "스포츠 경기 장면", "전쟁 현장"]
    text_embs = embed_texts(dummy_texts, model_name="ViT-B/32")
    print(f"텍스트 임베딩 shape: {text_embs.shape}")

    sim = cosine_similarity(text_embs[0], text_embs[1])
    print(f"'뉴스 영상' vs '스포츠 경기 장면' 유사도: {sim:.4f}")
