"""
src/mclip.py
다국어 CLIP (xlm-roberta-base-ViT-B-32, laion5b_s13b_b90k 가중치) 임베딩.

scripts/run_baseline_multilingual.py 와 동일한 인코더를 사용한다.
앱 런타임에서는 이 모듈을 import 해서 영상 임베딩을 추출한다.
"""

from typing import List

import numpy as np
import torch
from PIL import Image


MODEL_NAME = "xlm-roberta-base-ViT-B-32"
PRETRAINED = "laion5b_s13b_b90k"

_state = None  # (model, preprocess_fn, tokenizer, device)


def _get():
    global _state
    if _state is None:
        import open_clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess_fn = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        model.eval().to(device)
        _state = (model, preprocess_fn, tokenizer, device)
    return _state


def embed_images(image_paths: List[str], batch_size: int = 8) -> np.ndarray:
    model, preprocess_fn, _, device = _get()
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        imgs = []
        for p in batch_paths:
            try:
                imgs.append(preprocess_fn(Image.open(p).convert("RGB")))
            except Exception:
                imgs.append(preprocess_fn(Image.new("RGB", (224, 224))))
        batch = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_embs.append(feats.cpu().numpy())
    return np.vstack(all_embs).astype(np.float32)


def embed_text(text: str) -> np.ndarray:
    model, _, tokenizer, device = _get()
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)[0]
