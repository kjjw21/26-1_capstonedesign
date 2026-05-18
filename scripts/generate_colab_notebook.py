"""
scripts/generate_colab_notebook.py
notebooks/lora_finetune_colab.ipynb 를 생성한다.
콘텐츠는 이 스크립트 안에 셀 단위로 정의되어 있다.
"""

import nbformat as nbf
from pathlib import Path

OUT = Path("notebooks/lora_finetune_colab.ipynb")


cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


# ──────────────────────────────────────────────
# 1. 제목 + 개요
# ──────────────────────────────────────────────

md(r"""# Phase B-2 — LoRA fine-tune of multilingual CLIP text encoder

**목표**: 다국어 CLIP의 text encoder를 LoRA로 부분 학습하여
한국어 cheapfake 탐지 task에 적응시키고, baseline AUC를 개선한다.

**Stage B-1 결과 (frozen + projection MLP, InfoNCE)**:
- L1 baseline 0.722 → 0.707 (Δ -0.015)
- L2 baseline 0.522 → 0.496 (Δ -0.026)
- L3 baseline 0.405 → 0.401 (Δ -0.004)

→ frozen embedding의 한계 도달. encoder 자체 학습이 필요.

**이 노트북에서 하는 일**:
1. Google Drive에서 `colab_package.zip` 로드
2. 다국어 CLIP 로드 + text encoder에 LoRA 어댑터 추가
3. Supervised contrastive loss로 학습
4. 5-fold CV로 L1/L2/L3별 AUC 측정
5. Baseline 대비 개선도 보고
""")


# ──────────────────────────────────────────────
# 2. 환경 셋업
# ──────────────────────────────────────────────

md("## 1. 환경 셋업 (패키지 설치)")

code(r"""!pip install -q open_clip_torch transformers peft scikit-learn""")


# ──────────────────────────────────────────────
# 3. 데이터 마운트
# ──────────────────────────────────────────────

md(r"""## 2. 데이터 마운트

**필요한 작업** (Colab 첫 실행 시):
1. 로컬에서 `python scripts/package_for_colab.py` 실행 → `data/colab_package.zip` 생성
2. Google Drive 에 업로드 (어디든)
3. 아래 셀에서 Drive 마운트 후 zip 경로 지정
""")

code(r"""from google.colab import drive
drive.mount('/content/drive')

import os
# ↓↓↓ 여기를 본인이 zip 을 업로드한 경로로 변경 ↓↓↓
ZIP_PATH = '/content/drive/MyDrive/colab_package.zip'

assert os.path.exists(ZIP_PATH), f'zip not found at {ZIP_PATH}'
print(f'OK: {os.path.getsize(ZIP_PATH) / 1024 / 1024:.1f} MB')""")

code(r"""!rm -rf /content/data_pkg
!mkdir -p /content/data_pkg
!unzip -q "$ZIP_PATH" -d /content/data_pkg
!ls /content/data_pkg""")


# ──────────────────────────────────────────────
# 4. 데이터 로드
# ──────────────────────────────────────────────

md(r"""## 3. 데이터 로드

- `embeddings/<video_id>.npz`: image_embs (N, 512) 와 baseline text_emb (512)
- `text/<video_id>.txt`: raw STT + OCR 합성 텍스트 (LoRA 학습 시 다시 인코딩)
- `manifest/normal.jsonl`, `manifest/fake.jsonl`: 라벨과 kind
""")

code(r"""import json, numpy as np, os
from pathlib import Path

ROOT = Path('/content/data_pkg')
EMB_DIR = ROOT / 'embeddings'
TXT_DIR = ROOT / 'text'

def load_jsonl(p):
    items = []
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

# fake_id -> kind
kind_map = {}
for it in load_jsonl(ROOT / 'manifest' / 'fake.jsonl'):
    fid = it.get('fake_id')
    kind = it.get('kind') or 'audio_swap'
    if kind == 'audio_swap':
        kind = 'audio_swap_random'
    if fid:
        kind_map[fid] = kind

# 모든 npz 로드
img_means = []   # (N, 512)  — frozen
raw_texts = []   # str       — LoRA 학습 시 토크나이즈
labels = []      # 1=normal, 0=fake
kinds = []
ids = []
for npz_path in sorted(EMB_DIR.glob('*.npz')):
    vid = npz_path.stem
    d = np.load(npz_path, allow_pickle=True)
    img_mean = d['image_embs'].mean(axis=0).astype(np.float32)
    label = str(d['label'])
    txt_file = TXT_DIR / f'{vid}.txt'
    raw_text = txt_file.read_text(encoding='utf-8') if txt_file.exists() else '영상'
    img_means.append(img_mean)
    raw_texts.append(raw_text)
    labels.append(1 if label == 'normal' else 0)
    kinds.append('normal' if label == 'normal' else kind_map.get(vid, 'audio_swap_random'))
    ids.append(vid)

img_means = np.stack(img_means)
labels = np.array(labels, dtype=np.int64)
from collections import Counter
print(f'total: {len(labels)}, normal: {(labels==1).sum()}, fake: {(labels==0).sum()}')
print(f'kinds: {dict(Counter(kinds))}')""")


# ──────────────────────────────────────────────
# 5. 모델 + LoRA
# ──────────────────────────────────────────────

md(r"""## 4. 다국어 CLIP + LoRA 어댑터

- **Vision encoder**: frozen (이미 추출된 `img_means` 사용)
- **Text encoder (xlm-roberta-base)**: LoRA로 부분 학습
  - target: `query`, `value` projection in self-attention
  - rank 8, alpha 16, dropout 0.1
""")

code(r"""import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

model, _, _ = open_clip.create_model_and_transforms(
    'xlm-roberta-base-ViT-B-32',
    pretrained='laion5b_s13b_b90k',
)
tokenizer = open_clip.get_tokenizer('xlm-roberta-base-ViT-B-32')
model = model.to(device)

# vision encoder 는 frozen (사용 안 함, image_embs 직접 입력)
# text encoder 의 transformer (xlm-roberta) 에 LoRA
print(type(model.text))
print(type(model.text.transformer))""")

code(r"""# XLM-RoBERTa 의 attention linear 이름 확인
for name, module in model.text.transformer.named_modules():
    if isinstance(module, nn.Linear) and ('attention' in name or 'attn' in name):
        print(name, '->', module)
        # 처음 4개만 보기
        break

# LoRA target — XLM-RoBERTa 는 query / value 키 사용
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['query', 'value'],
    lora_dropout=0.1,
    bias='none',
    task_type='FEATURE_EXTRACTION',
)
model.text.transformer = get_peft_model(model.text.transformer, lora_config)
model.text.transformer.print_trainable_parameters()""")


# ──────────────────────────────────────────────
# 6. 텍스트 → 임베딩 함수
# ──────────────────────────────────────────────

md(r"""## 5. 텍스트 인코딩 헬퍼

`encode_text(list_of_strings)` → (B, 512) text embedding""")

code(r"""def encode_text(texts, batch_size=16, train=True):
    embs = []
    if train:
        model.train()
    else:
        model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = tokenizer(batch).to(device)
        out = model.encode_text(tokens)
        embs.append(out)
    return torch.cat(embs, dim=0)""")


# ──────────────────────────────────────────────
# 7. 5-fold CV 학습 + 평가
# ──────────────────────────────────────────────

md(r"""## 6. 5-fold CV 학습 + 평가

- 정상만 5등분, 4 fold 학습 + 1 fold 평가
- 학습 데이터: train fold 정상 + 모든 위조 (supervised, label 활용)
- 평가: test fold 정상 + 모든 위조 zero-shot
- Loss: **Supervised contrastive** — 정상 (img,txt) sim → 1, 위조 (img,txt) sim → 0 (BCE on sim)
- Metric: AUC (전체, L1, L2, L3 분리)
""")

code(r"""from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

EPOCHS = 30
LR = 5e-5
BATCH = 16
TEMP = 5.0

def reset_lora():
    \"\"\"매 fold 마다 LoRA 가중치 재초기화\"\"\"
    global model
    for name, p in model.text.transformer.named_parameters():
        if 'lora' in name.lower():
            if 'A' in name or 'a.weight' in name.lower():
                nn.init.kaiming_uniform_(p, a=5**0.5)
            else:
                nn.init.zeros_(p)

def train_one_fold(tr_idx_norm, te_idx_norm, fake_idx, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    reset_lora()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=1e-2,
    )

    train_idx = np.concatenate([tr_idx_norm, fake_idx])
    train_labels = labels[train_idx].astype(np.float32)
    train_imgs = torch.from_numpy(img_means[train_idx]).to(device)
    train_texts = [raw_texts[i] for i in train_idx]
    train_y = torch.from_numpy(train_labels).to(device)

    n = len(train_idx)
    epoch_losses = []
    for ep in range(EPOCHS):
        perm = np.random.permutation(n)
        ep_loss_sum = 0.0
        ep_batches = 0
        for i in range(0, n, BATCH):
            sel = perm[i:i+BATCH]
            opt.zero_grad()
            txt_emb = encode_text([train_texts[j] for j in sel], train=True)
            txt_emb = F.normalize(txt_emb, dim=-1)
            img_emb = train_imgs[torch.from_numpy(sel).long().to(device)]
            sim = (img_emb * txt_emb).sum(dim=-1)
            logits = sim * TEMP
            loss = F.binary_cross_entropy_with_logits(logits, train_y[torch.from_numpy(sel).long().to(device)])
            loss.backward()
            opt.step()
            ep_loss_sum += loss.item()
            ep_batches += 1
        ep_loss = ep_loss_sum / max(ep_batches, 1)
        epoch_losses.append(ep_loss)
        # 학습이 잘 가고 있는지 5 epoch 마다 출력
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'  ep {ep+1:>3}/{EPOCHS}  loss={ep_loss:.4f}')

    # 평가
    model.eval()
    eval_idx = np.concatenate([te_idx_norm, fake_idx])
    with torch.no_grad():
        eval_imgs = torch.from_numpy(img_means[eval_idx]).to(device)
        eval_texts = [raw_texts[i] for i in eval_idx]
        eval_txt_emb = encode_text(eval_texts, train=False)
        eval_txt_emb = F.normalize(eval_txt_emb, dim=-1)
        sim = (eval_imgs * eval_txt_emb).sum(dim=-1).cpu().numpy()
    fake_score = -sim
    y_te = labels[eval_idx]
    fake_target = 1 - y_te
    auc_all = roc_auc_score(fake_target, fake_score)
    kinds_te = [kinds[i] for i in eval_idx]
    kinds_arr = np.array(kinds_te)
    normal_mask = kinds_arr == 'normal'
    out = {'all': auc_all}
    for kname, key in [('L1','audio_swap_random'),('L2','audio_swap_topic'),('L3','audio_swap_topic_strong')]:
        mask = kinds_arr == key
        if mask.sum() < 2 or normal_mask.sum() < 2:
            out[kname] = float('nan'); continue
        sub_y = np.concatenate([np.ones(mask.sum()), np.zeros(normal_mask.sum())])
        sub_s = np.concatenate([fake_score[mask], fake_score[normal_mask]])
        out[kname] = roc_auc_score(sub_y, sub_s)
    out['epoch_losses'] = epoch_losses
    return out""")

code(r"""normal_idx_all = np.where(labels == 1)[0]
fake_idx_all = np.where(labels == 0)[0]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
for fold, (tr_n, te_n) in enumerate(kf.split(normal_idx_all), 1):
    tr = normal_idx_all[tr_n]
    te = normal_idx_all[te_n]
    print(f'\\n=== Fold {fold} ===  train normal={len(tr)}, test normal={len(te)}, fake={len(fake_idx_all)}')
    r = train_one_fold(tr, te, fake_idx_all, seed=42+fold)
    r['fold'] = fold
    fold_results.append(r)
    print(f'  AUC_all={r["all"]:.3f}  L1={r["L1"]:.3f}  L2={r["L2"]:.3f}  L3={r["L3"]:.3f}')""")


# ──────────────────────────────────────────────
# 8. 결과 정리
# ──────────────────────────────────────────────

md(r"""## 7. 결과 정리""")

code(r"""print('=== 5-fold CV 평균 ===')
for k in ['all','L1','L2','L3']:
    vals = np.array([f[k] for f in fold_results])
    print(f'  AUC_{k:<3}: {np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}')

print('\\n=== Baseline (zero-shot multilingual CLIP) ===')
print('  L1: 0.722   L2: 0.522   L3: 0.405')
print('=== Stage B-1 (InfoNCE projection) ===')
print('  L1: 0.707   L2: 0.496   L3: 0.401')
print('=== Stage B-2 (LoRA text encoder) ===')
L1 = float(np.nanmean([f['L1'] for f in fold_results]))
L2 = float(np.nanmean([f['L2'] for f in fold_results]))
L3 = float(np.nanmean([f['L3'] for f in fold_results]))
print(f'  L1: {L1:.3f}   L2: {L2:.3f}   L3: {L3:.3f}')
print(f'\\nΔ vs baseline: {L1-0.722:+.3f} / {L2-0.522:+.3f} / {L3-0.405:+.3f}')""")


# ──────────────────────────────────────────────
# 9. 결과 다운로드
# ──────────────────────────────────────────────

md(r"""## 8. 결과 저장 + 다운로드""")

code(r"""import json
result = {
    'fold_results': [{k: float(v) if isinstance(v, (int,float,np.floating)) else v for k,v in r.items()} for r in fold_results],
    'config': {'epochs': EPOCHS, 'lr': LR, 'batch': BATCH, 'temp': TEMP},
    'baseline': {'L1': 0.722, 'L2': 0.522, 'L3': 0.405},
}
with open('/content/lora_results.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

from google.colab import files
files.download('/content/lora_results.json')""")


# ──────────────────────────────────────────────
# 노트북 저장
# ──────────────────────────────────────────────

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.10",
    },
    "colab": {
        "provenance": [],
        "gpuType": "T4",
    },
    "accelerator": "GPU",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"노트북 생성됨: {OUT}")
print(f"셀 수: {len(cells)}")
