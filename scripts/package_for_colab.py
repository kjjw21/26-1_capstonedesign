"""
scripts/package_for_colab.py
Phase B-2 (LoRA fine-tune) 를 Colab 에서 돌리기 위해 필요한 최소 자료를
하나의 zip 으로 묶는다.

포함되는 것:
    embeddings/<video_id>.npz   — 다국어 CLIP image_embs 와 text_emb (frozen)
    manifest/normal.jsonl       — 정상 영상 라벨
    manifest/fake.jsonl         — 위조 영상 라벨 (L1/L2/L3 구분 kind 포함)
    text/<video_id>.txt         — raw STT+OCR 합성 텍스트 (LoRA 학습 시 토크나이즈할 원본)

영상 raw 파일은 포함 안 됨 (이미 image embedding 이 npz 안에 있고
vision encoder 는 frozen 으로 갈 예정이라 불필요).
"""

import json
import shutil
import sys
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


PROCESSED_DIR = Path("C:/cheapfake_data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")
MANIFEST_DIR = Path("data/manifest")
OUT_ZIP = Path("data/colab_package.zip")
WORK_DIR = Path("data/_colab_pkg_tmp")


def extract_texts():
    """preprocess_result.json 에서 combined_text 를 꺼내 video_id 별 .txt 로 저장."""
    text_dir = WORK_DIR / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    by_filepath_to_uid = {}
    # processed/<uid>/preprocess_result.json 에 video_path 가 들어있음
    # video_id <-> uid 매핑이 필요한데, evaluate 시에는 file_path 기반 uid 라
    # manifest 의 file_path 와 일치한다. 그래서 manifest 의 fake_id 또는 video_id 를
    # 키로 삼고, 그에 대응하는 file_path → uid 계산 후 cache 에서 읽는다.

    import hashlib

    def file_path_to_uid(fp: str) -> str:
        return hashlib.md5(fp.encode()).hexdigest()[:12]

    saved = 0
    skipped = 0

    # normal
    for it in _load_jsonl(MANIFEST_DIR / "normal.jsonl"):
        vid = it.get("video_id")
        fp = it.get("file_path") or ""
        if not vid or not fp:
            continue
        uid = file_path_to_uid(fp)
        cache = PROCESSED_DIR / uid / "preprocess_result.json"
        if not cache.exists():
            skipped += 1
            continue
        with cache.open("r", encoding="utf-8") as f:
            d = json.load(f)
        text = d.get("combined_text", "") or "영상"
        (text_dir / f"{vid}.txt").write_text(text, encoding="utf-8")
        saved += 1

    # fake
    for it in _load_jsonl(MANIFEST_DIR / "fake.jsonl"):
        vid = it.get("fake_id")
        fp = it.get("file_path") or ""
        if not vid or not fp:
            continue
        uid = file_path_to_uid(fp)
        cache = PROCESSED_DIR / uid / "preprocess_result.json"
        if not cache.exists():
            skipped += 1
            continue
        with cache.open("r", encoding="utf-8") as f:
            d = json.load(f)
        text = d.get("combined_text", "") or "영상"
        (text_dir / f"{vid}.txt").write_text(text, encoding="utf-8")
        saved += 1

    return saved, skipped


def _load_jsonl(p: Path):
    items = []
    if not p.exists():
        return items
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def main():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    # embeddings 복사
    emb_dst = WORK_DIR / "embeddings"
    emb_dst.mkdir(parents=True, exist_ok=True)
    n_emb = 0
    for npz in EMBEDDINGS_DIR.glob("*.npz"):
        shutil.copy2(npz, emb_dst / npz.name)
        n_emb += 1
    print(f"[pkg] embeddings: {n_emb}개 복사")

    # manifest 복사
    man_dst = WORK_DIR / "manifest"
    man_dst.mkdir(parents=True, exist_ok=True)
    for name in ["normal.jsonl", "fake.jsonl"]:
        src = MANIFEST_DIR / name
        if src.exists():
            shutil.copy2(src, man_dst / name)

    # 텍스트 추출
    print("[pkg] STT/OCR combined_text 추출 중...")
    saved, skipped = extract_texts()
    print(f"[pkg] texts: saved={saved}  skipped={skipped} (캐시 없음)")

    # zip 생성
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in WORK_DIR.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(WORK_DIR))

    size_mb = OUT_ZIP.stat().st_size / 1024 / 1024
    print(f"\n[pkg] zip 생성 완료")
    print(f"  path : {OUT_ZIP}")
    print(f"  size : {size_mb:.1f} MB")
    print(f"\n다음 단계: Google Drive 에 이 zip 을 업로드한 뒤 notebooks/lora_finetune_colab.ipynb 를 Colab 에서 여세요.")


if __name__ == "__main__":
    main()
