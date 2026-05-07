# Cheapfake 탐지 시스템

멀티모달 장면 이해 기반 시사 숏폼 영상의 맥락 불일치 탐지 시스템.  
CLIP + Whisper + OCR 파이프라인으로 영상 장면과 텍스트 설명 간 시공간적·의미적 불일치를 탐지한다.

---

## 설치

### 사전 요구사항

- Python 3.10+
- FFmpeg (`brew install ffmpeg` / `apt install ffmpeg`)
- (선택) CUDA 지원 GPU

### 환경 설정

```bash
# 1. 가상환경 생성
conda create -n cheapfake python=3.10 -y
conda activate cheapfake

# 2. 패키지 설치
pip install -r requirements.txt

# 3. CLIP 설치 (git 필요)
pip install git+https://github.com/openai/CLIP.git
```

---

## 실행

### 웹 UI (Gradio)

```bash
python app.py
```

공개 링크가 필요한 경우 (발표·데모용):
```bash
python app.py --share
```

### CLI (빠른 테스트)

```bash
# 로컬 파일 분석
python -m src.pipeline /path/to/video.mp4

# URL 분석
python -m src.pipeline https://www.youtube.com/watch?v=XXXXX
```

### 모듈 단위 테스트

```bash
# 전처리만 테스트
python src/preprocess.py /path/to/video.mp4

# CLIP 임베딩 테스트
python src/embed.py
```

---

## 프로젝트 구조

```
cheapfake-detector/
├── app.py              # Gradio 웹 UI 진입점
├── requirements.txt
├── src/
│   ├── ingest.py       # 입력 수신 (파일/URL)
│   ├── preprocess.py   # 키프레임 추출 + STT + OCR
│   ├── embed.py        # CLIP 임베딩
│   ├── score.py        # 불일치 점수 계산
│   ├── visualize.py    # 결과 시각화
│   └── pipeline.py     # 통합 파이프라인
├── data/
│   ├── raw/            # 다운로드된 원본 영상
│   └── processed/      # 키프레임 이미지 + JSON 캐시
└── outputs/            # 분석 결과 저장
```

---

## 파이프라인 개요

```
입력 (파일/URL)
    ↓  ingest.py
영상 저장 + 메타데이터 추출
    ↓  preprocess.py
키프레임 추출 (PySceneDetect)
    + STT (Whisper)
    + OCR (EasyOCR)
    ↓  embed.py
CLIP 이미지 임베딩 (키프레임)
CLIP 텍스트 임베딩 (STT+OCR 합성)
    ↓  score.py
프레임별 코사인 유사도 계산
시간적 이상치 탐지
종합 불일치 점수 (0~100)
    ↓  visualize.py
타임라인 히트맵 + 게이지
```

---

## 분석 결과 해석

| 점수 범위 | 판정 | 의미 |
|---|---|---|
| 0 ~ 40 | 정상 | 영상과 텍스트 맥락 일치 |
| 40 ~ 65 | 의심 | 일부 불일치 가능성 |
| 65 ~ 100 | 높은 위험 | 명확한 맥락 불일치 (cheapfake 의심) |

---

## GPU 없이 실행할 때 권장 설정

| 설정 | 권장값 | 이유 |
|---|---|---|
| Whisper 모델 | `base` | tiny 보다 정확, small 보다 빠름 |
| CLIP 모델 | `ViT-B/32` | ViT-L/14 대비 3배 빠름 |
| 장면 감지 임계값 | 27 (기본값) | 적당한 키프레임 수 |
