"""
app.py
Gradio 기반 웹 데모 UI.

실행:
    python app.py                   # 로컬 실행
    python app.py --share           # 공개 링크 생성
"""

import sys
import os
import argparse
from pathlib import Path

import gradio as gr

# src 패키지 경로 설정
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline
from src.visualize import build_timeline_chart, build_gauge_chart, build_frame_comparison_html


# ──────────────────────────────────────────────
# 분석 실행 함수
# ──────────────────────────────────────────────

def analyze(
    video_file,
    url_input: str,
    whisper_model: str,
    clip_model: str,
    scene_threshold: float,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Gradio 에서 호출되는 분석 함수.
    파일 또는 URL 중 하나를 입력으로 받는다.
    """
    # 입력 검증
    source = None
    if url_input and url_input.strip():
        source = url_input.strip()
    elif video_file is not None:
        source = video_file
    else:
        gr.Warning("파일을 업로드하거나 URL을 입력해주세요.")
        return (
            "입력이 없습니다.",
            None, None, None, "", ""
        )

    log_lines = []
    def status_cb(msg: str):
        log_lines.append(msg)
        progress(0, desc=msg)

    try:
        result = run_pipeline(
            source=source,
            whisper_model=whisper_model,
            clip_model=clip_model,
            scene_threshold=scene_threshold,
            status_cb=status_cb,
        )
    except Exception as e:
        err = f"오류 발생: {str(e)}"
        return err, None, None, None, "", ""

    vs = result.video_score
    prep = result.preprocess_result

    # 결과 빌드
    gauge_fig = build_gauge_chart(vs)
    timeline_fig = build_timeline_chart(vs)
    frame_html = build_frame_comparison_html(vs, prep.keyframes)

    # 텍스트 요약
    summary_md = f"""
### 분석 결과

**판정:** {vs.label}  
**불일치 점수:** {vs.overall_score:.1f} / 100  
**신뢰도:** {vs.confidence * 100:.0f}%  
**분석 프레임 수:** {len(vs.frame_scores)}개  
**이상 프레임:** {vs.anomaly_count}개  
**의심 구간:** {len(vs.suspicious_intervals)}개  

---

{vs.summary}
""".strip()

    stt_preview = prep.stt_text[:800] + ("..." if len(prep.stt_text) > 800 else "")
    log_text = "\n".join(log_lines)

    return summary_md, gauge_fig, timeline_fig, frame_html, stt_preview, log_text


# ──────────────────────────────────────────────
# Gradio UI 레이아웃
# ──────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Cheapfake 탐지 시스템",
        theme=gr.themes.Soft(primary_hue="teal"),
    ) as demo:

        gr.Markdown("""
# Cheapfake 탐지 시스템
**멀티모달 장면 이해 기반 시사 숏폼 영상 맥락 불일치 탐지**  
영상의 시각 장면과 텍스트 설명(자막·나레이션) 사이의 의미적 불일치를 CLIP 기반으로 분석합니다.
        """)

        with gr.Row():
            # ── 왼쪽: 입력 패널 ──
            with gr.Column(scale=1):
                gr.Markdown("### 입력")
                video_input = gr.Video(
                    label="영상 파일 업로드",
                    height=200,
                )
                url_input = gr.Textbox(
                    label="또는 영상 URL 입력",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                )

                gr.Markdown("### 분석 설정")
                with gr.Accordion("고급 설정", open=False):
                    whisper_model = gr.Dropdown(
                        label="Whisper 모델 (STT)",
                        choices=["tiny", "base", "small", "medium"],
                        value="base",
                        info="클수록 정확하지만 느림. GPU 없으면 base 권장.",
                    )
                    clip_model = gr.Dropdown(
                        label="CLIP 모델",
                        choices=["ViT-B/32", "ViT-L/14"],
                        value="ViT-L/14",
                        info="ViT-L/14 가 더 정확. GPU 없으면 ViT-B/32 권장.",
                    )
                    scene_threshold = gr.Slider(
                        label="장면 감지 민감도",
                        minimum=10,
                        maximum=50,
                        value=27,
                        step=1,
                        info="낮을수록 더 많은 키프레임 추출.",
                    )

                analyze_btn = gr.Button("분석 시작", variant="primary", size="lg")
                log_output = gr.Textbox(
                    label="진행 로그",
                    lines=5,
                    interactive=False,
                    placeholder="분석 시작 후 진행 상황이 여기 표시됩니다...",
                )

            # ── 오른쪽: 결과 패널 ──
            with gr.Column(scale=2):
                gr.Markdown("### 분석 결과")
                summary_output = gr.Markdown(
                    value="분석 결과가 여기에 표시됩니다.",
                )

                with gr.Row():
                    gauge_output = gr.Plot(label="불일치 점수 게이지")
                    pass

                timeline_output = gr.Plot(label="프레임별 불일치 타임라인")

                gr.Markdown("#### 주요 의심 프레임")
                frame_html_output = gr.HTML(
                    value="<p style='color:#999;font-size:13px;'>분석 후 의심 프레임이 여기에 표시됩니다.</p>"
                )

                with gr.Accordion("음성 인식 텍스트 (STT)", open=False):
                    stt_output = gr.Textbox(
                        label="STT 결과 (미리보기)",
                        lines=4,
                        interactive=False,
                    )

        # ── 이벤트 연결 ──
        analyze_btn.click(
            fn=analyze,
            inputs=[
                video_input,
                url_input,
                whisper_model,
                clip_model,
                scene_threshold,
            ],
            outputs=[
                summary_output,
                gauge_output,
                timeline_output,
                frame_html_output,
                stt_output,
                log_output,
            ],
        )

        gr.Markdown("""
---
<small>졸업 프로젝트 | 멀티모달 장면 이해와 메트릭 러닝을 활용한 시사 숏폼 영상의 맥락 불일치 탐지</small>
        """)

    return demo


# ──────────────────────────────────────────────
# 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="공개 링크 생성")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
