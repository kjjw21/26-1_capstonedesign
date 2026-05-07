"""
visualize.py
분석 결과 시각화 모듈.

- 프레임별 불일치 점수 타임라인 히트맵 (Plotly)
- 의심 구간 하이라이트
- Gradio 에서 바로 사용 가능한 Figure 반환
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .score import VideoScore, FrameScore


# ──────────────────────────────────────────────
# 색상 팔레트
# ──────────────────────────────────────────────

RISK_COLORS = {
    "normal":    "#1D9E75",   # teal
    "suspicious": "#EF9F27",  # amber
    "high_risk":  "#E24B4A",  # red
}

HEATMAP_COLORSCALE = [
    [0.0,  "#1D9E75"],  # 낮은 불일치 = 초록
    [0.4,  "#EF9F27"],  # 중간 = 노랑
    [1.0,  "#E24B4A"],  # 높은 불일치 = 빨강
]


# ──────────────────────────────────────────────
# 타임라인 차트
# ──────────────────────────────────────────────

def build_timeline_chart(video_score: VideoScore) -> go.Figure:
    """
    프레임별 불일치 점수를 시간축 위에 시각화한다.
    의심 구간은 배경 음영으로 하이라이트.

    Returns
    -------
    plotly Figure — Gradio Plot 컴포넌트에 바로 전달 가능
    """
    fs_list = video_score.frame_scores
    if not fs_list:
        fig = go.Figure()
        fig.update_layout(title="분석할 프레임이 없습니다")
        return fig

    timestamps = [fs.timestamp for fs in fs_list]
    mismatch = [fs.mismatch_score for fs in fs_list]
    similarities = [fs.similarity for fs in fs_list]
    anomaly_flags = [fs.is_anomaly for fs in fs_list]

    # 색상: 이상치 프레임은 더 진하게
    marker_colors = [
        RISK_COLORS["high_risk"] if a else _score_to_color(m)
        for a, m in zip(anomaly_flags, mismatch)
    ]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("프레임별 불일치 점수", "영상-텍스트 유사도"),
    )

    # ── Row 1: 불일치 점수 바 차트 ──
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=mismatch,
            marker_color=marker_colors,
            name="불일치 점수",
            hovertemplate="시간: %{x:.1f}s<br>불일치: %{y:.1f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 임계선 (40, 65)
    for thresh, label, color in [
        (40, "정상/의심 경계", "#EF9F27"),
        (65, "의심/위험 경계", "#E24B4A"),
    ]:
        fig.add_hline(
            y=thresh,
            line_dash="dash",
            line_color=color,
            line_width=1,
            annotation_text=label,
            annotation_position="right",
            row=1, col=1,
        )

    # ── Row 2: 유사도 라인 차트 ──
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=similarities,
            mode="lines+markers",
            line=dict(color="#378ADD", width=2),
            marker=dict(
                size=8,
                color=marker_colors,
                line=dict(color="white", width=1),
            ),
            name="CLIP 유사도",
            hovertemplate="시간: %{x:.1f}s<br>유사도: %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── 의심 구간 음영 ──
    for interval in video_score.suspicious_intervals:
        for row in [1, 2]:
            fig.add_vrect(
                x0=interval["start"],
                x1=interval["end"],
                fillcolor="#E24B4A",
                opacity=0.10,
                layer="below",
                line_width=0,
                row=row, col=1,
            )

    # ── 레이아웃 ──
    fig.update_layout(
        height=420,
        margin=dict(l=50, r=80, t=60, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(
        title_text="시간 (초)",
        row=2, col=1,
        gridcolor="rgba(128,128,128,0.2)",
    )
    fig.update_yaxes(
        title_text="점수",
        range=[0, 105],
        row=1, col=1,
        gridcolor="rgba(128,128,128,0.2)",
    )
    fig.update_yaxes(
        title_text="유사도",
        row=2, col=1,
        gridcolor="rgba(128,128,128,0.2)",
    )

    return fig


# ──────────────────────────────────────────────
# 게이지 차트 (전체 점수)
# ──────────────────────────────────────────────

def build_gauge_chart(video_score: VideoScore) -> go.Figure:
    """
    전체 불일치 점수를 게이지로 표시한다.
    """
    score = video_score.overall_score
    color = RISK_COLORS.get(video_score.label_en, "#888")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=score,
            number={"suffix": " / 100", "font": {"size": 28}},
            title={"text": f"불일치 점수<br><span style='font-size:16px;color:{color}'>{video_score.label}</span>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(29,158,117,0.15)"},
                    {"range": [40, 65],  "color": "rgba(239,159,39,0.15)"},
                    {"range": [65, 100], "color": "rgba(226,75,74,0.15)"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.85,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=30, r=30, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


# ──────────────────────────────────────────────
# 키프레임 비교 HTML
# ──────────────────────────────────────────────

def build_frame_comparison_html(
    video_score: VideoScore,
    keyframes,  # List[Keyframe]
    top_n: int = 5,
) -> str:
    """
    불일치 점수 상위 N 개 프레임의 이미지와 OCR 텍스트를 HTML 카드로 반환.
    Gradio HTML 컴포넌트에 전달.
    """
    sorted_fs = sorted(
        video_score.frame_scores,
        key=lambda x: x.mismatch_score,
        reverse=True,
    )[:top_n]

    kf_map = {kf.index: kf for kf in keyframes}

    cards = []
    for fs in sorted_fs:
        kf = kf_map.get(fs.index)
        if not kf:
            continue

        color = _score_to_color(fs.mismatch_score)
        badge_bg = _score_to_badge_bg(fs.mismatch_score)
        ocr = kf.ocr_text or "(OCR 텍스트 없음)"

        # 이미지를 base64 로 인코딩해 inline embed
        try:
            import base64
            with open(kf.path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_tag = f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%;border-radius:6px 6px 0 0;display:block;">'
        except Exception:
            img_tag = '<div style="height:80px;background:#eee;border-radius:6px 6px 0 0;"></div>'

        cards.append(f"""
        <div style="border:1px solid #ddd;border-radius:8px;overflow:hidden;font-family:sans-serif;font-size:13px;">
          {img_tag}
          <div style="padding:8px 10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
              <span style="color:#666;">{fs.timestamp:.1f}초</span>
              <span style="background:{badge_bg};color:white;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:500;">
                {fs.mismatch_score:.0f}점
              </span>
            </div>
            <div style="color:#444;line-height:1.4;word-break:break-all;">{ocr[:100]}</div>
          </div>
        </div>""")

    grid_html = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;padding:4px 0;">
      {''.join(cards)}
    </div>"""
    return grid_html


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _score_to_color(score: float) -> str:
    if score < 40:
        return RISK_COLORS["normal"]
    elif score < 65:
        return RISK_COLORS["suspicious"]
    return RISK_COLORS["high_risk"]


def _score_to_badge_bg(score: float) -> str:
    return _score_to_color(score)
