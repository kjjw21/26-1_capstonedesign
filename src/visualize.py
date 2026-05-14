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
    keyframes,
    top_n: int = 5,
) -> str:
    sorted_fs = sorted(
        video_score.frame_scores,
        key=lambda x: x.mismatch_score,
        reverse=True,
    )[:top_n]

    kf_map = {kf.index: kf for kf in keyframes}
    avg_sim = video_score.avg_similarity

    cards_html = []
    detail_panels = []

    for i, fs in enumerate(sorted_fs):
        kf = kf_map.get(fs.index)
        if not kf:
            continue

        badge_bg = _score_to_badge_bg(fs.mismatch_score)
        ocr = kf.ocr_text or "(화면 텍스트 없음)"
        anomaly_badge = '<span style="background:#E24B4A;color:white;font-size:10px;padding:1px 5px;border-radius:4px;margin-left:3px;">이상</span>' if fs.is_anomaly else ''

        # 의심 이유 (끊기지 않는 완성된 문장)
        if fs.mismatch_score >= 65:
            reason = (
                f"이 구간({fs.timestamp:.1f}초)의 시각-텍스트 유사도는 {fs.similarity:.4f}로, "
                f"전체 평균({avg_sim:.4f})보다 크게 낮습니다. "
                f"화면에 표시된 내용과 음성/자막의 내용이 의미적으로 심각하게 불일치하여 "
                f"맥락 조작 가능성이 높습니다."
            )
        elif fs.mismatch_score >= 40:
            reason = (
                f"이 구간({fs.timestamp:.1f}초)의 시각-텍스트 유사도는 {fs.similarity:.4f}로, "
                f"전체 평균({avg_sim:.4f})보다 낮습니다. "
                f"화면 내용과 음성 내용 사이에 부분적인 맥락 불일치가 감지되었습니다. "
                f"원본 영상과 비교 확인이 필요합니다."
            )
        else:
            reason = (
                f"이 구간({fs.timestamp:.1f}초)의 불일치 점수는 {fs.mismatch_score:.0f}점으로 "
                f"비교적 낮은 수준입니다. "
                f"시각-텍스트 유사도({fs.similarity:.4f})는 평균({avg_sim:.4f})과 유사한 수준이나 "
                f"상위 의심 프레임으로 분류되어 참고용으로 표시됩니다."
            )

        sim_dir = '▼ 평균보다 낮음' if fs.similarity < avg_sim else '▲ 평균보다 높음'
        sim_color = '#E24B4A' if fs.similarity < avg_sim else '#1D9E75'

        try:
            import base64
            with open(kf.path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
            img_src = f"data:image/jpeg;base64,{img_b64}"
            thumb = f'<img src="{img_src}" style="width:100%;border-radius:6px 6px 0 0;display:block;">'
            full_img = f'<img src="{img_src}" style="width:100%;max-height:300px;object-fit:contain;background:#000;display:block;border-radius:8px 0 0 8px;">'
        except Exception:
            thumb = '<div style="height:80px;background:#eee;border-radius:6px 6px 0 0;"></div>'
            full_img = ""
            img_src = ""

        # 작은 카드 (원래 크기 유지)
        cards_html.append(f"""
        <div onclick="showDetail({i})" id="card-{i}"
             style="border:1.5px solid #ddd;border-radius:8px;overflow:hidden;cursor:pointer;transition:border-color .2s;font-size:13px;">
          {thumb}
          <div style="padding:7px 9px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
              <span style="color:#666;font-size:12px;">{fs.timestamp:.1f}초{anomaly_badge}</span>
              <span style="background:{badge_bg};color:white;padding:1px 7px;border-radius:12px;font-size:11px;font-weight:500;">{fs.mismatch_score:.0f}점</span>
            </div>
            <div style="color:#999;font-size:11px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{ocr[:35]}{'...' if len(ocr)>35 else ''}</div>
          </div>
        </div>""")

        # 상세 패널 데이터 (그리드 아래에 하나만 표시)
        detail_panels.append({
            "img": img_src,
            "time": f"{fs.timestamp:.1f}초",
            "score": f"{fs.mismatch_score:.0f}",
            "badge_bg": badge_bg,
            "ocr": ocr[:400],
            "sim": f"{fs.similarity:.4f}",
            "sim_dir": sim_dir,
            "sim_color": sim_color,
            "avg_sim": f"{avg_sim:.4f}",
            "reason": reason,
            "anomaly": fs.is_anomaly,
        })

    # 상세 패널 HTML 미리 생성
    panels_html = []
    for i, d in enumerate(detail_panels):
        anomaly_tag = '<span style="background:#E24B4A;color:white;padding:1px 7px;border-radius:10px;font-size:11px;margin-left:4px;">⚠️ 이상 감지</span>' if d["anomaly"] else ''
        panels_html.append(f"""
        <div id="panel-{i}" style="display:none;">
          <div style="display:flex;flex-wrap:wrap;border:1.5px solid {d['badge_bg']};border-radius:10px;overflow:hidden;background:#fafafa;">
            <div style="flex:0 0 200px;min-width:140px;">{f'<img src="{d["img"]}" style="width:100%;height:100%;max-height:260px;object-fit:contain;background:#000;display:block;">' if d["img"] else ''}</div>
            <div style="flex:1;padding:13px 15px;min-width:180px;">
              <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px;flex-wrap:wrap;">
                <span style="font-weight:500;font-size:14px;">{d["time"]} 구간</span>
                <span style="background:{d['badge_bg']};color:white;padding:2px 9px;border-radius:10px;font-size:12px;">{d["score"]}점</span>
                {anomaly_tag}
              </div>
              <div style="margin-bottom:8px;">
                <div style="font-size:11px;color:#888;margin-bottom:2px;">화면 텍스트 (OCR)</div>
                <div style="font-size:12px;background:#f0f0f0;padding:5px 8px;border-radius:5px;line-height:1.6;color:#333;">{d["ocr"] or "(텍스트 없음)"}</div>
              </div>
              <div style="margin-bottom:10px;">
                <div style="font-size:11px;color:#888;margin-bottom:2px;">시각-텍스트 유사도</div>
                <div style="font-size:12px;color:#333;">{d["sim"]} <span style="color:{d['sim_color']};font-size:11px;">{d["sim_dir"]} (평균 {d["avg_sim"]})</span></div>
              </div>
              <div style="background:#fff8e1;border-left:3px solid #EF9F27;padding:8px 10px;border-radius:0 6px 6px 0;">
                <div style="font-size:11px;color:#888;margin-bottom:3px;">이 프레임이 의심되는 이유</div>
                <div style="font-size:12px;color:#333;line-height:1.7;">{d["reason"]}</div>
              </div>
            </div>
          </div>
        </div>""")

    html = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:10px;margin-bottom:4px;" id="frame-grid">
      {''.join(cards_html)}
    </div>
    <div id="detail-area" style="margin-top:8px;">
      {''.join(panels_html)}
    </div>
    <script>
    (function() {{
        var curIdx = null;
        window.showDetail = function(i) {{
            if (curIdx !== null) {{
                var oldPanel = document.getElementById('panel-' + curIdx);
                var oldCard = document.getElementById('card-' + curIdx);
                if (oldPanel) oldPanel.style.display = 'none';
                if (oldCard) oldCard.style.borderColor = '#ddd';
            }}
            if (curIdx === i) {{
                curIdx = null;
                return;
            }}
            var panel = document.getElementById('panel-' + i);
            var card = document.getElementById('card-' + i);
            if (panel) panel.style.display = 'block';
            if (card) card.style.borderColor = '#378ADD';
            curIdx = i;
        }};
    }})();
    </script>
    """
    return html


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