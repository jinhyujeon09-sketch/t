import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 끝
MIN_DAYS = 300        # 연간 최소 관측일 수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_yearly():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df = df.drop_duplicates(subset=["날짜"])
    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    return yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온 추세로 미래 기온을 예측해 봅니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(yearly) < 3:
    st.error("분석에 쓸 수 있는 연도가 부족합니다.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 최근 20년(끝 연도 포함 20개 연도 범위)만으로 구한 직선
RECENT_SPAN = 20
recent = yearly[yearly["연도"] > end_year - RECENT_SPAN]
has_recent = len(recent) >= 3
if has_recent:
    r_slope, r_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    r_corr = float(np.corrcoef(recent["연도"], recent["연평균기온"])[0, 1])

# 1. 예측
year = st.slider("예측할 연도", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">{year}년 서울 예상 연평균기온</div>
      <div style="font-size:4rem; font-weight:700; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 2. 그래프
x_line = np.array([1900, 2100], dtype=float)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=slope * x_line + intercept,
        mode="lines",
        name="회귀 직선",
        line=dict(width=2.5, dash="dash"),
    )
)
if has_recent:
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=r_slope * x_line + r_intercept,
            mode="lines",
            name=f"최근 {RECENT_SPAN}년 직선",
            line=dict(width=2, dash="dot"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예측",
        marker=dict(size=14, symbol="diamond", line=dict(width=2, color="white")),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    xaxis=dict(range=[1900, 2100]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
    height=450,
)
st.plotly_chart(fig, use_container_width=True)

# 3. 분석 정보
st.markdown(
    f"""
    <div style="text-align:center; padding:0.8rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">전체 기간 추세: 100년에</div>
      <div style="font-size:3.5rem; font-weight:700; line-height:1.2;">{slope * 100:+.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"**전체 기간** ({start_year}~{end_year}년, {n_years}개 연도)")
    st.markdown(
        f"<div style='font-size:2.2rem; font-weight:700;'>{slope * 100:+.2f} °C / 100년</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"연간 {slope:+.4f} °C · 상관계수 r = {r:.3f}")
with c2:
    if has_recent:
        r_start = int(recent["연도"].min())
        st.markdown(f"**최근 20년** ({r_start}~{end_year}년, {len(recent)}개 연도)")
        st.markdown(
            f"<div style='font-size:2.2rem; font-weight:700;'>{r_slope * 100:+.2f} °C / 100년</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"연간 {r_slope:+.4f} °C · 상관계수 r = {r_corr:.3f}")
    else:
        st.warning("최근 20년 구간에 쓸 수 있는 연도가 부족합니다.")

if has_recent:
    st.caption(
        f"최근 20년 기울기는 전체 기간의 {r_slope / slope:.1f}배입니다. "
        "표본이 20개 안팎이라 해마다 변동의 영향을 크게 받습니다."
    )

st.info(
    f"회귀 직선은 **{n_years}개 연도**의 자료로 만들었습니다 "
    f"(시작 연도 **{start_year}년**, 끝 연도 **{end_year}년**). "
    f"{LAST_YEAR}년 이후 자료와 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)
st.caption(f"회귀식: 기온 = {slope:.5f} × 연도 + {intercept:.3f}")
