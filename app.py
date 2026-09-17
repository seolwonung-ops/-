import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="AI 주식 미래 매매 가이드", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-box {
        background-color: #f8fafc; border-radius: 12px; padding: 16px;
        border: 1px solid #e2e8f0; text-align: center; margin-bottom: 12px;
    }
    .badge-buy { background-color: #dcfce7; color: #15803d; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 1.05rem; }
    .badge-hold { background-color: #fef3c7; color: #b45309; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 1.05rem; }
    .badge-sell { background-color: #fee2e2; color: #b91c1c; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 1.05rem; }
</style>
""", unsafe_allow_html=True)

DEFAULT_STOCKS = {
    "🇰🇷 삼성전자": "005930.KS",
    "🇰🇷 SK하이닉스": "000660.KS",
    "🇰🇷 현대차": "005380.KS",
    "🇰🇷 NAVER": "035420.KS",
    "🇰🇷 카카오": "035720.KS",
    "🇺🇸 애플": "AAPL",
    "🇺🇸 엔비디아": "NVDA",
    "🇺🇸 테슬라": "TSLA",
    "🇺🇸 S&P500 ETF": "SPY"
}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_STOCKS.copy()

st.sidebar.title("📌 주식 분석 설정")
stock_list = list(st.session_state.watchlist.keys())
selected_name = st.sidebar.selectbox("감시 종목 선택", stock_list)
ticker = st.session_state.watchlist[selected_name]

period = st.sidebar.select_slider("조회 기간", options=["3mo", "6mo", "1y", "2y"], value="1y")

with st.sidebar.expander("⚙️ 관심 종목 편집 (추가 / 삭제)", expanded=False):
    new_name = st.text_input("종목 이름 (예: 구글)", key="new_name")
    new_ticker = st.text_input("티커 심볼 (예: GOOGL, 196170.KQ)", key="new_ticker")
    if st.button("➕ 종목 추가", use_container_width=True):
        if new_name.strip() and new_ticker.strip():
            st.session_state.watchlist[new_name.strip()] = new_ticker.strip().upper()
            st.success(f"'{new_name}' 추가 완료!")
            st.rerun()
    if st.button(f"🗑️ '{selected_name}' 삭제", use_container_width=True):
        if len(st.session_state.watchlist) > 1:
            del st.session_state.watchlist[selected_name]
            st.rerun()

@st.cache_data(ttl=300)
def get_data(symbol, period_val):
    df = yf.download(symbol, period=period_val, progress=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    return df

df = get_data(ticker, period)

if df is None or df.empty:
    st.error(f"'{ticker}' 데이터를 불러오지 못했습니다.")
else:
    curr_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    diff = curr_close - prev_close
    pct_diff = (diff / prev_close) * 100
    sma20 = float(df['SMA20'].iloc[-1])
    sma60 = float(df['SMA60'].iloc[-1])
    
    buy_target = float(df['High'].iloc[-20:].max())
    stop_target = float(df['Low'].iloc[-20:].min())

    if curr_close > sma20 and sma20 > sma60:
        badge = '<span class="badge-buy">🟢 적극 매수 권고 (상승 추세)</span>'
    elif curr_close < sma20 and curr_close < sma60:
        badge = '<span class="badge-sell">🔴 매도 권고 (하락 추세)</span>'
    elif curr_close >= sma20 and curr_close <= sma60:
        badge = '<span class="badge-hold">🟡 매매 보류 / 관망 (반등 모색)</span>'
    else:
        badge = '<span class="badge-hold">🟡 매매 보류 / 관망 (단기 눌림목)</span>'

    c1, c2 = st.columns([2, 1])
    curr_unit = "원" if ".KS" in ticker or ".KQ" in ticker else "$"
    with c1:
        st.subheader(f"📊 {selected_name} ({ticker})")
        st.write(f"**현재 종가:** `{curr_close:,.2f} {curr_unit}` ({diff:+,.2f} / {pct_diff:+.2f}%)")
    with c2:
        st.markdown(f"<div style='text-align: right; padding-top: 15px;'>{badge}</div>", unsafe_allow_html=True)

    st.markdown("### 🎯 앞으로의 미래 매매 가이드라인")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #16a34a;">
            <div style="color: #64748b;">🟢 향후 매수 타이밍</div>
            <div style="font-size: 1.4rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{buy_target:,.2f} {curr_unit}</div>
            <div style="font-size: 0.85rem; color: #475569;">20일 최고가 돌파 마감 시 매수</div>
        </div>""", unsafe_allow_html=True)
    with g2:
        st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #d97706;">
            <div style="color: #64748b;">🟡 보유 및 관망 구간</div>
            <div style="font-size: 1.25rem; font-weight: bold; color: #d97706; margin: 8px 0;">{stop_target:,.0f} ~ {buy_target:,.0f} {curr_unit}</div>
            <div style="font-size: 0.85rem; color: #475569;">박스권 내에서는 포지션 유지</div>
        </div>""", unsafe_allow_html=True)
    with g3:
        st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #dc2626;">
            <div style="color: #64748b;">🔴 향후 전량 매도 타이밍</div>
            <div style="font-size: 1.4rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{stop_target:,.2f} {curr_unit}</div>
            <div style="font-size: 0.85rem; color: #475569;">20일 최저 지지선 이탈 시 손절</div>
        </div>""", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.72, 0.28])
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가",
                                 increasing_line_color="#ef4444", decreasing_line_color="#3b82f6"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)
    fig.add_hline(y=buy_target, line_dash="dash", line_color="#16a34a", line_width=1.5, annotation_text=f"▲ 매수가: {buy_target:,.0f}", annotation_position="top right", row=1, col=1)
    fig.add_hline(y=stop_target, line_dash="dash", line_color="#dc2626", line_width=1.5, annotation_text=f"▼ 손절가: {stop_target:,.0f}", annotation_position="bottom right", row=1, col=1)

    v_colors = ['#ef4444' if r['Close'] >= r['Open'] else '#3b82f6' for _, r in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=v_colors, name="거래량"), row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10), xaxis_rangeslider_visible=False, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
