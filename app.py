import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.set_page_config(page_title="AI 패턴 통계 기반 주식 매매기", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-box {
        background-color: #f8fafc; border-radius: 10px; padding: 14px;
        border: 1px solid #cbd5e1; text-align: center; margin-bottom: 10px;
    }
    .stat-card {
        background-color: #ffffff; border-radius: 10px; padding: 16px;
        border: 1px solid #e2e8f0; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .signal-banner {
        border-radius: 10px; padding: 14px 18px; margin: 12px 0; font-weight: bold;
    }
    .banner-buy { background-color: #dcfce7; color: #15803d; border-left: 6px solid #16a34a; }
    .banner-hold { background-color: #fef3c7; color: #b45309; border-left: 6px solid #f59e0b; }
    .banner-sell { background-color: #fee2e2; color: #991b1b; border-left: 6px solid #ef4444; }
    .briefing-box {
        background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
        padding: 14px 18px; font-size: 0.95rem; line-height: 1.6; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# 1. 네이버 모바일 API 수급 수집
@st.cache_data(ttl=300)
def fetch_korean_investor_trading(code):
    clean_code = code.replace(".KS", "").replace(".KQ", "")
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15",
        "Referer": f"https://m.stock.naver.com/item/{clean_code}/trend"
    }
    records = []
    url = f"https://m.stock.naver.com/api/stock/{clean_code}/trend?pageSize=80"
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for row in data:
                    biz_date = row.get("bizdate")
                    if not biz_date:
                        continue
                    date = datetime.strptime(str(biz_date), "%Y%m%d").strftime("%Y-%m-%d")
                    fore_net = int(str(row.get("foreignerPureBuyQuant", 0)).replace(",", ""))
                    inst_net = int(str(row.get("organPureBuyQuant", 0)).replace(",", ""))
                    records.append({"Date": date, "외국인순매수": fore_net, "기관순매수": inst_net})
    except Exception:
        pass
    if records:
        return pd.DataFrame(records).drop_duplicates("Date").set_index("Date").sort_index()
    return None

# 2. 기술 지표 계산
def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 3. 과거 유사 패턴 매칭 및 사후 성과 백테스트 엔진
def find_similar_patterns_outcome(full_df, target_date_str, has_sup):
    idx_loc = full_df.index.get_loc(target_date_str)
    t_row = full_df.iloc[idx_loc]
    
    t_rsi = t_row['RSI']
    t_trend = 1 if t_row['Close'] >= t_row['SMA20'] else -1
    t_sup = 0
    if has_sup:
        f_val = t_row.get('외국인순매수', 0)
        i_val = t_row.get('기관순매수', 0)
        t_sup = 1 if (f_val > 0 or i_val > 0) else -1

    similar_outcomes_20d = []
    
    # 과거 전체 봉을 스캔하며 유사 조건 탐색 (최소 20거래일 뒤 결과가 존재하는 봉)
    for i in range(20, len(full_df) - 20):
        if i == idx_loc:
            continue
        c_row = full_df.iloc[i]
        
        # 조건 1: RSI 오차범위 ±7
        rsi_match = abs(c_row['RSI'] - t_rsi) <= 7
        # 조건 2: 20일선 대비 주가 위치 일치
        c_trend = 1 if c_row['Close'] >= c_row['SMA20'] else -1
        trend_match = (c_trend == t_trend)
        
        sup_match = True
        if has_sup:
            c_f = c_row.get('외국인순매수', 0)
            c_i = c_row.get('기관순매수', 0)
            c_sup = 1 if (c_f > 0 or c_i > 0) else -1
            sup_match = (c_sup == t_sup)
            
        if rsi_match and trend_match and sup_match:
            base_p = c_row['Close']
            after_20d_p = full_df.iloc[i + 20]['Close']
            pct_return = ((after_20d_p - base_p) / base_p) * 100
            similar_outcomes_20d.append(pct_return)

    if not similar_outcomes_20d:
        return None

    wins = [r for r in similar_outcomes_20d if r > 0]
    win_rate = (len(wins) / len(similar_outcomes_20d)) * 100
    avg_return = np.mean(similar_outcomes_20d)
    max_return = np.max(similar_outcomes_20d)
    min_return = np.min(similar_outcomes_20d)

    return {
        "count": len(similar_outcomes_20d),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "max_return": max_return,
        "min_return": min_return
    }

# 4. 사이드바
DEFAULT_STOCKS = {
    "🇰🇷 삼성전자": "005930.KS",
    "🇰🇷 SK하이닉스": "000660.KS",
    "🇰🇷 현대차": "005380.KS",
    "🇰🇷 NAVER": "035420.KS",
    "🇰🇷 카카오": "035720.KS",
    "🇰🇷 알테오젠": "196170.KQ",
    "🇺🇸 애플": "AAPL",
    "🇺🇸 엔비디아": "NVDA",
    "🇺🇸 테슬라": "TSLA"
}

if "watchlist" not in st.session_state:
    st.session_state.watchlist = DEFAULT_STOCKS.copy()

st.sidebar.title("📌 종목 선택")
selected_name = st.sidebar.selectbox("감시 종목", list(st.session_state.watchlist.keys()))
ticker = st.session_state.watchlist[selected_name]
is_korean = ".KS" in ticker or ".KQ" in ticker
unit = "원" if is_korean else "$"

# 5. 충분한 통계 표본을 위해 2년치 데이터 다운로드
df = yf.download(ticker, period="2y", progress=False)
if df.empty:
    st.error("데이터를 불러오지 못했습니다.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")
df = calculate_indicators(df)

has_supply = False
if is_korean:
    df_supply = fetch_korean_investor_trading(ticker)
    if df_supply is not None and not df_supply.empty:
        df = df.join(df_supply, how="left")
        df['외국인순매수'] = df['외국인순매수'].fillna(0)
        df['기관순매수'] = df['기관순매수'].fillna(0)
        has_supply = True

# 차트에는 최근 130거래일(약 6개월)만 쾌적하게 렌더링
plot_df = df.tail(130)

buy_target = float(plot_df['High'].iloc[-20:].max())
stop_target = float(plot_df['Low'].iloc[-20:].min())

st.title(f"📈 {selected_name} 과거 통계 기반 투자 타이밍 분석기")
st.markdown("💡 **과거 캔들을 클릭하면, '그 당시와 똑같았던 과거 패턴들은 20영업일 뒤 어떻게 되었는가?'에 대한 실제 통계 승률과 현재 적용 가이드를 보여줍니다.**")

# 6. 차트 생성
rows_cnt = 3 if has_supply else 2
row_heights = [0.55, 0.20, 0.25] if has_supply else [0.70, 0.30]

fig = make_subplots(
    rows=rows_cnt, cols=1, shared_xaxes=False, vertical_spacing=0.10,
    row_heights=row_heights,
    subplot_titles=(["주가 및 매매 기준선", "RSI 지표 (과매수 70 / 과매도 30)", "외국인 / 기관 일별 순매수 수량 (주)"] if has_supply else ["주가 차트 및 매매 기준선", "RSI 지표"])
)

fig.add_trace(go.Candlestick(
    x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'],
    increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="주가"
), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)
fig.add_hline(y=buy_target, line_dash="dash", line_color="#16a34a", line_width=1.5, annotation_text=f"▲ 매수가: {buy_target:,.0f}", row=1, col=1)
fig.add_hline(y=stop_target, line_dash="dash", line_color="#dc2626", line_width=1.5, annotation_text=f"▼ 손절가: {stop_target:,.0f}", row=1, col=1)

fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI'], line=dict(color='#8b5cf6', width=1.8), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#2563eb", row=2, col=1)

if has_supply:
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['외국인순매수'], name="외국인", marker_color="#3b82f6"), row=3, col=1)
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['기관순매수'], name="기관", marker_color="#f97316"), row=3, col=1)

fig.update_xaxes(type='category', showticklabels=True, showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True, nticks=10)
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True)
fig.update_layout(height=880 if has_supply else 700, margin=dict(l=15, r=15, t=35, b=25), xaxis_rangeslider_visible=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified", legend=dict(orientation="h", y=1.03))

chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points"])

# 7. 캔들 클릭 날짜 포착
clicked_date_str = None
if chart_event and chart_event.get("selection") and chart_event["selection"].get("points"):
    clicked_point = chart_event["selection"]["points"][0]
    raw_x = str(clicked_point.get("x", ""))
    clicked_date_str = raw_x.split("T")[0].split(" ")[0].strip()

if not clicked_date_str or clicked_date_str not in df.index:
    clicked_date_str = plot_df.index[-1]

target_row = df.loc[clicked_date_str]
target_date_formatted = datetime.strptime(clicked_date_str, "%Y-%m-%d").strftime("%Y년 %m월 %d일")

# 현재 최신 영업일 정보
today_row = df.iloc[-1]
today_date_formatted = datetime.strptime(df.index[-1], "%Y-%m-%d").strftime("%Y년 %m월 %d일")
today_close = float(today_row['Close'])
target_close = float(target_row['Close'])
since_diff_pct = ((today_close - target_close) / target_close) * 100

# 8. 과거 유사 패턴 사후 통계 계산
stats = find_similar_patterns_outcome(df, clicked_date_str, has_supply)

st.markdown("---")
st.markdown(f"## 📊 [{target_date_formatted}] 캔들 기준 과거 기록 매칭 분석")

# 과거 통계 카드 3분할
s1, s2, s3 = st.columns(3)
if stats and stats['count'] > 0:
    with s1:
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid #3b82f6;">
            <div style="color: #64748b; font-size: 0.85rem;">과거 유사 패턴 발견 횟수</div>
            <div style="font-size: 1.6rem; font-weight: bold; color: #1e40af; margin: 6px 0;">{stats['count']} 회</div>
            <div style="font-size: 0.8rem; color: #64748b;">최근 2년간 동일 조건 발생 표본</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        win_color = "#16a34a" if stats['win_rate'] >= 60 else ("#dc2626" if stats['win_rate'] <= 40 else "#d97706")
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid {win_color};">
            <div style="color: #64748b; font-size: 0.85rem;">20영업일(1개월) 뒤 상승 확률(승률)</div>
            <div style="font-size: 1.6rem; font-weight: bold; color: {win_color}; margin: 6px 0;">{stats['win_rate']:.1f}%</div>
            <div style="font-size: 0.8rem; color: #64748b;">과거 유사 시점 매수 후 수익 확률</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        ret_color = "#16a34a" if stats['avg_return'] > 0 else "#dc2626"
        st.markdown(f"""
        <div class="stat-card" style="border-top: 4px solid {ret_color};">
            <div style="color: #64748b; font-size: 0.85rem;">1개월 뒤 평균 실측 수익률</div>
            <div style="font-size: 1.6rem; font-weight: bold; color: {ret_color}; margin: 6px 0;">{stats['avg_return']:+.2f}%</div>
            <div style="font-size: 0.8rem; color: #64748b;">최고 {stats['max_return']:+.1f}% / 최저 {stats['min_return']:+.1f}%</div>
        </div>""", unsafe_allow_html=True)
else:
    st.info("비교할 수 있는 충분한 과거 유사 패턴 표본이 부족합니다.")

# 9. 현재 상황과의 비교 및 실전 매매 타이밍 조언
st.markdown(f"### 🎯 지금 현재 시점({today_date_formatted})과의 비교 및 투자 커맨드")

fmt = "{:,.0f}" if is_korean else "{:,.2f}"

# 현재 포지션 진단
c_now = today_close
s20_now = today_row['SMA20']
s60_now = today_row['SMA60']
rsi_now = today_row['RSI']

if stats and stats['win_rate'] >= 65 and stats['avg_return'] > 3.0:
    verdict_text = "🟢 적극 매수 타이밍 (상승 확률 우세)"
    banner_c = "banner-buy"
    action_guide = f"선택하신 날({target_date_formatted})과 같은 조건은 과거 <b>상승 승률 {stats['win_rate']:.1f}%</b>를 기록한 강력한 패턴입니다. 현재 주가({fmt.format(c_now)}{unit}) 역시 해당 모멘텀의 영향권 내에 있다면 <b>신규 매수 또는 비중 확대</b>가 통계적으로 유리합니다."
elif stats and stats['win_rate'] <= 40:
    verdict_text = "🔴 매도 / 리스크 회피 타이밍 (하락 확률 우세)"
    banner_c = "banner-sell"
    action_guide = f"선택하신 날의 조건은 과거 20영업일 뒤 <b>하락 확률이 {100 - stats['win_rate']:.1f}%</b>로 손실 위험이 높았던 구간입니다. 현재 시점에서는 욕심을 내기보다 <b>비중 축소나 손절 기준가 엄수</b>가 필요합니다."
else:
    verdict_text = "🟡 매매 보류 / 관망 (방향성 중립)"
    banner_c = "banner-hold"
    action_guide = "과거 유사 사례에서 주가의 상승과 하락이 팽팽하게 엇갈렸던 자리입니다. 상단 저항선 돌파가 확인될 때까지 신규 진입을 미루는 것이 통계적으로 안전합니다."

st.markdown(f"""
<div class="signal-banner {banner_c}">
    <div style="font-size: 1.25rem;">{verdict_text}</div>
    <div style="margin-top: 6px; font-weight: normal; line-height: 1.6;">{action_guide}</div>
</div>
""", unsafe_allow_html=True)

# 가격 비교 요약
c_left, c_right = st.columns(2)
with c_left:
    st.markdown(f"""
    <div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">선택 날짜({target_date_formatted}) 당시 종가</div>
        <div style="font-size: 1.4rem; font-weight: bold; margin-top: 4px;">{fmt.format(target_close)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">당시 RSI: {target_row['RSI']:.1f}</div>
    </div>""", unsafe_allow_html=True)

with c_right:
    st.markdown(f"""
    <div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">지금 현재({today_date_formatted}) 주가 및 수익률</div>
        <div style="font-size: 1.4rem; font-weight: bold; margin-top: 4px; color: {'#ef4444' if since_diff_pct > 0 else '#3b82f6'};">
            {fmt.format(c_now)} {unit} ({since_diff_pct:+.2f}%)
        </div>
        <div style="font-size: 0.8rem; color: #64748b;">선택일 이후 현재까지의 누적 변동폭</div>
    </div>""", unsafe_allow_html=True)
