import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.set_page_config(page_title="AI 종합 주식 매매 분석기", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .metric-box {
        background-color: #f8fafc; border-radius: 10px; padding: 14px;
        border: 1px solid #cbd5e1; text-align: center; margin-bottom: 10px;
    }
    .price-box {
        background-color: #ffffff; border-radius: 10px; padding: 16px;
        text-align: center; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    .signal-banner {
        border-radius: 10px; padding: 14px 18px; margin: 12px 0; font-weight: bold;
    }
    .banner-strong-buy { background-color: #dcfce7; color: #15803d; border-left: 6px solid #16a34a; }
    .banner-buy { background-color: #f0fdf4; color: #166534; border-left: 6px solid #22c55e; }
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
    url = f"https://m.stock.naver.com/api/stock/{clean_code}/trend?pageSize=100"
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

# 3. 과거 유사 패턴 사후 통계 백테스트 엔진
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
    for i in range(20, len(full_df) - 20):
        if i == idx_loc:
            continue
        c_row = full_df.iloc[i]
        rsi_match = abs(c_row['RSI'] - t_rsi) <= 7
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
    return {
        "count": len(similar_outcomes_20d),
        "win_rate": (len(wins) / len(similar_outcomes_20d)) * 100,
        "avg_return": float(np.mean(similar_outcomes_20d)),
        "max_return": float(np.max(similar_outcomes_20d)),
        "min_return": float(np.min(similar_outcomes_20d))
    }

# 4. 사이드바 구성
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

st.sidebar.title("📌 종목 및 기간 설정")
stock_options = list(st.session_state.watchlist.keys())
selected_name = st.sidebar.selectbox("감시 종목 선택", stock_options)
ticker = st.session_state.watchlist[selected_name]
is_korean = ".KS" in ticker or ".KQ" in ticker
unit = "원" if is_korean else "$"

# 5년치 조회를 기본값으로 설정
selected_period = st.sidebar.select_slider(
    "조회 기간 선택",
    options=["1y", "2y", "3y", "5y", "max"],
    value="5y"
)

with st.sidebar.expander("⚙️ 관심 종목 편집 (추가 / 삭제)", expanded=False):
    st.markdown("**종목 추가**")
    new_name = st.text_input("종목명 (예: 에코프로, 구글)", key="add_name")
    new_code = st.text_input("티커 심볼 (예: 086520.KQ, GOOGL)", key="add_code")
    if st.button("➕ 종목 추가", use_container_width=True):
        if new_name.strip() and new_code.strip():
            st.session_state.watchlist[new_name.strip()] = new_code.strip().upper()
            st.success(f"'{new_name}' 추가 완료!")
            st.rerun()
    st.markdown("---")
    st.markdown("**선택 종목 삭제**")
    if st.button(f"🗑️ '{selected_name}' 삭제", use_container_width=True):
        if len(st.session_state.watchlist) > 1:
            del st.session_state.watchlist[selected_name]
            st.success(f"'{selected_name}' 삭제 완료!")
            st.rerun()
        else:
            st.error("최소 1개 이상의 종목은 남아있어야 합니다.")

# 5. 데이터 다운로드 (선택 기간 전체)
df = yf.download(ticker, period=selected_period, progress=False)
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

st.title(f"📈 {selected_name} {selected_period.upper()} 종합 투자 타이밍 분석기")

# 날짜 선택 슬라이더 (5년치 전체 인덱스 지원)
date_list = list(df.index)
selected_date = st.select_slider(
    "📅 분석 날짜 선택 (과거부터 현재까지 슬라이더를 움직여 당시 매매 기준 가격과 통계를 확인하세요):",
    options=date_list,
    value=date_list[-1]
)

# 6. 선택 날짜 기준 매매 기준 가격 계산
target_row = df.loc[selected_date]
target_date_formatted = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%Y년 %m월 %d일")
loc_idx = df.index.get_loc(selected_date)

sub_df = df.iloc[:loc_idx+1]
window = min(len(sub_df), 20)

buy_price = float(sub_df['High'].iloc[-window:].max())
stop_price = float(sub_df['Low'].iloc[-window:].min())

c_close = float(target_row['Close'])
s20 = float(target_row['SMA20']) if not pd.isna(target_row['SMA20']) else c_close
s60 = float(target_row['SMA60']) if not pd.isna(target_row['SMA60']) else c_close
rsi = float(target_row['RSI']) if not pd.isna(target_row['RSI']) else 50.0

# 7. 차트 렌더링 (5년치 전체 데이터 표시)
rows_cnt = 3 if has_supply else 2
row_heights = [0.55, 0.20, 0.25] if has_supply else [0.70, 0.30]

fig = make_subplots(
    rows=rows_cnt, cols=1, shared_xaxes=False, vertical_spacing=0.10,
    row_heights=row_heights,
    subplot_titles=(["주가 및 매매 기준선", "RSI 지표 (과매수 70 / 과매도 30)", "외국인 / 기관 일별 순매수 (주)"] if has_supply else ["주가 차트", "RSI 지표"])
)

fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="주가"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)

fig.add_hline(y=buy_price, line_dash="dash", line_color="#16a34a", line_width=1.5,
              annotation_text=f"▲ 매수가: {buy_price:,.0f}", row=1, col=1)
fig.add_hline(y=stop_price, line_dash="dash", line_color="#dc2626", line_width=1.5,
              annotation_text=f"▼ 매도가: {stop_price:,.0f}", row=1, col=1)

# 선택 날짜 보라색 수직선
fig.add_vline(x=selected_date, line_width=2, line_dash="dot", line_color="#8b5cf6", row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.8), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#2563eb", row=2, col=1)

if has_supply:
    fig.add_trace(go.Bar(x=df.index, y=df['외국인순매수'], name="외국인", marker_color="#3b82f6"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['기관순매수'], name="기관", marker_color="#f97316"), row=3, col=1)

fig.update_xaxes(type='category', showticklabels=True, showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True, nticks=12)
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True)
fig.update_layout(height=850 if has_supply else 680, margin=dict(l=15, r=15, t=35, b=25), xaxis_rangeslider_visible=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified", legend=dict(orientation="h", y=1.03))

st.plotly_chart(fig, use_container_width=True)

# 8. 투자 판단 알고리즘
score = 50
reasons = []

if c_close > s20 and s20 > s60:
    score += 25
    reasons.append("주가가 20일 및 60일선 위에 안착한 **상승 정배열 추세**입니다.")
elif c_close < s20 and s20 < s60:
    score -= 25
    reasons.append("주가가 이동평균선 아래로 내려앉은 **하락 추세**입니다.")
else:
    reasons.append("방향성을 탐색하는 **박스권 횡보 구간**입니다.")

if rsi <= 35:
    score += 15
    reasons.append(f"RSI 수치가 **{rsi:.1f}**로 과매도(바닥권) 영역에 있어 기술적 반등 확률이 높습니다.")
elif rsi >= 70:
    score -= 15
    reasons.append(f"RSI 수치가 **{rsi:.1f}**로 과열권에 진입해 단기 차익 실현 매물 출회에 유의해야 합니다.")
else:
    reasons.append(f"RSI는 **{rsi:.1f}**로 심리적 과열/침체 없이 안정적입니다.")

if has_supply:
    f_net = target_row.get('외국인순매수', 0)
    i_net = target_row.get('기관순매수', 0)
    if f_net > 0 and i_net > 0:
        score += 20
        reasons.append(f"**외국인(+{f_net:,.0f}주)과 기관(+{i_net:,.0f}주)의 쌍끌이 순매수**가 유입되었습니다.")
    elif f_net < 0 and i_net < 0:
        score -= 20
        reasons.append(f"**외국인({f_net:,.0f}주)과 기관({i_net:,.0f}주)의 동반 순매도**로 매물 부담이 큽니다.")
    elif f_net > 0:
        score += 10
        reasons.append(f"**외국인이 +{f_net:,.0f}주 순매수**하며 하방을 방어 중입니다.")
    elif i_net > 0:
        score += 10
        reasons.append(f"**기관이 +{i_net:,.0f}주 순매수**로 방어선을 구축 중입니다.")

if score >= 75:
    verdict = "🟢 강력 매수 (STRONG BUY)"
    b_class = "banner-strong-buy"
    action = "추세, 심리, 수급이 일치합니다. 매수 진입 또는 비중 확대를 적극 검토하세요."
elif score >= 60:
    verdict = "🟢 매수 고려 (BUY)"
    b_class = "banner-buy"
    action = "상승 모멘텀이 유효합니다. 지지선 확인 후 분할 매수를 검토할 수 있습니다."
elif score <= 35:
    verdict = "🔴 적극 매도 / 손절 (SELL)"
    b_class = "banner-sell"
    action = "추세 붕괴 및 수급 이탈이 겹쳤습니다. 현금 확보 및 손절을 권장합니다."
else:
    verdict = "🟡 매매 보류 / 관망 (HOLD)"
    b_class = "banner-hold"
    action = "방향성이 뚜렷하지 않은 혼조세입니다. 명확한 돌파 전까지 신규 진입을 멈추고 관망하세요."

# 9. 결과 출력
st.markdown("---")
st.markdown(f"## 🔍 [{target_date_formatted}] 종합 투자 판단 및 매매 기준 가격")

st.markdown(f"""
<div class="signal-banner {b_class}">
    <div style="font-size: 1.25rem;">{verdict} (모멘텀 점수: {score}점/100점)</div>
    <div style="margin-top: 4px; font-weight: normal;">{action}</div>
</div>
""", unsafe_allow_html=True)

fmt = "{:,.0f}" if is_korean else "{:,.2f}"

st.markdown(f"### 🎯 [{target_date_formatted}] 기준 실전 매매 가격 가이드라인")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #16a34a;">
        <div style="color: #16a34a; font-weight: bold; font-size: 0.95rem;">🟢 추천 매수 기준가</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{fmt.format(buy_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">20일 최고 저항선 상향 돌파 시 추가 매수</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #d97706;">
        <div style="color: #d97706; font-weight: bold; font-size: 0.95rem;">🟡 매매 보류 / 관망 구간</div>
        <div style="font-size: 1.3rem; font-weight: bold; color: #d97706; margin: 8px 0;">
            {fmt.format(stop_price)} ~ {fmt.format(buy_price)} {unit}
        </div>
        <div style="font-size: 0.8rem; color: #64748b;">박스권 내에서는 신규 매매 없이 보유 포지션 유지</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #dc2626;">
        <div style="color: #dc2626; font-weight: bold; font-size: 0.95rem;">🔴 추천 매도 / 손절가</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{fmt.format(stop_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">20일 최저 지지선 하향 이탈 시 전량 손절/매도</div>
    </div>
    """, unsafe_allow_html=True)

# 통계 백테스트
stats = find_similar_patterns_outcome(df, selected_date, has_supply)
if stats and stats['count'] > 0:
    st.markdown(f"### 📊 과거 동일 조건 발생 시 실측 통계 ({selected_period.upper()} 데이터 기준)")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f"""<div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">동일 패턴 발생 횟수</div>
            <div style="font-size: 1.4rem; font-weight: bold; margin-top: 4px;">{stats['count']} 회</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        w_col = "#16a34a" if stats['win_rate'] >= 60 else ("#dc2626" if stats['win_rate'] <= 40 else "#d97706")
        st.markdown(f"""<div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">20영업일 뒤 상승 확률(승률)</div>
            <div style="font-size: 1.4rem; font-weight: bold; color: {w_col}; margin-top: 4px;">{stats['win_rate']:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        r_col = "#16a34a" if stats['avg_return'] > 0 else "#dc2626"
        st.markdown(f"""<div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">1개월 뒤 평균 실측 수익률</div>
            <div style="font-size: 1.4rem; font-weight: bold; color: {r_col}; margin-top: 4px;">{stats['avg_return']:+.2f}%</div>
        </div>""", unsafe_allow_html=True)

# 브리핑 요약
st.markdown("### 📝 AI 시황 브리핑 요약")
briefing_body = "<br>".join([f"• {r}" for r in reasons])
st.markdown(f"""
<div class="briefing-box">
    <b>[{target_date_formatted} 진단]</b><br>
    {briefing_body}
</div>
""", unsafe_allow_html=True)
