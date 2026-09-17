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
        background-color: #f8fafc; border-radius: 12px; padding: 14px;
        border: 1px solid #e2e8f0; text-align: center; margin-bottom: 12px;
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

# 1. 네이버 증권 외국인 / 기관 일별 수급 크롤링
@st.cache_data(ttl=300)
def fetch_korean_investor_trading(code):
    clean_code = code.replace(".KS", "").replace(".KQ", "")
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    for page in range(1, 3):
        url = f"https://finance.naver.com/item/frgn.naver?code={clean_code}&page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            soup = BeautifulSoup(res.text, "html.parser")
            tables = soup.select("table.type2")
            if len(tables) < 2:
                continue
            rows = tables[1].select("tr")
            for r in rows:
                cols = r.select("td")
                if len(cols) >= 9:
                    date_text = cols[0].text.strip()
                    if not date_text or "." not in date_text:
                        continue
                    try:
                        date = datetime.strptime(date_text, "%Y.%m.%d")
                        inst_net = int(cols[5].text.strip().replace(",", ""))
                        fore_net = int(cols[6].text.strip().replace(",", ""))
                        records.append({"Date": date, "기관순매수": inst_net, "외국인순매수": fore_net})
                    except Exception:
                        continue
        except Exception:
            pass
    if records:
        df_supply = pd.DataFrame(records).drop_duplicates("Date").set_index("Date").sort_index()
        return df_supply
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

# 3. 사이드바 구성
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

period = st.sidebar.select_slider("조회 기간", options=["3mo", "6mo", "1y"], value="6mo")

with st.sidebar.expander("⚙️ 관심 종목 편집", expanded=False):
    n_name = st.text_input("종목명 (예: 셀트리온)")
    n_code = st.text_input("티커 (예: 068270.KS)")
    if st.button("➕ 추가"):
        if n_name and n_code:
            st.session_state.watchlist[n_name] = n_code.upper()
            st.rerun()

# 4. 데이터 로드 및 결합
df = yf.download(ticker, period=period, progress=False)
if df.empty:
    st.error("데이터를 가져오지 못했습니다.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df = calculate_indicators(df)

if is_korean:
    df_supply = fetch_korean_investor_trading(ticker)
    if df_supply is not None:
        df = df.join(df_supply, how="left")
        df['외국인순매수'] = df['외국인순매수'].fillna(0)
        df['기관순매수'] = df['기관순매수'].fillna(0)

latest = df.iloc[-1]
prev = df.iloc[-2]
recent_date_str = df.index[-1].strftime("%Y-%m-%d")

# 매매 추정 기준 가격 (최근 20일 기준)
buy_target = float(df['High'].iloc[-20:].max())
stop_target = float(df['Low'].iloc[-20:].min())

# 점수 및 브리핑 생성
trend_bullish = latest['Close'] > latest['SMA20'] and latest['SMA20'] > latest['SMA60']
trend_bearish = latest['Close'] < latest['SMA20'] and latest['SMA20'] < latest['SMA60']
rsi_val = latest['RSI']

score = 50
reasons = []

if trend_bullish:
    score += 25
    reasons.append("주가가 20일선 및 60일선 위에 안착한 상승 정배열 추세입니다.")
elif trend_bearish:
    score -= 25
    reasons.append("주가가 주요 이동평균선 아래로 내려앉아 하락 추세가 지속되고 있습니다.")
else:
    reasons.append("단기 이평선이 수렴하며 박스권 방향성을 저울질하는 구간입니다.")

if rsi_val <= 35:
    score += 15
    reasons.append(f"RSI 수치가 {rsi_val:.1f}로 과매도(바닥권) 영역에 있어 기술적 반등 확률이 높습니다.")
elif rsi_val >= 70:
    score -= 15
    reasons.append(f"RSI 수치가 {rsi_val:.1f}로 과열권에 진입하여 단기 숨고르기 매물 출회가 우려됩니다.")
else:
    reasons.append(f"RSI는 {rsi_val:.1f}로 안정적인 흐름을 유지하고 있습니다.")

if is_korean and '외국인순매수' in df.columns:
    last3_fore = df['외국인순매수'].iloc[-3:].sum()
    last3_inst = df['기관순매수'].iloc[-3:].sum()
    
    if last3_fore > 0 and last3_inst > 0:
        score += 20
        reasons.append(f"최근 3일간 **외국인(+{last3_fore:,.0f}주)과 기관(+{last3_inst:,.0f}주)이 동반 순매수**하며 시세를 주도하고 있습니다.")
    elif last3_fore < 0 and last3_inst < 0:
        score -= 20
        reasons.append(f"최근 3일간 **외국인({last3_fore:,.0f}주)과 기관({last3_inst:,.0f}주)의 쌍끌이 순매도**로 수급 부담이 커진 상태입니다.")
    elif last3_fore > 0:
        score += 10
        reasons.append(f"기관 매도 속에서도 **외국인이 +{last3_fore:,.0f}주 순매수**하며 하방을 지지 중입니다.")
    elif last3_inst > 0:
        score += 10
        reasons.append(f"외인 이탈 속에서 **기관이 +{last3_inst:,.0f}주 순매수**로 방어선을 구축하고 있습니다.")

if score >= 75:
    final_verdict = "🟢 강력 매수 (STRONG BUY)"
    banner_class = "banner-strong-buy"
    action_text = "추세, 수급, 심리가 일치합니다. 매수 진입 또는 비중 확대를 고려할 수 있습니다."
elif score >= 60:
    final_verdict = "🟢 매수 고려 (BUY)"
    banner_class = "banner-buy"
    action_text = "상승 압력이 우세합니다. 지지선 가격대를 체크하며 분할 매수를 검토하세요."
elif score <= 35:
    final_verdict = "🔴 적극 매도 / 손절 (SELL)"
    banner_class = "banner-sell"
    action_text = "수급 이탈과 추세 하락이 겹쳤습니다. 현금 확보 및 손절 대응이 필요합니다."
else:
    final_verdict = "🟡 매매 보류 / 관망 (HOLD)"
    banner_class = "banner-hold"
    action_text = "돌파 또는 지지 여부를 확인할 때까지 신규 매매를 멈추고 관망하는 것이 유리합니다."

# 5. 대시보드 상단 정보
curr_price = latest['Close']
diff = curr_price - prev['Close']
pct_diff = (diff / prev['Close']) * 100
unit = "원" if is_korean else "$"

st.title(f"📈 {selected_name} 종합 진단 리포트")
st.write(f"**기준일:** `{recent_date_str}` | **현재 종가:** `{curr_price:,.0f} {unit}` ({diff:+,.0f} / {pct_diff:+.2f}%)")

st.markdown(f"""
<div class="signal-banner {banner_class}">
    <div style="font-size: 1.25rem;">{final_verdict} (모멘텀 점수: {score}점/100점)</div>
    <div style="margin-top: 4px; font-weight: normal;">{action_text}</div>
</div>
""", unsafe_allow_html=True)

# 미래 매매 추정가 가이드 카드
st.markdown("### 🎯 실전 매매 추정 가격 가이드라인")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #16a34a;">
        <div style="color: #64748b; font-size: 0.85rem;">🟢 향후 돌파 매수가</div>
        <div style="font-size: 1.4rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{buy_target:,.0f} {unit}</div>
        <div style="font-size: 0.8rem; color: #475569;">20일 최고가 상향 돌파 시 추가 매수</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #d97706;">
        <div style="color: #64748b; font-size: 0.85rem;">🟡 관망 및 보유 유지 구간</div>
        <div style="font-size: 1.25rem; font-weight: bold; color: #d97706; margin: 8px 0;">{stop_target:,.0f} ~ {buy_target:,.0f} {unit}</div>
        <div style="font-size: 0.8rem; color: #475569;">박스권 내 주가 형성 시 포지션 유지</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #dc2626;">
        <div style="color: #64748b; font-size: 0.85rem;">🔴 향후 이탈 손절가</div>
        <div style="font-size: 1.4rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{stop_target:,.0f} {unit}</div>
        <div style="font-size: 0.8rem; color: #475569;">20일 최저 지지선 이탈 시 전량 매도</div>
    </div>""", unsafe_allow_html=True)

# 시황 브리핑 박스
st.markdown("### 📝 AI 수급 & 시황 브리핑")
briefing_html = f"<b>[{recent_date_str} 기준 시황]</b><br>" + "<br>".join([f"• {r}" for r in reasons])
st.markdown(f'<div class="briefing-box">{briefing_html}</div>', unsafe_allow_html=True)

# 6. 차트 시각화
rows_cnt = 3 if is_korean else 2
row_heights = [0.52, 0.20, 0.28] if is_korean else [0.70, 0.30]

fig = make_subplots(
    rows=rows_cnt, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=row_heights,
    subplot_titles=(["주가 및 매매 기준선", "RSI 지표 (과매수 70 / 과매도 30)", "외국인 / 기관 일별 순매수 수량 (주)"] if is_korean else ["주가 차트", "RSI"])
)

# 1행: 캔들 + 이평선 + 추정 기준선
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="주가"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)
fig.add_hline(y=buy_target, line_dash="dash", line_color="#16a34a", line_width=1.5,
              annotation_text=f"▲ 매수 기준가: {buy_target:,.0f}", annotation_position="top right", row=1, col=1)
fig.add_hline(y=stop_target, line_dash="dash", line_color="#dc2626", line_width=1.5,
              annotation_text=f"▼ 손절 기준가: {stop_target:,.0f}", annotation_position="bottom right", row=1, col=1)

# 2행: RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.5), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)

# 3행: 수급 막대 그래프 + 수치 레이블(숫자 표시)
if is_korean and '외국인순매수' in df.columns:
    def format_qty(val):
        if abs(val) >= 10000:
            return f"{val/10000:+.1f}만"
        return f"{val:+,.0f}" if val != 0 else ""

    fore_texts = [format_qty(v) for v in df['외국인순매수']]
    inst_texts = [format_qty(v) for v in df['기관순매수']]

    fig.add_trace(go.Bar(
        x=df.index, y=df['외국인순매수'], name="외국인 순매수", marker_color="#3b82f6",
        text=fore_texts, textposition='outside'
    ), row=3, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df['기관순매수'], name="기관 순매수", marker_color="#f97316",
        text=inst_texts, textposition='outside'
    ), row=3, col=1)

# X축 날짜 서식 설정
fig.update_xaxes(
    type='date',
    tickformat='%Y-%m-%d',
    dtick="M1",
    showgrid=True,
    gridcolor="#f1f5f9"
)

fig.update_layout(
    height=850,
    margin=dict(l=10, r=10, t=30, b=30),
    xaxis_rangeslider_visible=False,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.03)
)

st.plotly_chart(fig, use_container_width=True)

# 7. 최근 일별 상세 수급 데이터 테이블
if is_korean and '외국인순매수' in df.columns:
    st.markdown("### 📊 최근 7영업일 수급 상세 수치")
    table_df = df[['Close', '외국인순매수', '기관순매수']].tail(7).copy()
    table_df.index = table_df.index.strftime('%Y-%m-%d')
    table_df.columns = ['종가 (원)', '외국인 순매수 (주)', '기관 순매수 (주)']
    
    st.dataframe(
        table_df.style.format({
            '종가 (원)': '{:,.0f}',
            '외국인 순매수 (주)': '{:+,.0f}',
            '기관 순매수 (주)': '{:+,.0f}'
        }),
        use_container_width=True
    )
