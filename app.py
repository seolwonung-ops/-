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
    .metric-card {
        background-color: #f8fafc; border-radius: 12px; padding: 16px;
        border: 1px solid #e2e8f0; text-align: center;
    }
    .signal-banner {
        border-radius: 10px; padding: 16px 20px; margin: 15px 0; font-weight: bold;
    }
    .banner-strong-buy { background-color: #dcfce7; color: #15803d; border-left: 6px solid #16a34a; }
    .banner-buy { background-color: #f0fdf4; color: #166534; border-left: 6px solid #22c55e; }
    .banner-hold { background-color: #fef3c7; color: #b45309; border-left: 6px solid #f59e0b; }
    .banner-sell { background-color: #fee2e2; color: #991b1b; border-left: 6px solid #ef4444; }
    .briefing-box {
        background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
        padding: 16px 20px; font-size: 0.95rem; line-height: 1.6; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 1. 국내 주식 네이버 외국인/기관 수급 크롤링 함수
@st.cache_data(ttl=600)
def fetch_korean_investor_trading(code):
    clean_code = code.replace(".KS", "").replace(".KQ", "")
    headers = {"User-Agent": "Mozilla/5.0"}
    records = []
    # 최근 2페이지 (약 40영업일) 수급 수집
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
                        inst_net = int(cols[5].text.strip().replace(",", ""))  # 기관
                        fore_net = int(cols[6].text.strip().replace(",", ""))  # 외국인
                        records.append({"Date": date, "기관순매수": inst_net, "외국인순매수": fore_net})
                    except Exception:
                        continue
        except Exception:
            pass
    if records:
        df_supply = pd.DataFrame(records).drop_duplicates("Date").set_index("Date").sort_index()
        return df_supply
    return None

# 2. 기술적 지표 계산 함수 (SMA, RSI)
def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 3. 사이드바 종목 설정
DEFAULT_STOCKS = {
    "🇰🇷 삼성전자": "005930.KS",
    "🇰🇷 SK하이닉스": "000660.KS",
    "🇰🇷 현대차": "005380.KS",
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

with st.sidebar.expander("⚙️ 관심 종목 편집", expanded=False):
    n_name = st.text_input("종목명 (예: NAVER)")
    n_code = st.text_input("티커 (예: 035420.KS)")
    if st.button("➕ 추가"):
        if n_name and n_code:
            st.session_state.watchlist[n_name] = n_code.upper()
            st.rerun()

# 4. 데이터 로드 및 결합
df = yf.download(ticker, period="6mo", progress=False)
if df.empty:
    st.error("데이터를 가져올 수 없습니다.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df = calculate_indicators(df)

# 국내 주식일 경우 수급 결합
df_supply = None
if is_korean:
    df_supply = fetch_korean_investor_trading(ticker)
    if df_supply is not None:
        df = df.join(df_supply, how="left")
        df['외국인순매수'] = df['외국인순매수'].fillna(0)
        df['기관순매수'] = df['기관순매수'].fillna(0)

# 5. 종합 점수 및 브리핑 생성 엔진
latest = df.iloc[-1]
prev = df.iloc[-2]
recent_date = df.index[-1].strftime("%m월 %d일")

# (1) 추세 분석
trend_bullish = latest['Close'] > latest['SMA20'] and latest['SMA20'] > latest['SMA60']
trend_bearish = latest['Close'] < latest['SMA20'] and latest['SMA20'] < latest['SMA60']

# (2) RSI 심리 분석
rsi_val = latest['RSI']
rsi_status = "과매수(조정 경계)" if rsi_val >= 70 else ("과매도(반등 기회)" if rsi_val <= 30 else "중립")

# (3) 수급 분석 (국내 주식 기준 최근 3일 누적)
reasons = []
score = 50  # 100점 만점 기준 점수화

if trend_bullish:
    score += 25
    reasons.append("이동평균선이 20일선 > 60일선 정배열로 강한 상승 추세에 안착했습니다.")
elif trend_bearish:
    score -= 25
    reasons.append("이동평균선이 20일선 및 60일선 아래로 이탈하며 하락 추세를 보이고 있습니다.")
else:
    reasons.append("단기 이동평균선이 수렴하며 방향성을 탐색하는 횡보 구간입니다.")

if rsi_val <= 35:
    score += 15
    reasons.append(f"RSI가 {rsi_val:.1f} 수준으로 바닥권(과매도)에 근접해 기술적 반등 가능성이 높습니다.")
elif rsi_val >= 70:
    score -= 15
    reasons.append(f"RSI가 {rsi_val:.1f}로 과열권에 진입하여 단기 차익 실현 매물이 출회될 수 있습니다.")
else:
    reasons.append(f"RSI 지수는 {rsi_val:.1f}로 과열이나 침체 없이 안정적인 심리를 유지 중입니다.")

if is_korean and '외국인순매수' in df.columns:
    last3_fore = df['외국인순매수'].iloc[-3:].sum()
    last3_inst = df['기관순매수'].iloc[-3:].sum()
    
    if last3_fore > 0 and last3_inst > 0:
        score += 20
        reasons.append(f"최근 3영업일간 **외국인(+{last3_fore:,.0f}주)과 기관(+{last3_inst:,.0f}주)의 쌍끌이 순매수 유입**이 주가 상승을 강하게 견인하고 있습니다.")
    elif last3_fore < 0 and last3_inst < 0:
        score -= 20
        reasons.append(f"최근 3영업일간 **외국인({last3_fore:,.0f}주)과 기관({last3_inst:,.0f}주)이 동반 순매도**하며 물량을 쏟아내고 있어 하방 압력이 큽니다.")
    elif last3_fore > 0:
        score += 10
        reasons.append(f"기관 매도세에도 불구하고 **외국인이 +{last3_fore:,.0f}주 순매수**하며 물량을 방어하고 있습니다.")
    elif last3_inst > 0:
        score += 10
        reasons.append(f"외국인 이탈에도 **국내 기관이 +{last3_inst:,.0f}주 순매수**로 지지선을 받치고 있습니다.")

# 최종 종합 판정
if score >= 75:
    final_verdict = "🟢 강력 매수 (STRONG BUY)"
    banner_class = "banner-strong-buy"
    action_text = "추세, 수급, 심리 지표가 모두 매수를 지지합니다. 조정 시 적극 분할 매수 타이밍입니다."
elif score >= 60:
    final_verdict = "🟢 매수 고려 (BUY)"
    banner_class = "banner-buy"
    action_text = "우상향 모멘텀이 유효합니다. 지지선 확인 후 매수 진입을 검토할 수 있습니다."
elif score <= 35:
    final_verdict = "🔴 적극 매도 / 손절 (SELL)"
    banner_class = "banner-sell"
    action_text = "이평선 붕괴 및 메이저 수급 이탈이 겹쳤습니다. 현금 확보 및 손절을 권장합니다."
else:
    final_verdict = "🟡 매매 보류 / 관망 (HOLD)"
    banner_class = "banner-hold"
    action_text = "방향성이 뚜렷하지 않은 혼조세입니다. 추가 수급 확인 전까지 무리한 진입을 보류하세요."

# 6. 화면 렌더링
curr_price = latest['Close']
prev_price = prev['Close']
chg = curr_price - prev_price
pct_chg = (chg / prev_price) * 100
unit = "원" if is_korean else "$"

st.title(f"📈 {selected_name} 종합 AI 투자 진단")
st.write(f"**기준일자:** {recent_date} | **현재가:** `{curr_price:,.0f} {unit}` ({chg:+,.0f} / {pct_chg:+.2f}%)")

# 상단 최종 커맨드 배너
st.markdown(f"""
<div class="signal-banner {banner_class}">
    <div style="font-size: 1.3rem;">{final_verdict} (종합 모멘텀 점수: {score}점/100점)</div>
    <div style="margin-top: 5px; font-weight: normal;">{action_text}</div>
</div>
""", unsafe_allow_html=True)

# 💡 자연어 실전 브리핑
st.markdown("### 📝 AI 종합 분석 브리핑")
briefing_html = f"<b>[{recent_date} 시황 요약]</b><br>" + "<br>".join([f"• {r}" for r in reasons])
st.markdown(f'<div class="briefing-box">{briefing_html}</div>', unsafe_allow_html=True)

# 3단 지표 카드
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div style="color: #64748b; font-size: 0.85rem;">20일 / 60일 추세</div>
        <div style="font-size: 1.2rem; font-weight: bold; margin-top: 6px;">{"상승 정배열" if trend_bullish else ("하락 역배열" if trend_bearish else "혼조/박스권")}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div style="color: #64748b; font-size: 0.85rem;">RSI (14) 심리</div>
        <div style="font-size: 1.2rem; font-weight: bold; margin-top: 6px;">{rsi_val:.1f} ({rsi_status})</div>
    </div>""", unsafe_allow_html=True)
with c3:
    supply_info = "미국주식(해외)"
    if is_korean and '외국인순매수' in df.columns:
        f_today = latest.get('외국인순매수', 0)
        i_today = latest.get('기관순매수', 0)
        supply_info = f"외인: {f_today:+,.0f}주 / 기관: {i_today:+,.0f}주"
    st.markdown(f"""<div class="metric-card">
        <div style="color: #64748b; font-size: 0.85rem;">당일 메이저 수급</div>
        <div style="font-size: 0.95rem; font-weight: bold; margin-top: 6px;">{supply_info}</div>
    </div>""", unsafe_allow_html=True)

# 7. 종합 멀티 서브플롯 차트 (1행: 캔들+이평선, 2행: RSI, 3행: 외국인/기관 수급)
rows_cnt = 3 if is_korean else 2
row_heights = [0.55, 0.20, 0.25] if is_korean else [0.70, 0.30]

fig = make_subplots(
    rows=rows_cnt, cols=1, shared_xaxes=True, vertical_spacing=0.04,
    row_heights=row_heights,
    subplot_titles=(["주가 및 이동평균선", "RSI 지표 (과매수 70 / 과매도 30)", "외국인 / 기관 일별 순매수 수급"] if is_korean else ["주가 및 이동평균선", "RSI 지표"])
)

# 1행: 주가 캔들 + 이평선
fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                             increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="캔들"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)

# 2행: RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.5), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)

# 3행: 수급 (국내 주식만)
if is_korean and '외국인순매수' in df.columns:
    fig.add_trace(go.Bar(x=df.index, y=df['외국인순매수'], name="외국인 순매수", marker_color="#3b82f6"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['기관순매수'], name="기관 순매수", marker_color="#f97316"), row=3, col=1)

fig.update_layout(height=800, xaxis_rangeslider_visible=False, template="plotly_white", legend=dict(orientation="h", y=1.03))
st.plotly_chart(fig, use_container_width=True)
