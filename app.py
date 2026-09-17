import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import urllib.parse

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
    .news-card {
        background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px 16px; margin-bottom: 8px;
    }
    .news-title {
        font-weight: 600; font-size: 1rem; color: #1e40af; text-decoration: none;
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
    url = f"https://m.stock.naver.com/api/stock/{clean_code}/trend?pageSize=60"
    try:
        res = requests.get(url, headers=headers, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list):
                for row in data:
                    biz_date = row.get("bizdate")
                    if not biz_date:
                        continue
                    date = datetime.strptime(str(biz_date), "%Y%m%d")
                    fore_net = int(str(row.get("foreignerPureBuyQuant", 0)).replace(",", ""))
                    inst_net = int(str(row.get("organPureBuyQuant", 0)).replace(",", ""))
                    records.append({"Date": date, "외국인순매수": fore_net, "기관순매수": inst_net})
    except Exception:
        pass
    if records:
        return pd.DataFrame(records).drop_duplicates("Date").set_index("Date").sort_index()
    return None

# 2. 종목 관련 뉴스 수집 함수
@st.cache_data(ttl=600)
def fetch_stock_news(keyword, date_str=None):
    clean_keyword = keyword.replace("🇰🇷 ", "").replace("🇺🇸 ", "").split("(")[0].strip()
    encoded = urllib.parse.quote(clean_keyword)
    headers = {"User-Agent": "Mozilla/5.0"}
    news_items = []
    
    # 구글 금융 뉴스 RSS 피드 (국내/해외 공통 지원)
    url = f"https://news.google.com/rss/search?q={encoded}+주가+when:14d&hl=ko&gl=KR&ceid=KR:ko"
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, "xml")
        items = soup.find_all("item")[:5]
        for it in items:
            title = it.title.text if it.title else ""
            link = it.link.text if it.link else "#"
            pub_date = it.pubDate.text[:16] if it.pubDate else ""
            news_items.append({"title": title, "link": link, "date": pub_date})
    except Exception:
        pass
    return news_items

# 3. 기술 지표 계산
def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA60'] = df['Close'].rolling(window=60).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

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

period = st.sidebar.select_slider("조회 기간", options=["3mo", "6mo", "1y"], value="6mo")

with st.sidebar.expander("⚙️ 관심 종목 편집", expanded=False):
    n_name = st.text_input("종목명 (예: 셀트리온)")
    n_code = st.text_input("티커 (예: 068270.KS)")
    if st.button("➕ 추가"):
        if n_name and n_code:
            st.session_state.watchlist[n_name] = n_code.upper()
            st.rerun()

# 5. 데이터 로드 및 정렬
df = yf.download(ticker, period=period, progress=False)
if df.empty:
    st.error("데이터를 가져오지 못했습니다.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
df = calculate_indicators(df)

has_supply = False
if is_korean:
    df_supply = fetch_korean_investor_trading(ticker)
    if df_supply is not None and not df_supply.empty:
        df_supply.index = pd.to_datetime(df_supply.index).tz_localize(None).normalize()
        df = df.join(df_supply, how="left")
        df['외국인순매수'] = df['외국인순매수'].fillna(0)
        df['기관순매수'] = df['기관순매수'].fillna(0)
        has_supply = True

latest = df.iloc[-1]
prev = df.iloc[-2]
recent_date_str = df.index[-1].strftime("%Y-%m-%d")

# 매매 추정 기준 가격
buy_target = float(df['High'].iloc[-20:].max())
stop_target = float(df['Low'].iloc[-20:].min())

# 지표 분석 및 브리핑
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
    reasons.append("단기 이평선이 수렴하며 박스권 방향성을 탐색하는 구간입니다.")

if rsi_val <= 35:
    score += 15
    reasons.append(f"RSI 수치가 {rsi_val:.1f}로 과매도(바닥권) 영역에 있어 기술적 반등 확률이 높습니다.")
elif rsi_val >= 70:
    score -= 15
    reasons.append(f"RSI 수치가 {rsi_val:.1f}로 과열권에 진입하여 단기 숨고르기 매물 출회가 우려됩니다.")
else:
    reasons.append(f"RSI는 {rsi_val:.1f}로 안정적인 흐름을 유지하고 있습니다.")

if has_supply:
    last3_fore = df['외국인순매수'].iloc[-3:].sum()
    last3_inst = df['기관순매수'].iloc[-3:].sum()
    if last3_fore > 0 and last3_inst > 0:
        score += 20
        reasons.append(f"최근 3일간 **외국인(+{last3_fore:,.0f}주)과 기관(+{last3_inst:,.0f}주)의 쌍끌이 순매수**가 유입되었습니다.")
    elif last3_fore < 0 and last3_inst < 0:
        score -= 20
        reasons.append(f"최근 3일간 **외국인({last3_fore:,.0f}주)과 기관({last3_inst:,.0f}주)의 쌍끌이 순매도**로 매물 부담이 있습니다.")

if score >= 75:
    final_verdict = "🟢 강력 매수 (STRONG BUY)"
    banner_class = "banner-strong-buy"
    action_text = "추세, 수급, 심리가 정배열을 형성했습니다. 분할 매수를 권장합니다."
elif score >= 60:
    final_verdict = "🟢 매수 고려 (BUY)"
    banner_class = "banner-buy"
    action_text = "상승 압력이 우세합니다. 지지선 가격대를 체크하며 진입을 검토하세요."
elif score <= 35:
    final_verdict = "🔴 적극 매도 / 손절 (SELL)"
    banner_class = "banner-sell"
    action_text = "수급 이탈과 추세 붕괴가 겹쳤습니다. 현금 확보 및 손절 대응이 필요합니다."
else:
    final_verdict = "🟡 매매 보류 / 관망 (HOLD)"
    banner_class = "banner-hold"
    action_text = "돌파 또는 지지 여부를 확인할 때까지 신규 매매를 멈추고 관망하는 것이 유리합니다."

# 6. 상단 요약 배너 및 추정 가격 가이드
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

st.markdown("### 🎯 실전 매매 추정 가격 가이드라인")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="metric-box" style="border-top: 4px solid #16a34a;">
        <div style="color: #64748b; font-size: 0.85rem;">🟢 향후 돌파 매수가</div>
        <div style="font-size: 1.4rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{buy_target:,.0f} {unit}</div>
        <div style="font-size: 0.8rem; color: #475569;">20일 최고가 돌파 마감 시 추가 매수</div>
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

st.markdown("### 📝 AI 수급 & 시황 브리핑")
briefing_html = f"<b>[{recent_date_str} 기준 시황]</b><br>" + "<br>".join([f"• {r}" for r in reasons])
st.markdown(f'<div class="briefing-box">{briefing_html}</div>', unsafe_allow_html=True)

# 7. 인터랙티브 차트
rows_cnt = 3 if has_supply else 2
row_heights = [0.52, 0.20, 0.28] if has_supply else [0.70, 0.30]

fig = make_subplots(
    rows=rows_cnt, cols=1, shared_xaxes=True, vertical_spacing=0.06,
    row_heights=row_heights,
    subplot_titles=(["주가 및 매매 기준선 (💡 캔들을 클릭하면 해당 날짜 수급과 뉴스가 뜹니다)", "RSI 지표 (과매수 70 / 과매도 30)", "외국인 / 기관 일별 순매수 수량 (주)"] if has_supply else ["주가 차트", "RSI"])
)

fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="주가"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)
fig.add_hline(y=buy_target, line_dash="dash", line_color="#16a34a", line_width=1.5,
              annotation_text=f"▲ 매수: {buy_target:,.0f}", annotation_position="top right", row=1, col=1)
fig.add_hline(y=stop_target, line_dash="dash", line_color="#dc2626", line_width=1.5,
              annotation_text=f"▼ 손절: {stop_target:,.0f}", annotation_position="bottom right", row=1, col=1)

fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.5), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="blue", row=2, col=1)

if has_supply:
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

fig.update_xaxes(type='date', tickformat='%Y-%m-%d', dtick="M1", showgrid=True, gridcolor="#f1f5f9")
fig.update_layout(
    height=850, margin=dict(l=10, r=10, t=30, b=30),
    xaxis_rangeslider_visible=False, template="plotly_white",
    hovermode="x unified", legend=dict(orientation="h", y=1.03)
)

# 캔들/바 클릭 인터랙션 감지
chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points"])

# 8. 캔들 클릭 시 해당 일자 상세 분석 & 뉴스 표시
clicked_date = None
if chart_event and chart_event.get("selection") and chart_event["selection"].get("points"):
    clicked_point = chart_event["selection"]["points"][0]
    clicked_date_str = str(clicked_point.get("x", "")).split("T")[0].split(" ")[0]
    try:
        clicked_date = pd.to_datetime(clicked_date_str)
    except Exception:
        pass

# 기본값은 최신 날짜
if clicked_date is None or clicked_date not in df.index:
    clicked_date = df.index[-1]

target_row = df.loc[clicked_date]
target_date_formatted = clicked_date.strftime("%Y년 %m월 %d일")

st.markdown("---")
st.markdown(f"### 🔍 [{target_date_formatted}] 상세 수급 분석 및 관련 뉴스")

# 클릭 일자 수급 요약 박스
col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    c_close = target_row['Close']
    st.markdown(f"""<div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">당일 종가</div>
        <div style="font-size: 1.3rem; font-weight: bold; margin-top: 4px;">{c_close:,.0f} {unit}</div>
    </div>""", unsafe_allow_html=True)

with col_s2:
    f_val = target_row.get('외국인순매수', 0) if has_supply else 0
    f_color = "#3b82f6" if f_val >= 0 else "#ef4444"
    st.markdown(f"""<div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">외국인 순매매</div>
        <div style="font-size: 1.3rem; font-weight: bold; color: {f_color}; margin-top: 4px;">{f_val:+,.0f} 주</div>
    </div>""", unsafe_allow_html=True)

with col_s3:
    i_val = target_row.get('기관순매수', 0) if has_supply else 0
    i_color = "#f97316" if i_val >= 0 else "#ef4444"
    st.markdown(f"""<div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">기관 순매매</div>
        <div style="font-size: 1.3rem; font-weight: bold; color: {i_color}; margin-top: 4px;">{i_val:+,.0f} 주</div>
    </div>""", unsafe_allow_html=True)

# 관련 뉴스 카드 렌더링
st.markdown(f"#### 📰 [{selected_name}] 시장 핵심 뉴스")
news_list = fetch_stock_news(selected_name, target_date_formatted)
if news_list:
    for n in news_list:
        st.markdown(f"""
        <div class="news-card">
            <a class="news-title" href="{n['link']}" target="_blank">🔗 {n['title']}</a>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 4px;">발행일시: {n['date']}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("해당 일자 전후의 관련 뉴스를 조회 중입니다.")
