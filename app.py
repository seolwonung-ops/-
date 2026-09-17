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
        border: 1px solid #cbd5e1; text-align: center; margin-bottom: 10px;
    }
    .price-box {
        background-color: #ffffff; border-radius: 10px; padding: 16px 14px;
        text-align: center; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.06);
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
    .news-card {
        background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 8px;
    }
    .news-title { font-weight: 600; font-size: 0.95rem; color: #1e40af; text-decoration: none; }
</style>
""", unsafe_allow_html=True)

# 1. 네이버 모바일 API 수급 수집 (국내 주식 전용)
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
                    date = datetime.strptime(str(biz_date), "%Y%m%d").strftime("%Y-%m-%d")
                    fore_net = int(str(row.get("foreignerPureBuyQuant", 0)).replace(",", ""))
                    inst_net = int(str(row.get("organPureBuyQuant", 0)).replace(",", ""))
                    records.append({"Date": date, "외국인순매수": fore_net, "기관순매수": inst_net})
    except Exception:
        pass
    if records:
        return pd.DataFrame(records).drop_duplicates("Date").set_index("Date").sort_index()
    return None

# 2. 관련 뉴스 수집 (국내 / 미국 공통)
@st.cache_data(ttl=600)
def fetch_stock_news(keyword):
    clean_keyword = keyword.replace("🇰🇷 ", "").replace("🇺🇸 ", "").split("(")[0].strip()
    encoded = urllib.parse.quote(clean_keyword)
    headers = {"User-Agent": "Mozilla/5.0"}
    news_items = []
    url = f"https://news.google.com/rss/search?q={encoded}+주식+when:14d&hl=ko&gl=KR&ceid=KR:ko"
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.content, "xml")
        items = soup.find_all("item")[:4]
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
unit = "원" if is_korean else "$"

period = st.sidebar.select_slider("조회 기간", options=["3mo", "6mo", "1y"], value="6mo")

with st.sidebar.expander("⚙️ 관심 종목 편집", expanded=False):
    n_name = st.text_input("종목명 (예: 셀트리온)")
    n_code = st.text_input("티커 (예: 068270.KS)")
    if st.button("➕ 추가"):
        if n_name and n_code:
            st.session_state.watchlist[n_name] = n_code.upper()
            st.rerun()

# 5. 데이터 로드 및 타임존 완벽 정규화 (핵심 해결 포인트)
df = yf.download(ticker, period=period, progress=False)
if df.empty:
    st.error("데이터를 가져오지 못했습니다.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 미국 주식의 UTC 시차 오차를 완전히 없애기 위해 'YYYY-MM-DD' 문자열 인덱스로 통일
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

buy_target = float(df['High'].iloc[-20:].max())
stop_target = float(df['Low'].iloc[-20:].min())

# 6. 상단 안내
st.title(f"📈 {selected_name} 종합 진단 리포트")
st.markdown("💡 **국내 및 미국 주식 모두 캔들을 클릭하면 해당 일자의 매매 제시 가격과 종합 리포트가 아래에 즉시 열립니다.**")

# 7. 차트 렌더링
rows_cnt = 3 if has_supply else 2
row_heights = [0.55, 0.20, 0.25] if has_supply else [0.70, 0.30]

fig = make_subplots(
    rows=rows_cnt, cols=1, shared_xaxes=False, vertical_spacing=0.10,
    row_heights=row_heights,
    subplot_titles=(["주가 및 매매 기준선", "RSI 지표 (과매수 70 / 과매도 30)", "외국인 / 기관 일별 순매수 수량 (주)"] if has_supply else ["주가 차트 및 매매 기준선 (💡 캔들을 클릭해 보세요)", "RSI 지표 (과매수 70 / 과매도 30)"])
)

# 1행: 주가 캔들 + 이평선
fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="주가"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)
fig.add_hline(y=buy_target, line_dash="dash", line_color="#16a34a", line_width=1.5,
              annotation_text=f"▲ 매수가: {buy_target:,.2f}", annotation_position="top right", row=1, col=1)
fig.add_hline(y=stop_target, line_dash="dash", line_color="#dc2626", line_width=1.5,
              annotation_text=f"▼ 손절가: {stop_target:,.2f}", annotation_position="bottom right", row=1, col=1)

# 2행: RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.8), name="RSI"), row=2, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", line_width=1.2, row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#2563eb", line_width=1.2, row=2, col=1)

# 3행: 수급 (국내 주식)
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

# 사각 테두리 및 눈금선 설정
fig.update_xaxes(
    type='category',  # 문자열 인덱스를 완벽하게 매칭하기 위해 category 타입으로 설정
    showticklabels=True,
    showgrid=True,
    gridcolor="#e2e8f0",
    showline=True,
    linewidth=1.5,
    linecolor="#475569",
    mirror=True,
    nticks=10
)
fig.update_yaxes(
    showgrid=True,
    gridcolor="#e2e8f0",
    showline=True,
    linewidth=1.5,
    linecolor="#475569",
    mirror=True
)

fig.update_layout(
    height=900 if has_supply else 700,
    margin=dict(l=15, r=15, t=35, b=25),
    xaxis_rangeslider_visible=False,
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.03)
)

chart_event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", selection_mode=["points"])

# 8. 캔들 클릭 감지 (국내/미국 주식 완벽 호환 파싱)
clicked_date_str = None
if chart_event and chart_event.get("selection") and chart_event["selection"].get("points"):
    clicked_point = chart_event["selection"]["points"][0]
    raw_x = str(clicked_point.get("x", ""))
    # 2026-09-16T00:00:00 또는 2026-09-16 모두 YYYY-MM-DD 형태로 추출
    clicked_date_str = raw_x.split("T")[0].split(" ")[0].strip()

# 데이터프레임 인덱스에 존재하지 않으면 최신 날짜로 지정
if not clicked_date_str or clicked_date_str not in df.index:
    clicked_date_str = df.index[-1]

target_row = df.loc[clicked_date_str]
target_date_formatted = datetime.strptime(clicked_date_str, "%Y-%m-%d").strftime("%Y년 %m월 %d일")

loc_idx = df.index.get_loc(clicked_date_str)
if loc_idx > 0:
    prev_row = df.iloc[loc_idx - 1]
    day_diff = target_row['Close'] - prev_row['Close']
    day_pct = (day_diff / prev_row['Close']) * 100
else:
    day_diff, day_pct = 0, 0.0

# 9. 해당 일자 기준 매매 제시 가격 계산
sub_df = df.iloc[:loc_idx+1]
window = min(len(sub_df), 20)

calc_buy_target = float(sub_df['High'].iloc[-window:].max())
calc_stop_target = float(sub_df['Low'].iloc[-window:].min())

c_close = float(target_row['Close'])
s20 = float(target_row['SMA20']) if not pd.isna(target_row['SMA20']) else c_close
s60 = float(target_row['SMA60']) if not pd.isna(target_row['SMA60']) else c_close
rsi = float(target_row['RSI']) if not pd.isna(target_row['RSI']) else 50.0

# 1차 익절 목표가
target_profit_price = calc_buy_target * 1.07 if c_close >= calc_buy_target else c_close * 1.08

t_bull = c_close > s20 and s20 > s60
t_bear = c_close < s20 and s20 < s60

date_score = 50
date_reasons = []

if t_bull:
    date_score += 25
    date_reasons.append("주가가 20일 및 60일 이동평균선 위에 위치한 **상승 정배열 추세**를 보였습니다.")
elif t_bear:
    date_score -= 25
    date_reasons.append("주가가 주요 이동평균선 아래로 내려앉으며 **하락 역배열 추세**를 나타냈습니다.")
else:
    date_reasons.append("단기 이평선이 수렴하며 방향성을 탐색하는 **박스권 횡보 구간**이었습니다.")

if rsi <= 35:
    date_score += 15
    date_reasons.append(f"RSI 지표가 **{rsi:.1f}**로 과매도(바닥권) 영역에 진입하여 기술적 반등 확률이 높았던 자리입니다.")
elif rsi >= 70:
    date_score -= 15
    date_reasons.append(f"RSI 지표가 **{rsi:.1f}**로 과열권에 진입하여 차익 실현 매물 출회 가능성이 컸던 구간입니다.")
else:
    date_reasons.append(f"RSI는 **{rsi:.1f}**로 안정적인 심리 상태를 나타냈습니다.")

if has_supply:
    f_net = target_row.get('외국인순매수', 0)
    i_net = target_row.get('기관순매수', 0)
    if f_net > 0 and i_net > 0:
        date_score += 20
        date_reasons.append(f"**외국인(+{f_net:,.0f}주)과 기관(+{i_net:,.0f}주)의 쌍끌이 순매수**가 유입되어 상승 모멘텀을 형성했습니다.")
    elif f_net < 0 and i_net < 0:
        date_score -= 20
        date_reasons.append(f"**외국인({f_net:,.0f}주)과 기관({i_net:,.0f}주)의 동반 순매도**로 매물 부담이 컸던 날입니다.")
    elif f_net > 0:
        date_score += 10
        date_reasons.append(f"**외국인이 +{f_net:,.0f}주 순매수**하며 하방을 지탱했습니다.")
    elif i_net > 0:
        date_score += 10
        date_reasons.append(f"**국내 기관이 +{i_net:,.0f}주 순매수**로 방어선을 형성했습니다.")
else:
    # 미국 주식: 거래량 급증 여부 분석
    avg_vol = sub_df['Volume'].iloc[-20:].mean()
    curr_vol = target_row['Volume']
    if curr_vol > avg_vol * 1.5:
        date_score += 10
        date_reasons.append("평균 대비 **150% 이상의 대량 거래량**이 실리며 시장의 강한 관심을 입증했습니다.")

# 종합 판정
if date_score >= 75:
    date_verdict = "🟢 강력 매수 (STRONG BUY)"
    banner_cls = "banner-strong-buy"
    date_action = "추세, 수급, 심리가 삼박자로 일치했던 구간입니다. 적극적 매수 진입 또는 비중 확대를 취하기 유리한 시점이었습니다."
elif date_score >= 60:
    date_verdict = "🟢 매수 고려 (BUY)"
    banner_cls = "banner-buy"
    date_action = "상승 모멘텀이 우세했던 구간입니다. 분할 매수 관점의 진입이 유효했습니다."
elif date_score <= 35:
    date_verdict = "🔴 적극 매도 / 손절 (SELL)"
    banner_cls = "banner-sell"
    date_action = "수급 이탈과 추세 붕괴가 겹쳤던 자리입니다. 리스크 관리를 위한 비중 축소 및 손절이 권장되었던 시점입니다."
else:
    date_verdict = "🟡 매매 보류 / 관망 (HOLD)"
    banner_cls = "banner-hold"
    date_action = "추세의 방향성이 불명확했던 혼조세였습니다. 돌파 또는 지지 확인 전까지 관망하는 것이 현명한 자리였습니다."

# 10. 선택 날짜 리포트 & 구체적 매매 가격 제시
st.markdown("---")
st.markdown(f"## 🔍 [{target_date_formatted}] 종합 상세 진단 및 투자 의견")

st.markdown(f"""
<div class="signal-banner {banner_cls}">
    <div style="font-size: 1.25rem;">{date_verdict} (당일 모멘텀 점수: {date_score}점/100점)</div>
    <div style="margin-top: 4px; font-weight: normal;">{date_action}</div>
</div>
""", unsafe_allow_html=True)

# 실전 매매 가격 가이드 3분할 박스 (소수점 지원)
st.markdown(f"### 🎯 [{target_date_formatted}] 기준 추천 매매(매수/매도) 가격")
p1, p2, p3 = st.columns(3)

fmt = "{:,.0f}" if is_korean else "{:,.2f}"

with p1:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #16a34a;">
        <div style="color: #16a34a; font-weight: bold; font-size: 0.95rem;">🟢 추천 매수 기준가</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{fmt.format(calc_buy_target)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">당시 직전 20일 저항선 상향 돌파 시 진입</div>
    </div>
    """, unsafe_allow_html=True)

with p2:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #2563eb;">
        <div style="color: #2563eb; font-weight: bold; font-size: 0.95rem;">🎯 1차 목표(익절) 가격</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #2563eb; margin: 6px 0;">{fmt.format(target_profit_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">매수가 돌파 후 단기 분할 익절 목표 구간</div>
    </div>
    """, unsafe_allow_html=True)

with p3:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #dc2626;">
        <div style="color: #dc2626; font-weight: bold; font-size: 0.95rem;">🔴 추천 손절/이탈 매도가</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{fmt.format(calc_stop_target)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">당시 직전 20일 최저 지지선 하향 이탈 시 전량 손절</div>
    </div>
    """, unsafe_allow_html=True)

# 당일 세부 지표 카드
if has_supply:
    k1, k2, k3, k4 = st.columns(4)
else:
    k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""<div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">종가 (전일 대비)</div>
        <div style="font-size: 1.35rem; font-weight: bold; margin-top: 4px;">{fmt.format(c_close)} {unit}</div>
        <div style="font-size: 0.85rem; color: {'#ef4444' if day_diff > 0 else '#3b82f6'};">({day_diff:+,.2f} / {day_pct:+.2f}%)</div>
    </div>""", unsafe_allow_html=True)

with k2:
    st.markdown(f"""<div class="metric-box">
        <div style="color: #64748b; font-size: 0.85rem;">시가 / 고가 / 저가</div>
        <div style="font-size: 0.95rem; font-weight: bold; margin-top: 6px;">시: {fmt.format(target_row['Open'])} | 고: {fmt.format(target_row['High'])}</div>
        <div style="font-size: 0.95rem; font-weight: bold; color: #3b82f6;">저: {fmt.format(target_row['Low'])} {unit}</div>
    </div>""", unsafe_allow_html=True)

if has_supply:
    with k3:
        f_qty = target_row.get('외국인순매수', 0)
        f_col = "#3b82f6" if f_qty >= 0 else "#ef4444"
        st.markdown(f"""<div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">외국인 순매매</div>
            <div style="font-size: 1.35rem; font-weight: bold; color: {f_col}; margin-top: 4px;">{f_qty:+,.0f} 주</div>
            <div style="font-size: 0.8rem; color: #64748b;">당일 집계 기준</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        i_qty = target_row.get('기관순매수', 0)
        i_col = "#f97316" if i_qty >= 0 else "#ef4444"
        st.markdown(f"""<div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">기관 순매매</div>
            <div style="font-size: 1.35rem; font-weight: bold; color: {i_col}; margin-top: 4px;">{i_qty:+,.0f} 주</div>
            <div style="font-size: 0.8rem; color: #64748b;">당일 집계 기준</div>
        </div>""", unsafe_allow_html=True)
else:
    with k3:
        v_qty = target_row['Volume']
        st.markdown(f"""<div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">당일 총 거래량</div>
            <div style="font-size: 1.35rem; font-weight: bold; margin-top: 4px;">{v_qty:,.0f} 주</div>
            <div style="font-size: 0.8rem; color: #64748b;">미국 거래소 집계</div>
        </div>""", unsafe_allow_html=True)

# 당일 분석 브리핑
st.markdown(f"### 📝 [{target_date_formatted}] AI 심층 분석 요약")
briefing_body = "<br>".join([f"• {r}" for r in date_reasons])
st.markdown(f"""
<div class="briefing-box">
    <b>[{target_date_formatted} 진단 결과]</b><br>
    {briefing_body}
</div>
""", unsafe_allow_html=True)

# 관련 시장 핵심 뉴스
st.markdown(f"#### 📰 [{selected_name}] 시장 핵심 뉴스")
news_list = fetch_stock_news(selected_name)
if news_list:
    for n in news_list:
        st.markdown(f"""
        <div class="news-card">
            <a class="news-title" href="{n['link']}" target="_blank">🔗 {n['title']}</a>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 3px;">발행일시: {n['date']}</div>
        </div>
        """, unsafe_allow_html=True)
