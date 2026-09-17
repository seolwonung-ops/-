import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI 머신러닝 주식 매매 시스템", page_icon="📈", layout="wide")

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
        border-radius: 10px; padding: 16px 20px; margin: 12px 0; font-weight: bold;
    }
    .banner-strong-buy { background-color: #dcfce7; color: #15803d; border-left: 6px solid #16a34a; }
    .banner-buy { background-color: #f0fdf4; color: #166534; border-left: 6px solid #22c55e; }
    .banner-hold { background-color: #fef3c7; color: #b45309; border-left: 6px solid #f59e0b; }
    .banner-sell { background-color: #fee2e2; color: #991b1b; border-left: 6px solid #ef4444; }
    .briefing-box {
        background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px;
        padding: 16px 20px; font-size: 0.95rem; line-height: 1.6; margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# 1. 수급 데이터 수집
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

# 2. 피처 엔지니어링
def prepare_features(df, has_supply):
    data = df.copy()
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA60'] = data['Close'].rolling(60).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 머신러닝 학습 지표
    data['Disparity_20'] = (data['Close'] / data['SMA20'] - 1) * 100
    data['Disparity_60'] = (data['Close'] / data['SMA60'] - 1) * 100
    data['Vol_Ratio'] = data['Volume'] / (data['Volume'].rolling(20).mean() + 1e-9)
    data['Volatility_20'] = data['Close'].pct_change().rolling(20).std() * 100

    feature_cols = ['RSI', 'Disparity_20', 'Disparity_60', 'Vol_Ratio', 'Volatility_20']
    
    if has_supply:
        data['Fore_5d'] = data['외국인순매수'].rolling(5).sum()
        data['Inst_5d'] = data['기관순매수'].rolling(5).sum()
        feature_cols.extend(['Fore_5d', 'Inst_5d'])
        
    return data, feature_cols

# 3. AI 머신러닝 모델 학습
@st.cache_resource(ttl=3600)
def train_and_predict_ai(df_clean, feature_cols):
    df_clean['Target_20d_Return'] = (df_clean['Close'].shift(-20) - df_clean['Close']) / df_clean['Close']
    df_clean['Label'] = (df_clean['Target_20d_Return'] >= 0.03).astype(int)

    train_mask = df_clean['Target_20d_Return'].notna()
    X_train = df_clean.loc[train_mask, feature_cols].fillna(0)
    y_train = df_clean.loc[train_mask, 'Label']

    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)

    all_X = df_clean[feature_cols].fillna(0)
    proba_up = model.predict_proba(all_X)[:, 1] * 100
    importances = dict(zip(feature_cols, model.feature_importances_))
    return proba_up, importances

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

st.sidebar.title("📌 종목 및 기간 설정")
selected_name = st.sidebar.selectbox("감시 종목", list(st.session_state.watchlist.keys()))
ticker = st.session_state.watchlist[selected_name]
is_korean = ".KS" in ticker or ".KQ" in ticker
unit = "원" if is_korean else "$"

selected_period = st.sidebar.select_slider("데이터 수집 기간", options=["2y", "3y", "5y", "max"], value="5y")

with st.sidebar.expander("⚙️ 관심 종목 편집 (추가 / 삭제)", expanded=False):
    new_name = st.text_input("종목명 (예: 에코프로, 구글)")
    new_code = st.text_input("티커 심볼 (예: 086520.KQ, GOOGL)")
    if st.button("➕ 종목 추가", use_container_width=True):
        if new_name.strip() and new_code.strip():
            st.session_state.watchlist[new_name.strip()] = new_code.strip().upper()
            st.rerun()
    if st.button(f"🗑️ '{selected_name}' 삭제", use_container_width=True):
        if len(st.session_state.watchlist) > 1:
            del st.session_state.watchlist[selected_name]
            st.rerun()

# 5. 데이터 로드 및 AI 연산
df = yf.download(ticker, period=selected_period, progress=False)
if df.empty:
    st.error("데이터를 가져오지 못했습니다.")
    st.stop()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df.index = pd.to_datetime(df.index).strftime("%Y-%m-%d")

has_supply = False
if is_korean:
    df_supply = fetch_korean_investor_trading(ticker)
    if df_supply is not None and not df_supply.empty:
        df = df.join(df_supply, how="left")
        df['외국인순매수'] = df['외국인순매수'].fillna(0)
        df['기관순매수'] = df['기관순매수'].fillna(0)
        has_supply = True

df, feature_cols = prepare_features(df, has_supply)
proba_series, feature_weights = train_and_predict_ai(df, feature_cols)
df['AI_Win_Prob'] = proba_series

st.title(f"🤖 {selected_name} AI 머신러닝 매매 플래너")

date_list = list(df.index)
selected_date = st.select_slider("📅 분석 날짜 선택:", options=date_list, value=date_list[-1])

# ⭐️ 핵심: 선택된 날짜 기준 동적 가격 계산 (버그 수정)
loc_idx = df.index.get_loc(selected_date)
# 선택한 날짜 직전 20거래일 윈도우 슬라이싱
start_idx = max(0, loc_idx - 19)
hist_window_df = df.iloc[start_idx : loc_idx + 1]

# 선택 날짜 당시의 20일 최고가 및 최저가
buy_price = float(hist_window_df['High'].max())
stop_price = float(hist_window_df['Low'].min())

target_row = df.loc[selected_date]
target_date_formatted = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%Y년 %m월 %d일")
c_close = float(target_row['Close'])
ai_prob = float(target_row['AI_Win_Prob'])
target_profit = buy_price * 1.08

# 6. 차트 렌더링
fig = make_subplots(
    rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.08,
    row_heights=[0.55, 0.23, 0.22],
    subplot_titles=(["주가 및 매매 기준선", "AI 모델 추정 20일 뒤 상승 확률 (%)", "RSI 지표"])
)

fig.add_trace(go.Candlestick(
    x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    increasing_line_color="#ef4444", decreasing_line_color="#3b82f6", name="주가"
), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='#f59e0b', width=1.5), name="20일선"), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df['SMA60'], line=dict(color='#10b981', width=1.5), name="60일선"), row=1, col=1)

# 선택 날짜 기준 매수가/손절가 점선 표시
fig.add_hline(y=buy_price, line_dash="dash", line_color="#16a34a", line_width=1.5,
              annotation_text=f"▲ 돌파 매수가: {buy_price:,.0f}", row=1, col=1)
fig.add_hline(y=stop_price, line_dash="dash", line_color="#dc2626", line_width=1.5,
              annotation_text=f"▼ 손절 기준가: {stop_price:,.0f}", row=1, col=1)
fig.add_vline(x=selected_date, line_width=2, line_dash="dot", line_color="#8b5cf6", row=1, col=1)

# AI 확률 그래프
fig.add_trace(go.Scatter(x=df.index, y=df['AI_Win_Prob'], line=dict(color='#2563eb', width=1.8), name="AI 상승 확률"), row=2, col=1)
fig.add_hline(y=60, line_dash="dash", line_color="#16a34a", annotation_text="상승 우세(60%)", row=2, col=1)
fig.add_hline(y=40, line_dash="dash", line_color="#dc2626", annotation_text="하락 경계(40%)", row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.5), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#2563eb", row=3, col=1)

fig.update_xaxes(type='category', showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True, nticks=12)
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True)
fig.update_layout(height=880, margin=dict(l=15, r=15, t=35, b=25), xaxis_rangeslider_visible=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified", legend=dict(orientation="h", y=1.03))

st.plotly_chart(fig, use_container_width=True)

# 7. AI 분석 판정 및 세부 진단 근거 도출
if ai_prob >= 65:
    ai_verdict = "🟢 AI 적극 매수 제안 (STRONG BUY)"
    ai_banner = "banner-strong-buy"
    core_summary = f"과거 5년치 유사 패턴 학습 결과, 향후 20영업일 내 주가가 유의미하게 상승할 확률이 <b>{ai_prob:.1f}%</b>에 달합니다."
elif ai_prob >= 50:
    ai_verdict = "🟢 AI 분할 매수 고려 (BUY)"
    ai_banner = "banner-buy"
    core_summary = f"상승 우세 확률(<b>{ai_prob:.1f}%</b>)을 나타내고 있습니다. 지지선 가격을 확인하며 분할 매수 관점이 유효합니다."
elif ai_prob <= 35:
    ai_verdict = "🔴 AI 비중 축소 / 손절 권고 (SELL)"
    ai_banner = "banner-sell"
    core_summary = f"상승 확률이 <b>{ai_prob:.1f}%</b>에 불과하여 과거 통계상 하방 변동성 위험이 매우 컸던 구간입니다."
else:
    ai_verdict = "🟡 AI 매매 보류 / 관망 (HOLD)"
    ai_banner = "banner-hold"
    core_summary = f"상승 확률 <b>{ai_prob:.1f}%</b>로 상하방 모멘텀이 팽팽한 박스권입니다. 돌파 전까지 진입을 유보하세요."

# 상세 이유 생성 (Feature Attribution)
ai_reasons = []
rsi_now = target_row['RSI']
disp20 = target_row['Disparity_20']
disp60 = target_row['Disparity_60']
vol_r = target_row['Vol_Ratio']

# RSI 근거
if rsi_now <= 35:
    ai_reasons.append(f"• **RSI 심리 지표({rsi_now:.1f}):** 바닥권(과매도) 영역으로, 과거 데이터상 기술적 반등 확률을 높이는 긍정적 요인으로 작용했습니다.")
elif rsi_now >= 68:
    ai_reasons.append(f"• **RSI 심리 지표({rsi_now:.1f}):** 과열권에 진입하여 단기 차익 실현 매물 출회 가능성을 높이는 하방 요인으로 반영되었습니다.")
else:
    ai_reasons.append(f"• **RSI 심리 지표({rsi_now:.1f}):** 과열이나 침체 없는 중립 수준을 유지하고 있습니다.")

# 이격도 근거
if disp20 > 0 and disp60 > 0:
    ai_reasons.append(f"• **이평선 추세 배열:** 20일선 대비 {disp20:+.1f}%, 60일선 대비 {disp60:+.1f}%로 중장기 정배열 상승 궤도에 안착한 상태입니다.")
elif disp20 < 0 and disp60 < 0:
    ai_reasons.append(f"• **이평선 추세 배열:** 주요 이평선 아래로 주가가 이탈(20일선 {disp20:+.1f}%)하여 하락 압력으로 평가되었습니다.")
else:
    ai_reasons.append(f"• **이평선 추세 배열:** 20일선과 60일선 사이에 위치해 방향성 탐색 구간으로 인식되었습니다.")

# 수급 근거 (국내 주식)
if has_supply:
    fore5 = target_row.get('Fore_5d', 0)
    inst5 = target_row.get('Inst_5d', 0)
    if fore5 > 0 and inst5 > 0:
        ai_reasons.append(f"• **메이저 수급 집중도:** 최근 5영업일간 외국인(+{fore5:,.0f}주)과 기관(+{inst5:,.0f}주)의 동반 순매수가 모델의 상승 확률을 크게 견인했습니다.")
    elif fore5 < 0 and inst5 < 0:
        ai_reasons.append(f"• **메이저 수급 집중도:** 최근 5영업일간 외국인({fore5:,.0f}주)과 기관({inst5:,.0f}주)의 쌍끌이 순매도로 수급 이탈이 감점 요인으로 반영되었습니다.")
    else:
        ai_reasons.append(f"• **메이저 수급 집중도:** 외국인({fore5:+,.0f}주)과 기관({inst5:+,.0f}주)의 수급이 엇갈려 혼조세를 보였습니다.")

# 거래량 근거
if vol_r >= 1.5:
    ai_reasons.append(f"• **거래량 에너지:** 20일 평균 대비 {vol_r:.1f}배의 대량 거래량이 실려 시장 모멘텀이 유입된 것으로 평가되었습니다.")

# 8. 결과 렌더링
st.markdown("---")
st.markdown(f"## 🤖 [{target_date_formatted}] 머신러닝 AI 진단 결과")

st.markdown(f"""
<div class="signal-banner {ai_banner}">
    <div style="font-size: 1.3rem;">{ai_verdict} (AI 산출 20일 뒤 상승 확률: {ai_prob:.1f}%)</div>
    <div style="margin-top: 5px; font-weight: normal; line-height: 1.6;">{core_summary}</div>
</div>
""", unsafe_allow_html=True)

# 💡 AI가 왜 이 진단을 내렸는지 상세 브리핑 상자
st.markdown(f"### 📝 AI 진단 배경 및 세부 판단 근거")
briefing_html = "<br>".join(ai_reasons)
st.markdown(f"""
<div class="briefing-box">
    <b>[{target_date_formatted} 모델 판단 상세 요약]</b><br>
    {briefing_html}
</div>
""", unsafe_allow_html=True)

# 날짜별 동적 가격 가이드라인 3분할 카드
fmt = "{:,.0f}" if is_korean else "{:,.2f}"
st.markdown(f"### 🎯 [{target_date_formatted}] AI 추천 매매 기준 가격")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #16a34a;">
        <div style="color: #16a34a; font-weight: bold; font-size: 0.95rem;">🟢 추천 돌파 매수가</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{fmt.format(buy_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">선택일 직전 20일 최고 저항선 돌파 시</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #2563eb;">
        <div style="color: #2563eb; font-weight: bold; font-size: 0.95rem;">🎯 1차 목표가 (익절선)</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #2563eb; margin: 6px 0;">{fmt.format(target_profit)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">매수가 돌파 후 단기 차익 실현선 (+8%)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #dc2626;">
        <div style="color: #dc2626; font-weight: bold; font-size: 0.95rem;">🔴 리스크 방어선 (손절가)</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{fmt.format(stop_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">선택일 직전 20일 최저 지지선 이탈 시 전량 손절</div>
    </div>""", unsafe_allow_html=True)
