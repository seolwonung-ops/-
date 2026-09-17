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
        padding: 14px 18px; font-size: 0.95rem; line-height: 1.6; margin-bottom: 16px;
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

# 2. 피처 엔지니어링 (AI 학습용 지표 생성)
def prepare_features(df, has_supply):
    data = df.copy()
    data['SMA20'] = data['Close'].rolling(20).mean()
    data['SMA60'] = data['Close'].rolling(60).mean()
    
    # RSI (14)
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 머신러닝 입력 특성
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

# 3. AI 머신러닝 학습 및 확률 추론 엔진
@st.cache_resource(ttl=3600)
def train_and_predict_ai(df_clean, feature_cols):
    # 정답 라벨링: 향후 20영업일 뒤 주가가 +3% 이상 유의미하게 상승했는가? (1: 상승, 0: 횡보/하락)
    df_clean['Target_20d_Return'] = (df_clean['Close'].shift(-20) - df_clean['Close']) / df_clean['Close']
    df_clean['Label'] = (df_clean['Target_20d_Return'] >= 0.03).astype(int)

    # 훈련셋: 20일 뒤 결과를 알 수 있는 과거 데이터
    train_mask = df_clean['Target_20d_Return'].notna()
    X_train = df_clean.loc[train_mask, feature_cols].fillna(0)
    y_train = df_clean.loc[train_mask, 'Label']

    # AI 모델 구축 (랜덤 포레스트)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=3, random_state=42)
    model.fit(X_train, y_train)

    # 전체 기간에 대한 AI 상승 예측 확률 계산
    all_X = df_clean[feature_cols].fillna(0)
    proba_up = model.predict_proba(all_X)[:, 1] * 100
    
    # 중요하게 본 지표 가중치
    importances = dict(zip(feature_cols, model.feature_importances_))
    return proba_up, importances

# 4. 사이드바 종목 설정
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

target_row = df.loc[selected_date]
target_date_formatted = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%Y년 %m월 %d일")
loc_idx = df.index.get_loc(selected_date)

sub_df = df.iloc[:loc_idx+1]
window = min(len(sub_df), 20)
buy_price = float(sub_df['High'].iloc[-window:].max())
stop_price = float(sub_df['Low'].iloc[-window:].min())

c_close = float(target_row['Close'])
ai_prob = float(target_row['AI_Win_Prob'])

# 6. 차트 렌더링 (AI 예측 확률 서브플롯 추가)
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
fig.add_hline(y=buy_price, line_dash="dash", line_color="#16a34a", line_width=1.5, annotation_text=f"▲ 매수가: {buy_price:,.0f}", row=1, col=1)
fig.add_hline(y=stop_price, line_dash="dash", line_color="#dc2626", line_width=1.5, annotation_text=f"▼ 손절가: {stop_price:,.0f}", row=1, col=1)
fig.add_vline(x=selected_date, line_width=2, line_dash="dot", line_color="#8b5cf6", row=1, col=1)

# AI 확률 그래프
fig.add_trace(go.Scatter(x=df.index, y=df['AI_Win_Prob'], line=dict(color='#2563eb', width=1.8), name="AI 상승 확률"), row=2, col=1)
fig.add_hline(y=60, line_dash="dash", line_color="#16a34a", annotation_text="상승 우세 기준(60%)", row=2, col=1)
fig.add_hline(y=40, line_dash="dash", line_color="#dc2626", annotation_text="하락 경계 기준(40%)", row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.5), name="RSI"), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="#2563eb", row=3, col=1)

fig.update_xaxes(type='category', showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True, nticks=12)
fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True)
fig.update_layout(height=880, margin=dict(l=15, r=15, t=35, b=25), xaxis_rangeslider_visible=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified", legend=dict(orientation="h", y=1.03))

st.plotly_chart(fig, use_container_width=True)

# 7. AI 분석 판정
if ai_prob >= 65:
    ai_verdict = "🟢 AI 적극 매수 제안 (STRONG BUY)"
    ai_banner = "banner-strong-buy"
    ai_action = f"과거 5년치 패턴 학습 결과, 현재와 유사한 환경에서 20영업일 내 주가가 유의미하게 상승할 확률은 <b>{ai_prob:.1f}%</b>로 매우 높습니다. 분할 매수 진입을 적극 검토할 구간입니다."
elif ai_prob >= 50:
    ai_verdict = "🟢 AI 분할 매수 고려 (BUY)"
    ai_banner = "banner-buy"
    ai_action = f"상승 우세 확률(<b>{ai_prob:.1f}%</b>)을 보이고 있습니다. 지지선 이탈 여부를 확인하며 보수적 분할 매수가 유효합니다."
elif ai_prob <= 35:
    ai_verdict = "🔴 AI 비중 축소 / 손절 권고 (SELL)"
    ai_banner = "banner-sell"
    ai_action = f"상승 확률이 <b>{ai_prob:.1f}%</b>에 불과하여 과거 통계상 하방 압력이 훨씬 컸던 구간입니다. 현금 확보 및 리스크 관리가 시급합니다."
else:
    ai_verdict = "🟡 AI 매매 보류 / 관망 (HOLD)"
    ai_banner = "banner-hold"
    ai_action = f"상승 확률 <b>{ai_prob:.1f}%</b>로 상승과 하락 모멘텀이 팽팽합니다. 방향성이 확정될 때까지 진입을 유보하세요."

# 8. 결과 렌더링
st.markdown("---")
st.markdown(f"## 🤖 [{target_date_formatted}] 머신러닝 AI 진단 결과")

st.markdown(f"""
<div class="signal-banner {ai_banner}">
    <div style="font-size: 1.3rem;">{ai_verdict} (AI 산출 상승 확률: {ai_prob:.1f}%)</div>
    <div style="margin-top: 5px; font-weight: normal; line-height: 1.6;">{ai_action}</div>
</div>
""", unsafe_allow_html=True)

# 가격 가이드라인
fmt = "{:,.0f}" if is_korean else "{:,.2f}"
st.markdown(f"### 🎯 [{target_date_formatted}] AI 추천 매매 기준 가격")
c1, c2, c3 = st.columns(3)

target_profit = buy_price * 1.08

with c1:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #16a34a;">
        <div style="color: #16a34a; font-weight: bold; font-size: 0.95rem;">🟢 추천 매수 기준가</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #16a34a; margin: 6px 0;">{fmt.format(buy_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">20일 최고 저항선 상향 돌파 시</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #2563eb;">
        <div style="color: #2563eb; font-weight: bold; font-size: 0.95rem;">🎯 AI 1차 목표가 (익절선)</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #2563eb; margin: 6px 0;">{fmt.format(target_profit)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">돌파 성공 시 1차 단기 차익 실현선 (+8%)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="price-box" style="border-top: 4px solid #dc2626;">
        <div style="color: #dc2626; font-weight: bold; font-size: 0.95rem;">🔴 리스크 방어선 (손절가)</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{fmt.format(stop_price)} {unit}</div>
        <div style="font-size: 0.8rem; color: #64748b;">20일 최저 지지선 하향 이탈 시 전량 손절</div>
    </div>""", unsafe_allow_html=True)

# 모델이 중요하게 판단한 피처 가중치 표시
st.markdown("### 🧠 AI가 이번 판단에서 가장 중요하게 반영한 지표 TOP 3")
sorted_weights = sorted(feature_weights.items(), key=lambda x: x[1], reverse=True)[:3]
w1, w2, w3 = st.columns(3)
for col, (f_name, weight) in zip([w1, w2, w3], sorted_weights):
    with col:
        st.markdown(f"""
        <div class="metric-box">
            <div style="color: #64748b; font-size: 0.85rem;">중요 지표: {f_name}</div>
            <div style="font-size: 1.3rem; font-weight: bold; color: #1e40af; margin-top: 4px;">{weight*100:.1f}% 영향도</div>
        </div>""", unsafe_allow_html=True)
