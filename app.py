import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from datetime import datetime

st.set_page_config(page_title="코스피 실시간 15대 퀀트 가치투자 시스템", page_icon="🏛️", layout="wide")

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
    .quant-card {
        background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .banner-strong-buy { background-color: #dcfce7; color: #15803d; border-left: 6px solid #16a34a; padding: 16px 20px; border-radius: 10px; font-weight: bold; margin: 12px 0; }
    .banner-hold { background-color: #fef3c7; color: #b45309; border-left: 6px solid #f59e0b; padding: 16px 20px; border-radius: 10px; font-weight: bold; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# 1. 네이버 금융에서 코스피 저평가/우량 후보군 실시간 자동 수집
@st.cache_data(ttl=600)
def fetch_realtime_kospi_candidates():
    headers = {"User-Agent": "Mozilla/5.0"}
    candidates = {}
    
    # 1) 코스피 저PER/배당/저PBR 상위 페이지 실시간 수집
    urls = [
        "https://finance.naver.com/sise/sise_low_per.naver?sosok=0", # 코스피 저PER
        "https://finance.naver.com/sise/dividend_list.naver?sosok=0", # 코스피 고배당
        "https://finance.naver.com/sise/sise_market_sum.naver?sosok=0&page=1" # 코스피 시총 상위 50
    ]
    
    for url in urls:
        try:
            res = requests.get(url, headers=headers, timeout=6)
            soup = BeautifulSoup(res.text, "html.parser")
            links = soup.find_all("a", href=True)
            for a in links:
                href = a['href']
                if "/item/main.naver?code=" in href:
                    code = href.split("code=")[1].strip()
                    name = a.text.strip()
                    if len(code) == 6 and code.isdigit() and name:
                        # 우선주, 스팩, ETF 제외
                        if not name.endswith("우") and not name.endswith("우B") and "스팩" not in name and "TIGER" not in name and "KODEX" not in name:
                            candidates[name] = f"{code}.KS"
                            if len(candidates) >= 30: # 빠른 분석을 위해 상위 30개 종목 추출
                                break
        except Exception:
            pass
            
    # 기본 안전망 (네트워크 차단 대비)
    if len(candidates) < 10:
        candidates.update({
            "현대차": "005380.KS", "기아": "000270.KS", "하이스틸": "071090.KS",
            "휴스틸": "005010.KS", "모토닉": "009680.KS", "대한제강": "084010.KS",
            "세아제강": "306200.KS", "한국앤컴퍼니": "000240.KS", "신세계": "004170.KS"
        })
    return candidates

# 2. 수급 데이터
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

# 3. 15가지 퀀트 가치투자 평가 엔진
@st.cache_data(ttl=3600)
def evaluate_15_quant_criteria(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info or {}
        
        bs = tk.balance_sheet
        inc = tk.financials

        mcap = info.get('marketCap', 0)
        per = info.get('trailingPE', info.get('forwardPE', None))
        pbr = info.get('priceToBook', None)
        
        ocf = info.get('operatingCashflow', None)
        pcr = (mcap / ocf) if (mcap and ocf and ocf > 0) else None
        ev_ebitda = info.get('enterpriseToEbitda', None)
        
        div_yield = (info.get('dividendYield') or 0.0) * 100
        bond_benchmark = 3.8 * 0.67 # 약 2.54%

        curr_assets = None
        curr_liab = None
        total_liab = None
        total_equity = None
        inventory = None
        receivables = None
        
        if bs is not None and not bs.empty:
            latest_bs = bs.iloc[:, 0]
            curr_assets = latest_bs.get('Current Assets', None)
            curr_liab = latest_bs.get('Current Liabilities', None)
            total_liab = latest_bs.get('Total Liabilities Net Minority Interest', latest_bs.get('Total Debt', None))
            total_equity = latest_bs.get('Stockholders Equity', latest_bs.get('Common Stock Equity', None))
            inventory = latest_bs.get('Inventory', None)
            receivables = latest_bs.get('Receivables', latest_bs.get('Accounts Receivable', None))

        rev = None
        rev_prev = None
        if inc is not None and not inc.empty:
            latest_inc = inc.iloc[:, 0]
            rev = latest_inc.get('Total Revenue', None)
            if inc.shape[1] > 1:
                rev_prev = inc.iloc[:, 1].get('Total Revenue', None)

        roe = (info.get('returnOnEquity') or 0.0) * 100
        op_margin = (info.get('operatingMargins') or 0.0) * 100
        
        debt_ratio = None
        if total_liab and total_equity and total_equity > 0:
            debt_ratio = (total_liab / total_equity) * 100
        else:
            debt_ratio = info.get('debtToEquity', None)

        curr_ratio = None
        if curr_assets and curr_liab and curr_liab > 0:
            curr_ratio = (curr_assets / curr_liab) * 100
        elif info.get('currentRatio'):
            curr_ratio = info.get('currentRatio') * 100

        ncav = (curr_assets - total_liab) if (curr_assets and total_liab) else None
        p_ncav = (mcap / ncav) if (mcap and ncav and ncav > 0) else None

        inv_turnover = (rev / inventory) if (rev and inventory and inventory > 0) else None
        rec_turnover = (rev / receivables) if (rev and receivables and receivables > 0) else None

        eps_growth = (info.get('earningsGrowth') or 0.0) * 100
        rev_growth = ((rev - rev_prev) / rev_prev * 100) if (rev and rev_prev and rev_prev > 0) else ((info.get('revenueGrowth') or 0.0) * 100)

        checklist = [
            ("1. PER (주가수익비율)", "0 < PER < 7.2", f"{per:.2f}배" if per else "N/A", (per is not None and 0 < per < 7.2)),
            ("2. PBR (주가순자산비율)", "0 < PBR < 0.65", f"{pbr:.2f}배" if pbr else "N/A", (pbr is not None and 0 < pbr < 0.65)),
            ("3. PCR (주가현금흐름비율)", "0 < PCR < 10.0", f"{pcr:.2f}배" if pcr else "N/A", (pcr is not None and 0 < pcr < 10.0)),
            ("4. EV/EBITDA", "EV/EBITDA < 4.0", f"{ev_ebitda:.2f}배" if ev_ebitda else "N/A", (ev_ebitda is not None and 0 < ev_ebitda < 4.0)),
            ("5. 배당수익률", f"> {bond_benchmark:.2f}%", f"{div_yield:.2f}%", (div_yield > bond_benchmark)),
            ("6. 부채비율", "0 < 부채비율 ≤ 150%", f"{debt_ratio:.1f}%" if debt_ratio else "N/A", (debt_ratio is not None and 0 < debt_ratio <= 150)),
            ("7. 유동비율", "유동비율 ≥ 200%", f"{curr_ratio:.1f}%" if curr_ratio else "N/A", (curr_ratio is not None and curr_ratio >= 200)),
            ("8. P/NCAV", "0 < P/NCAV < 1.0", f"{p_ncav:.2f}" if p_ncav else "N/A", (p_ncav is not None and 0 < p_ncav < 1.0)),
            ("9. 순유동자산 규모", "순유동자산 > 시총의 50%", f"{(ncav/mcap*100):.1f}%" if (ncav and mcap) else "N/A", (ncav is not None and mcap > 0 and ncav > mcap * 0.5)),
            ("10. ROE", "ROE > 10.0%", f"{roe:.1f}%", (roe > 10.0)),
            ("11. 영업이익률", "영업이익률 > 10.0%", f"{op_margin:.1f}%", (op_margin > 10.0)),
            ("12. 재고자산 회전율", "회전율 > 12회", f"{inv_turnover:.1f}회" if inv_turnover else "N/A", (inv_turnover is not None and inv_turnover > 12.0)),
            ("13. 매출채권 회전율", "회전율 > 6회", f"{rec_turnover:.1f}회" if rec_turnover else "N/A", (rec_turnover is not None and rec_turnover > 6.0)),
            ("14. EPS 성장률", "EPS 성장률 > 10.0%", f"{eps_growth:.1f}%", (eps_growth > 10.0)),
            ("15. 매출 성장률", "매출 성장률 > 10.0%", f"{rev_growth:.1f}%", (rev_growth > 10.0))
        ]

        total_score = sum([1 for c in checklist if c[3]])
        
        return {
            "total_score": total_score,
            "checklist": checklist,
            "is_recommended": (total_score >= 13),
            "per": per,
            "pbr": pbr,
            "roe": roe,
            "div": div_yield
        }
    except Exception:
        return None

# 4. 차트 지표 계산
def calculate_indicators(df):
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA60'] = df['Close'].rolling(60).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# 사이드바
if "watchlist" not in st.session_state:
    st.session_state.watchlist = {
        "🇰🇷 현대차": "005380.KS", "🇰🇷 기아": "000270.KS",
        "🇰🇷 삼성전자": "005930.KS", "🇰🇷 하이스틸": "071090.KS"
    }

tab1, tab2 = st.tabs(["🏛️ 코스피 실시간 15대 퀀트 발굴기", "📊 개별 종목 정밀 매매 차트"])

# ========================================================
# 탭 1: 코스피 실시간 자동 탐색 엔진
# ========================================================
with tab1:
    st.header("🏛️ 실시간 코스피(KOSPI) 15대 퀀트 가치투자 자동 추천")
    st.markdown("""
    사용자가 종목을 직접 입력하지 않아도, **네이버 금융의 실시간 코스피 시장 데이터**에서 
    저평가·우량주 후보군을 동적으로 긁어와 **15가지 퀀트 가치투자 지표(만점 15점)**를 실시간 자동 판정합니다.
    """)

    col_btn, col_info = st.columns([1, 2])
    with col_btn:
        start_scan = st.button("🚀 실시간 코스피 전수 스캔 및 발굴 시작", type="primary", use_container_width=True)
    with col_info:
        st.caption("※ 네이버 금융의 실시간 코스피 저PER / 고배당 / 시총 상위 랭킹을 크롤링하여 자동 스크리닝합니다.")

    if start_scan:
        with st.spinner("네이버 금융에서 실시간 코스피 후보군 목록을 가져오는 중..."):
            kospi_pool = fetch_realtime_kospi_candidates()

        st.info(f"💡 실시간 코스피 후보군 **{len(kospi_pool)}개 종목**을 성공적으로 수집했습니다. 15대 퀀트 분석을 시작합니다.")

        results = []
        p_bar = st.progress(0)
        status_lbl = st.empty()
        
        t_len = len(kospi_pool)
        for idx, (name, tk) in enumerate(kospi_pool.items()):
            status_lbl.text(f"[{idx+1}/{t_len}] {name} ({tk}) 재무제표 15대 지표 채점 중...")
            res = evaluate_15_quant_criteria(tk)
            if res:
                results.append({
                    "name": name,
                    "ticker": tk,
                    "score": res['total_score'],
                    "rec": res['is_recommended'],
                    "per": res['per'],
                    "pbr": res['pbr'],
                    "roe": res['roe'],
                    "div": res['div'],
                    "checklist": res['checklist']
                })
            p_bar.progress((idx + 1) / t_len)

        status_lbl.text("✅ 코스피 실시간 가치투자 분석 완료!")
        p_bar.empty()

        if results:
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # 13점 이상 추천 종목과 상위 종목 분리
            recommended = [r for r in results if r['rec']]
            
            st.markdown("---")
            if recommended:
                st.markdown(f"## 🎯 13점 이상 최초 매수 추천 종목 ({len(recommended)}건 발견)")
                for item in recommended:
                    st.markdown(f"""
                    <div class="banner-strong-buy">
                        <div style="font-size: 1.35rem;">🟢 [{item['name']} ({item['ticker']})] 총점: {item['score']}점 / 15점 만점</div>
                        <div style="margin-top: 6px; font-weight: normal; line-height: 1.6;">
                            • PER: <b>{item['per']:.1f}배</b> | PBR: <b>{item['pbr']:.2f}배</b> | ROE: <b>{item['roe']:.1f}%</b> | 배당: <b>{item['div']:.1f}%</b><br>
                            • <b>판정:</b> 15개 지표 중 13개 이상을 충족한 극저평가 알짜 우량주입니다. <b>18개월 목표 보유 전략</b>으로 포트폴리오 편입을 추천합니다.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ 현재 코스피 실시간 후보군 중 13점 이상을 완벽히 충족한 종목이 없습니다. (원칙에 따라 현금 보유 권고)")

            # 전체 실시간 코스피 랭킹표
            st.markdown("### 📊 실시간 코스피 가치투자 종합 랭킹 (상위 10개)")
            for rank_i, item in enumerate(results[:10], 1):
                border_c = "#16a34a" if item['rec'] else ("#2563eb" if item['score'] >= 10 else "#64748b")
                verdict_t = "🟢 매수 추천 (13점 이상)" if item['rec'] else f"🟡 관망/보류 ({item['score']}점)"
                
                with st.expander(f"#{rank_i} {item['name']} ({item['ticker']}) — 총점: {item['score']}/15점 [{verdict_t}]"):
                    st.markdown(f"""
                    <b>주요 지표 요약:</b> PER <b>{f"{item['per']:.1f}배" if item['per'] else 'N/A'}</b> | 
                    PBR <b>{f"{item['pbr']:.2f}배" if item['pbr'] else 'N/A'}</b> | 
                    ROE <b>{f"{item['roe']:.1f}%" if item['roe'] else 'N/A'}</b> | 
                    배당수익률 <b>{f"{item['div']:.1f}%" if item['div'] else 'N/A'}</b>
                    """, unsafe_allow_html=True)
                    
                    df_chk = pd.DataFrame(item['checklist'], columns=["지표명", "기준 조건", "측정값", "충족 여부"])
                    df_chk["부여 점수"] = df_chk["충족 여부"].apply(lambda x: "1점" if x else "0점")
                    df_chk["충족 여부"] = df_chk["충족 여부"].apply(lambda x: "✅ O" if x else "❌ X")
                    st.dataframe(df_chk, use_container_width=True, hide_index=True)

# ========================================================
# 탭 2: 개별 종목 주가 흐름 & 매매 기준선
# ========================================================
with tab2:
    st.sidebar.title("📌 개별 종목 차트 설정")
    stock_options = list(st.session_state.watchlist.keys())
    selected_name = st.sidebar.selectbox("감시 종목 선택", stock_options)
    ticker = st.session_state.watchlist[selected_name]
    is_korean = ".KS" in ticker or ".KQ" in ticker
    unit = "원" if is_korean else "$"

    selected_period = st.sidebar.select_slider("차트 기간", options=["1y", "2y", "3y", "5y", "max"], value="5y")

    with st.sidebar.expander("⚙️ 감시 종목 편집 (추가 / 삭제)", expanded=False):
        n_name = st.text_input("종목명 (예: 하이스틸)")
        n_code = st.text_input("티커 심볼 (예: 071090.KS)")
        if st.button("➕ 종목 추가", use_container_width=True):
            if n_name.strip() and n_code.strip():
                st.session_state.watchlist[n_name.strip()] = n_code.strip().upper()
                st.rerun()
        if st.button(f"🗑️ '{selected_name}' 삭제", use_container_width=True):
            if len(st.session_state.watchlist) > 1:
                del st.session_state.watchlist[selected_name]
                st.rerun()

    df = yf.download(ticker, period=selected_period, progress=False)
    if not df.empty:
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

        st.title(f"📈 {selected_name} 주가 차트 및 매매 기준선")
        date_list = list(df.index)
        selected_date = st.select_slider("📅 분석 날짜 선택:", options=date_list, value=date_list[-1])

        loc_idx = df.index.get_loc(selected_date)
        start_idx = max(0, loc_idx - 19)
        hist_window_df = df.iloc[start_idx : loc_idx + 1]

        buy_price = float(hist_window_df['High'].max())
        stop_price = float(hist_window_df['Low'].min())
        target_profit = buy_price * 1.08

        target_row = df.loc[selected_date]
        target_date_formatted = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%Y년 %m월 %d일")
        c_close = float(target_row['Close'])

        if loc_idx > 0:
            prev_close = float(df.iloc[loc_idx - 1]['Close'])
            day_diff = c_close - prev_close
            day_pct = (day_diff / prev_close) * 100
        else:
            day_diff, day_pct = 0.0, 0.0

        diff_str = f"{day_diff:+,.0f}" if is_korean else f"{day_diff:+,.2f}"
        fmt = "{:,.0f}" if is_korean else "{:,.2f}"

        # 차트
        rows_cnt = 3 if has_supply else 2
        row_heights = [0.55, 0.20, 0.25] if has_supply else [0.70, 0.30]

        fig = make_subplots(
            rows=rows_cnt, cols=1, shared_xaxes=False, vertical_spacing=0.10,
            row_heights=row_heights,
            subplot_titles=(["주가 및 매매 기준선", "RSI 지표", "외국인 / 기관 순매수 (주)"] if has_supply else ["주가 차트", "RSI 지표"])
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
                      annotation_text=f"▼ 손절가: {stop_price:,.0f}", row=1, col=1)
        fig.add_vline(x=selected_date, line_width=2, line_dash="dot", line_color="#8b5cf6", row=1, col=1)

        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='#8b5cf6', width=1.5), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#dc2626", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#2563eb", row=2, col=1)

        if has_supply:
            fig.add_trace(go.Bar(x=df.index, y=df['외국인순매수'], name="외국인", marker_color="#3b82f6"), row=3, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['기관순매수'], name="기관", marker_color="#f97316"), row=3, col=1)

        fig.update_xaxes(type='category', showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True, nticks=12)
        fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", showline=True, linewidth=1.5, linecolor="#475569", mirror=True)
        fig.update_layout(height=850 if has_supply else 680, margin=dict(l=15, r=15, t=35, b=25), xaxis_rangeslider_visible=False, plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", hovermode="x unified", legend=dict(orientation="h", y=1.03))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### 🎯 [{target_date_formatted}] 당일 종가 및 추천 가격")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="price-box" style="border-top: 4px solid #3b82f6;">
                <div style="color: #64748b; font-size: 0.85rem;">당일 종가 (Close)</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #1e3a8a; margin: 4px 0;">{fmt.format(c_close)} {unit}</div>
                <div style="font-size: 0.8rem; color: {'#ef4444' if day_diff > 0 else '#3b82f6'};">전일비: {diff_str} ({day_pct:+.2f}%)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="price-box" style="border-top: 4px solid #16a34a;">
                <div style="color: #16a34a; font-weight: bold; font-size: 0.95rem;">🟢 추천 돌파 매수가</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #16a34a; margin: 4px 0;">{fmt.format(buy_price)} {unit}</div>
                <div style="font-size: 0.8rem; color: #64748b;">20일 최고 저항선 돌파 시</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="price-box" style="border-top: 4px solid #dc2626;">
                <div style="color: #dc2626; font-weight: bold; font-size: 0.95rem;">🔴 리스크 방어선 (손절가)</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #dc2626; margin: 6px 0;">{fmt.format(stop_price)} {unit}</div>
                <div style="font-size: 0.8rem; color: #64748b;">20일 최저 지지선 이탈 시 손절</div>
            </div>""", unsafe_allow_html=True)
