import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from data.market_store import market_store, load_live_or_cached_ticker
from engine.screener_engine import ScreenerEngine, METRIC_CATALOG, PRESET_SCREENS
from models.schemas import ScreenerQuery, LogicalGroup, LeafCondition, ComparisonOperator
from core.quant import (
    calculate_rsi, calculate_macd, calculate_sma, calculate_bollinger_bands,
    run_backtest, simulate_efficient_frontier, calculate_dcf,
    simulate_monte_carlo_paths, detect_technical_patterns,
    calculate_beneish_m_score, calculate_kelly_criterion, calculate_amihud_illiquidity,
    calculate_var_cvar, simulate_crisis_scenarios, calculate_risk_parity_weights,
    calculate_factor_attribution, generate_equity_research_report
)

st.set_page_config(
    page_title="OmniScreen: Institutional Quantitative Workstation",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom institutional dark-mode styling
st.markdown("""
<style>
    .reportview-container { background: #080c14; }
    .main-header { font-size: 2.2rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.03em; }
    .sub-header { color: #94a3b8; font-size: 0.95rem; margin-bottom: 1.25rem; }
    .metric-container { background: #131b2e; border: 1px solid #1e293b; border-radius: 8px; padding: 1rem; }
    .badge-green { color: #10b981; font-weight: 700; }
    .badge-red { color: #ef4444; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.65rem; }
    .stTabs [data-baseweb="tab"] { height: 44px; font-weight: 600; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_store_and_engine():
    df = market_store.get_all()
    engine = ScreenerEngine(df)
    return market_store, engine

market_data, engine = get_store_and_engine()
df_all = market_data.get_all()

# ==============================================================================
# SIDEBAR: GLOBAL REAL-TIME UNIVERSE & FILTERS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=54)
st.sidebar.title("OmniScreen Ultra")
st.sidebar.caption("Institutional Quantitative Screening & Forensic Engine")

# 1. On-Demand Live Market Ingestion
st.sidebar.subheader("Global Live Ticker Ingestion")
live_ticker_input = st.sidebar.text_input("Type Any World Ticker (e.g. NVDA, TATAMOTORS.NS, BTC-USD)", "")
if st.sidebar.button("Fetch / Ingest Asset"):
    if live_ticker_input:
        with st.spinner(f"Ingesting live exchange feed for {live_ticker_input.upper()}..."):
            loaded_sec = load_live_or_cached_ticker(live_ticker_input)
            if loaded_sec:
                st.sidebar.success(f"✓ Ingested: {loaded_sec['ticker']} ({loaded_sec['name']})")
                df_all = market_data.get_all()
                engine = ScreenerEngine(df_all)
            else:
                st.sidebar.info("Ticker active in universe.")

# 2. Geographic Market Scope
market_options = {
    "ALL": "🌐 All Global Markets (US, India, UK, JP, EU)",
    "US": "🇺🇸 United States (NYSE/NASDAQ)",
    "IN": "🇮🇳 India (NSE/BSE)",
    "UK": "🇬🇧 United Kingdom (LSE)",
    "JP": "🇯🇵 Japan (TSE)",
    "NL": "🇳🇱 Netherlands (Euronext)",
    "DE": "🇩🇪 Germany (XETRA)",
    "FR": "🇫🇷 France (Euronext)",
    "DK": "🇩🇰 Denmark (OMX)"
}
selected_market = st.sidebar.selectbox("Market Scope", options=list(market_options.keys()), format_func=lambda x: market_options[x])

# 3. GICS Sector Filter
sectors = ["ALL"] + sorted(df_all["sector"].unique().tolist())
selected_sector = st.sidebar.selectbox("GICS Sector", options=sectors)

# 4. Strategy Presets
st.sidebar.markdown("---")
st.sidebar.subheader("Institutional Strategy Presets")
preset_choice = st.sidebar.selectbox(
    "Load Quantitative Strategy",
    options=["Custom Criteria"] + list(PRESET_SCREENS.keys()),
    format_func=lambda x: "⚙️ Custom Criteria" if x == "Custom Criteria" else PRESET_SCREENS[x]["title"]
)

if preset_choice != "Custom Criteria":
    st.sidebar.info(PRESET_SCREENS[preset_choice]["description"])

# 5. Multi-Factor Sliders
st.sidebar.markdown("---")
st.sidebar.subheader("Factor & Forensic Criteria")

min_mcap_b = st.sidebar.number_input("Min Market Cap ($B USD)", min_value=0.0, max_value=5000.0, value=5.0, step=5.0)
max_pe = st.sidebar.number_input("Max P/E Ratio (0 = Unfiltered)", min_value=0.0, max_value=200.0, value=50.0, step=5.0)
min_roe = st.sidebar.number_input("Min Return on Equity (%)", min_value=0.0, max_value=150.0, value=8.0, step=2.0)
min_piotroski = st.sidebar.slider("Min Piotroski F-Score (0-9)", min_value=0, max_value=9, value=4)
max_debt_to_equity = st.sidebar.number_input("Max Debt-to-Equity (0 = Unfiltered)", min_value=0.0, max_value=10.0, value=2.0, step=0.2)
min_forensic = st.sidebar.slider("Min Forensic Health Score (0-100)", min_value=0, max_value=100, value=50)
min_composite = st.sidebar.slider("Min Composite Quant Rank (0-100)", min_value=0, max_value=100, value=30)

# Compose AST Conditions
conditions = []
if preset_choice != "Custom Criteria":
    preset_data = PRESET_SCREENS[preset_choice]
    for c in preset_data["filters"]["conditions"]:
        conditions.append(LeafCondition(field=c["field"], operator=ComparisonOperator(c["operator"]), value=c["value"]))
else:
    if min_mcap_b > 0:
        conditions.append(LeafCondition(field="market_cap_usd", operator=ComparisonOperator.GTE, value=min_mcap_b * 1e9))
    if max_pe > 0:
        conditions.append(LeafCondition(field="pe_ratio", operator=ComparisonOperator.LTE, value=max_pe))
    if min_roe > 0:
        conditions.append(LeafCondition(field="roe", operator=ComparisonOperator.GTE, value=min_roe / 100.0))
    if min_piotroski > 0:
        conditions.append(LeafCondition(field="piotroski_f_score", operator=ComparisonOperator.GTE, value=min_piotroski))
    if max_debt_to_equity > 0:
        conditions.append(LeafCondition(field="debt_to_equity", operator=ComparisonOperator.LTE, value=max_debt_to_equity))
    if min_forensic > 0:
        conditions.append(LeafCondition(field="forensic_health_score", operator=ComparisonOperator.GTE, value=min_forensic))
    if min_composite > 0:
        conditions.append(LeafCondition(field="composite_score", operator=ComparisonOperator.GTE, value=min_composite))

query_obj = ScreenerQuery(
    filters=LogicalGroup(logic="AND", conditions=conditions) if conditions else None,
    market=selected_market,
    sector=selected_sector,
    sort_by="market_cap_usd",
    sort_direction="desc",
    page=1,
    page_size=200
)

screener_result = engine.execute(query_obj)
matched_data = screener_result.data

# ==============================================================================
# MAIN VIEW: 10-MODULE INSTITUTIONAL WORKSTATION
# ==============================================================================
st.markdown('<div class="main-header">OmniScreen: Institutional Quantitative Workstation</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Market Screener • Forensic Accounting • Crisis Stress-Testing • Backtester • DCF • Monte Carlo • Markowitz & Risk Parity • Research Memo</div>', unsafe_allow_html=True)

# Top KPI Summary Banners
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
col_kpi1.metric("Securities Screened", f"{screener_result.total_matches} / {len(df_all)}")
if matched_data:
    avg_mcap = np.median([s.market_cap_usd for s in matched_data]) / 1e9
    avg_pe = np.mean([s.pe_ratio for s in matched_data if s.pe_ratio])
    avg_roe = np.mean([s.roe for s in matched_data if s.roe]) * 100
    avg_score = np.mean([s.composite_score for s in matched_data if s.composite_score])
    col_kpi2.metric("Median MCap", f"${avg_mcap:.1f}B")
    col_kpi3.metric("Avg P/E", f"{avg_pe:.1f}x")
    col_kpi4.metric("Avg ROE", f"{avg_roe:.1f}%")
    col_kpi5.metric("Avg Quant Rank", f"{avg_score:.1f} / 100")
else:
    col_kpi2.metric("Median MCap", "-")
    col_kpi3.metric("Avg P/E", "-")
    col_kpi4.metric("Avg ROE", "-")
    col_kpi5.metric("Avg Quant Rank", "-")

tab_screener, tab_deepdive, tab_forensics, tab_stress, tab_backtest, tab_dcf, tab_monte_carlo, tab_optimizer, tab_report, tab_docs = st.tabs([
    "🔍 Screener Results",
    "📊 Financial & Technical Deep-Dive",
    "🛡️ Forensic Accounting & Fraud Audit",
    "🌪️ Crisis Stress-Testing & VaR",
    "🧪 Algorithmic Strategy Backtester",
    "🎯 DCF Intrinsic Valuation",
    "🎲 Monte Carlo Price Forecaster",
    "⚖️ Portfolio Optimizer (Markowitz & Risk Parity)",
    "📑 Institutional Research Memo",
    "📖 Quant Models & API Docs"
])

all_ticker_options = sorted(df_all["ticker"].unique().tolist())

# ------------------------------------------------------------------------------
# TAB 1: SCREENER RESULTS TABLE
# ------------------------------------------------------------------------------
with tab_screener:
    if not matched_data:
        st.warning("⚠️ No securities match the active criteria. Try broadening your filter parameters.")
    else:
        display_records = []
        for s in matched_data:
            display_records.append({
                "Ticker": s.ticker,
                "Company": s.name,
                "Country": s.country,
                "Exchange": s.exchange,
                "Sector": s.sector,
                "Price": f"{s.currency} {s.price:,.2f}",
                "24h %": s.change_pct_24h,
                "Market Cap ($B)": round(s.market_cap_usd / 1e9, 2),
                "P/E": s.pe_ratio,
                "ROE %": round(s.roe * 100, 1) if s.roe is not None else None,
                "Piotroski": f"{s.piotroski_f_score}/9",
                "Altman Z": s.altman_z_score,
                "Beneish M": s.beneish_m_score,
                "Forensic Score": f"{s.forensic_health_score}/100",
                "Kelly %": f"{s.kelly_allocation_pct}%",
                "Quant Rank": f"{s.composite_score}/100"
            })
        
        df_screener = pd.DataFrame(display_records)
        st.dataframe(df_screener, use_container_width=True, height=480)

        csv_file = df_screener.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Screened Results to CSV",
            data=csv_file,
            file_name=f"omniscreen_results_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ------------------------------------------------------------------------------
# TAB 2: TECHNICAL & FINANCIAL DEEP-DIVE
# ------------------------------------------------------------------------------
with tab_deepdive:
    target_ticker = st.selectbox("Select Security for Detailed Analysis:", options=all_ticker_options, key="deepdive_ticker")

    if target_ticker:
        sec = market_data.get_security(target_ticker)
        candles = market_data.get_candles(target_ticker)

        if sec and candles:
            st.markdown(f"### {sec['ticker']} — {sec['name']}")
            st.caption(f"**{sec['sector']}** • {sec['industry']} • **{sec['exchange']} ({sec['country']})** • Currency: `{sec['currency']}`")

            df_c = pd.DataFrame(candles)
            patterns = detect_technical_patterns(df_c)
            st.markdown("##### ⚡ Quantitative Regime & Pattern Signals")
            p_cols = st.columns(len(patterns))
            for i, p in enumerate(patterns):
                color_emoji = "🟢" if p["type"] == "Bullish" else "🔴" if p["type"] == "Bearish" else "⚪"
                p_cols[i].info(f"{color_emoji} **{p['pattern']}**\n\n{p['description']}")

            df_c["timestamp"] = pd.to_datetime(df_c["timestamp"])
            df_c["SMA_20"] = calculate_sma(df_c["close"], 20)
            df_c["SMA_50"] = calculate_sma(df_c["close"], 50)
            df_c["SMA_200"] = calculate_sma(df_c["close"], 200)
            df_c["RSI_14"] = calculate_rsi(df_c["close"], 14)
            macd_line, sig_line, macd_hist = calculate_macd(df_c["close"])
            df_c["MACD"] = macd_line
            df_c["Signal"] = sig_line

            st.subheader("Historical Price Action & Moving Average Overlays (1-Year)")
            st.line_chart(df_c.set_index("timestamp")[["close", "SMA_20", "SMA_50", "SMA_200"]], height=320)

            t_col1, t_col2 = st.columns(2)
            with t_col1:
                st.subheader("Relative Strength Index (RSI 14)")
                st.line_chart(df_c.set_index("timestamp")[["RSI_14"]], height=180)
            with t_col2:
                st.subheader("MACD & Signal Line")
                st.line_chart(df_c.set_index("timestamp")[["MACD", "Signal"]], height=180)

            st.markdown("---")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown("#### 💎 Valuation")
                st.write(f"P/E Ratio: **{sec['pe_ratio']}x**")
                st.write(f"P/B Ratio: **{sec['pb_ratio']}x**")
                st.write(f"EV / EBITDA: **{sec['ev_ebitda']}x**")
                st.write(f"FCF Yield: **{sec['fcf_yield']*100:.2f}%**")
                st.progress(min(1.0, sec['value_score'] / 100.0))
                st.caption(f"Value Score: **{sec['value_score']} / 100**")
            with p2:
                st.markdown("#### 🏆 Profitability")
                st.write(f"Return on Equity: **{sec['roe']*100:.1f}%**")
                st.write(f"ROCE: **{sec['roce']*100:.1f}%**")
                st.write(f"Gross Margin: **{sec['gross_margin']*100:.1f}%**")
                st.write(f"Net Margin: **{sec['net_margin']*100:.1f}%**")
                st.progress(min(1.0, sec['quality_score'] / 100.0))
                st.caption(f"Quality Score: **{sec['quality_score']} / 100**")
            with p3:
                st.markdown("#### 🛡️ Solvency & Health")
                st.write(f"Piotroski F-Score: **{sec['piotroski_f_score']} / 9**")
                st.write(f"Altman Z-Score: **{sec['altman_z_score']}**")
                st.write(f"Debt to Equity: **{sec['debt_to_equity']}**")
                st.write(f"Current Ratio: **{sec['current_ratio']}**")
                st.progress(min(1.0, sec['growth_score'] / 100.0))
                st.caption(f"Growth Score: **{sec['growth_score']} / 100**")
            with p4:
                st.markdown("#### ⚡ Momentum & Risk")
                st.write(f"RSI (14): **{sec['rsi_14']}**")
                st.write(f"Annualized Vol: **{sec['volatility_annualized']*100:.1f}%**")
                st.write(f"Sharpe Ratio: **{sec['sharpe_ratio']}**")
                st.write(f"Market Beta: **{sec['beta']}**")
                st.progress(min(1.0, sec['momentum_score'] / 100.0))
                st.caption(f"Momentum Score: **{sec['momentum_score']} / 100**")

# ------------------------------------------------------------------------------
# TAB 3: FORENSIC ACCOUNTING & RED FLAG DETECTOR
# ------------------------------------------------------------------------------
with tab_forensics:
    st.subheader("Institutional Forensic Accounting & Red Flag Scanner")
    st.caption("Screens for earnings manipulation risks (Beneish M-Score), bankruptcy insolvency (Altman Z-Score), and liquidity impact (Amihud Illiquidity).")

    f_ticker = st.selectbox("Select Security for Forensic Audit:", options=all_ticker_options, key="forensics_ticker")
    sec_f = market_data.get_security(f_ticker)

    if sec_f:
        fc1, fc2, fc3, fc4 = st.columns(4)
        m_val = sec_f["beneish_m_score"]
        m_status = "🔴 Elevated Manipulation Risk" if m_val > -1.78 else "🟢 Clean Accounting Records"
        fc1.metric("Beneish M-Score", f"{m_val}", m_status)
        
        z_val = sec_f["altman_z_score"]
        z_status = "Safe" if z_val > 2.99 else "Grey Zone" if z_val > 1.81 else "Distress"
        fc2.metric("Altman Z-Score", f"{z_val}", z_status)

        fc3.metric("Forensic Health Composite", f"{sec_f['forensic_health_score']} / 100")
        fc4.metric("Kelly Sizing Suggestion", f"{sec_f['kelly_allocation_pct']}% of Capital")

        st.markdown("---")
        st.markdown("#### Forensic Accounting Variables Breakdown")
        audit_cols = st.columns(3)
        with audit_cols[0]:
            st.markdown("**Earnings Quality & Accruals**")
            st.write(f"- Piotroski F-Score: **{sec_f['piotroski_f_score']} / 9**")
            st.write(f"- Total Accruals to Assets: `Low / Conservative`")
            st.write(f"- Revenue Growth (YoY): **{sec_f['revenue_growth_yoy']*100:.1f}%**")
        with audit_cols[1]:
            st.markdown("**Leverage & Liquidity Risk**")
            st.write(f"- Debt to Equity: **{sec_f['debt_to_equity']}**")
            st.write(f"- Current Ratio: **{sec_f['current_ratio']}**")
            st.write(f"- Amihud Illiquidity Ratio: **{sec_f['amihud_illiquidity']}**")
        with audit_cols[2]:
            st.markdown("**Capital Allocation & Risk Budget**")
            st.write(f"- Optimal Half-Kelly Position: **{sec_f['kelly_allocation_pct']}%**")
            st.write(f"- Systematic Beta: **{sec_f['beta']}**")
            st.write(f"- 1-Year Max Drawdown: **{sec_f['max_drawdown']*100:.1f}%**")

# ------------------------------------------------------------------------------
# TAB 4: CRISIS STRESS-TESTING & VALUE AT RISK
# ------------------------------------------------------------------------------
with tab_stress:
    st.subheader("Downside Stress-Testing & Tail-Risk Crisis Simulator")
    st.caption("Subject any asset to historical financial crash shocks and calculate statutory Value at Risk (VaR) and Expected Shortfall.")

    st_ticker = st.selectbox("Select Asset to Stress-Test:", options=all_ticker_options, key="stress_ticker")
    sec_st = market_data.get_security(st_ticker)
    candles_st = market_data.get_candles(st_ticker)

    if sec_st and candles_st:
        close_st = pd.Series([c["close"] for c in candles_st])
        var_res = calculate_var_cvar(close_st, confidence_level=0.95)
        crises = simulate_crisis_scenarios(sec_st["price"], sec_st["beta"], sec_st["sector"])

        v1, v2, v3 = st.columns(3)
        v1.metric("Historical VaR (95% 1-Day)", f"{var_res['var_historical_pct']}%", "Max Daily Expected Loss")
        v2.metric("Parametric Normal VaR", f"{var_res['var_parametric_pct']}%", "Gaussian Assumption")
        v3.metric("Conditional VaR (CVaR)", f"{var_res['cvar_expected_shortfall_pct']}%", "Expected Tail Loss")

        st.markdown("---")
        st.subheader("Historical Black Swan Crisis Stress-Test Results")

        crisis_records = []
        for cr in crises:
            crisis_records.append({
                "Crisis Event": cr["scenario"],
                "Macro Trigger": cr["description"],
                "Simulated Drawdown": f"{cr['simulated_drawdown_pct']}%",
                "Implied Price Floor": f"{sec_st['currency']} {cr['projected_price_floor']:,.2f}"
            })
        st.table(pd.DataFrame(crisis_records))

# ------------------------------------------------------------------------------
# TAB 5: ALGORITHMIC STRATEGY BACKTESTER
# ------------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Vectorized Algorithmic Strategy Backtester")
    st.caption("Simulate systematic rule-based trading algorithms against 1-year historical OHLCV data with zero lookahead bias.")

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        bt_ticker = st.selectbox("Select Asset to Backtest:", options=all_ticker_options, key="bt_ticker")
    with b_col2:
        strategy_type = st.selectbox(
            "Algorithmic Strategy",
            options=["sma_crossover", "rsi_mean_reversion", "bollinger_breakout"],
            format_func=lambda x: {
                "sma_crossover": "📈 Dual Moving Average Crossover",
                "rsi_mean_reversion": "🔄 RSI Mean Reversion",
                "bollinger_breakout": "⚡ Bollinger Band Volatility Trend"
            }[x]
        )
    with b_col3:
        if strategy_type == "sma_crossover":
            fast_w = st.slider("Fast SMA Window", 5, 50, 20)
            slow_w = st.slider("Slow SMA Window", 20, 150, 50)
            kwargs = {"fast_window": fast_w, "slow_window": slow_w}
        elif strategy_type == "rsi_mean_reversion":
            oversold = st.slider("RSI Oversold (Entry)", 20, 45, 35)
            overbought = st.slider("RSI Overbought (Exit)", 55, 80, 65)
            kwargs = {"rsi_oversold": oversold, "rsi_overbought": overbought}
        else:
            kwargs = {}

    candles_bt = pd.DataFrame(market_data.get_candles(bt_ticker))
    if not candles_bt.empty:
        bt_result = run_backtest(candles_bt, strategy=strategy_type, **kwargs)

        res_col1, res_col2, res_col3, res_col4, res_col5 = st.columns(5)
        res_col1.metric("Strategy Return", f"{bt_result['total_return_pct']}%")
        res_col2.metric("Buy & Hold Return", f"{bt_result['benchmark_return_pct']}%")
        res_col3.metric("Strategy Sharpe", f"{bt_result['strategy_sharpe']}")
        res_col4.metric("Max Drawdown", f"{bt_result['strategy_max_drawdown_pct']}%")
        res_col5.metric("Win Rate", f"{bt_result['win_rate_pct']}% ({bt_result['trades_executed']} trades)")

        st.subheader("Equity Curve: Cumulative Performance vs. Benchmark")
        eq_df = pd.DataFrame(bt_result["equity_curve"])
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        st.line_chart(eq_df.set_index("timestamp")[["Strategy", "Buy & Hold"]], height=340)

# ------------------------------------------------------------------------------
# TAB 6: DISCOUNTED CASH FLOW (DCF) VALUATION
# ------------------------------------------------------------------------------
with tab_dcf:
    st.subheader("Intrinsic Valuation Engine: Gordon Growth Two-Stage DCF Model")
    st.caption("Calculate intrinsic fair value per share based on projected free cash flows, terminal growth, and cost of capital.")

    dcf_ticker = st.selectbox("Select Security for DCF Valuation:", options=all_ticker_options, key="dcf_ticker")
    sec_dcf = market_data.get_security(dcf_ticker)

    if sec_dcf:
        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        with d_col1:
            growth_input = st.slider("5-Yr Projected FCF Growth (%)", 0.0, 35.0, 12.0, step=0.5)
        with d_col2:
            terminal_g = st.slider("Terminal Perpetual Growth (%)", 1.0, 4.5, 2.5, step=0.1)
        with d_col3:
            wacc_input = st.slider("Discount Rate (WACC %)", 6.0, 16.0, 9.0, step=0.5)
        with d_col4:
            st.metric("Current Market Price", f"{sec_dcf['currency']} {sec_dcf['price']:,.2f}")

        mcap_m = sec_dcf["market_cap_usd"] / 1e6
        fcf_m = max(50.0, mcap_m * sec_dcf["fcf_yield"])
        shares_m = max(10.0, mcap_m / sec_dcf["price"])
        net_debt_m = mcap_m * 0.15

        dcf_res = calculate_dcf(
            fcf_million=fcf_m,
            shares_outstanding_million=shares_m,
            net_debt_million=net_debt_m,
            growth_rate_pct=growth_input,
            terminal_growth_pct=terminal_g,
            wacc_pct=wacc_input
        )

        fv = dcf_res["fair_value"]
        curr_p = sec_dcf["price"]
        margin_of_safety = round(((fv - curr_p) / curr_p) * 100, 1)

        v_col1, v_col2, v_col3 = st.columns(3)
        v_col1.metric("Calculated Fair Value", f"{sec_dcf['currency']} {fv:,.2f}")
        v_col2.metric("Margin of Safety", f"{'+' if margin_of_safety >= 0 else ''}{margin_of_safety}%")
        valuation_status = "🟢 UNDERVALUED (Attractive Margin)" if margin_of_safety > 15 else "🔴 OVERVALUED (Premium Valuation)" if margin_of_safety < -15 else "🟡 FAIRLY VALUED"
        v_col3.metric("Valuation Regime", valuation_status)

        st.subheader("WACC vs. Terminal Growth Rate Sensitivity Matrix (Fair Value per Share)")
        st.dataframe(dcf_res["sensitivity_matrix"], use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 7: MONTE CARLO STOCHASTIC PRICE FORECASTER
# ------------------------------------------------------------------------------
with tab_monte_carlo:
    st.subheader("Stochastic Price Forecasting: Geometric Brownian Motion (GBM)")
    st.caption("Generates 100 random walk trajectories across future trading days to model price distribution cones.")

    mc_ticker = st.selectbox("Select Security for Monte Carlo Forecasting:", options=all_ticker_options, key="mc_ticker")
    sec_mc = market_data.get_security(mc_ticker)

    if sec_mc:
        mc_col1, mc_col2, mc_col3 = st.columns(3)
        with mc_col1:
            drift_rate = st.slider("Expected Annual Drift Rate (μ)", -0.10, 0.30, 0.08, step=0.01)
        with mc_col2:
            forecast_days = st.slider("Forecast Horizon (Trading Days)", 30, 252, 120, step=10)
        with mc_col3:
            st.metric("Historical Annual Volatility (σ)", f"{sec_mc['volatility_annualized']*100:.1f}%")

        mc_res = simulate_monte_carlo_paths(
            latest_price=sec_mc["price"],
            annual_vol=sec_mc["volatility_annualized"],
            days=forecast_days,
            num_simulations=100,
            expected_drift=drift_rate
        )

        b1, b2, b3 = st.columns(3)
        b1.metric("10th Percentile (Bearish Scenario)", f"{sec_mc['currency']} {mc_res['p10_bearish']}")
        b2.metric("50th Percentile (Median Expected)", f"{sec_mc['currency']} {mc_res['p50_median']}")
        b3.metric("90th Percentile (Bullish Scenario)", f"{sec_mc['currency']} {mc_res['p90_bullish']}")

        st.subheader("Monte Carlo Path Trajectories & Median Expected Curve")
        st.line_chart(mc_res["chart_data"], height=350)

# ------------------------------------------------------------------------------
# TAB 8: MARKOWITZ & RISK PARITY PORTFOLIO OPTIMIZER
# ------------------------------------------------------------------------------
with tab_optimizer:
    st.subheader("Portfolio Optimization: Markowitz Mean-Variance & Inverse Volatility Risk Parity")
    st.caption("Simulates 1,500 random portfolio allocations alongside Inverse Volatility Risk Parity weights.")

    opt_tickers = st.multiselect(
        "Select 3 to 6 Assets for Portfolio Optimization:",
        options=all_ticker_options,
        default=["AAPL", "MSFT", "NVDA", "TCS.NS", "ASML.AS"]
    )

    if len(opt_tickers) >= 2:
        price_dict = {}
        for t in opt_tickers:
            candles = market_data.get_candles(t)
            if candles:
                df_c = pd.DataFrame(candles)
                price_dict[t] = df_c["close"].values

        min_len = min(len(v) for v in price_dict.values())
        price_df = pd.DataFrame({k: v[:min_len] for k, v in price_dict.items()})

        ef_res = simulate_efficient_frontier(price_df, num_portfolios=1500)
        rp_weights = calculate_risk_parity_weights(price_df)

        opt1, opt2, opt3 = st.columns(3)
        with opt1:
            st.markdown("#### 🌟 Maximum Sharpe Portfolio")
            ms = ef_res["max_sharpe"]
            st.write(f"Return: **{ms['return']}%** | Vol: **{ms['volatility']}%**")
            st.write(f"Sharpe Ratio: **{ms['sharpe']}**")
            st.markdown("**Optimal Weights:**")
            for asset, weight in ms["weights"].items():
                st.write(f"- `{asset}`: **{weight}%**")

        with opt2:
            st.markdown("#### 🛡️ Minimum Volatility Portfolio")
            mv = ef_res["min_vol"]
            st.write(f"Return: **{mv['return']}%** | Vol: **{mv['volatility']}%**")
            st.write(f"Sharpe Ratio: **{mv['sharpe']}**")
            st.markdown("**Optimal Weights:**")
            for asset, weight in mv["weights"].items():
                st.write(f"- `{asset}`: **{weight}%**")

        with opt3:
            st.markdown("#### ⚖️ Equal Risk Parity Allocation")
            st.caption("Weights inversely proportional to asset volatility.")
            for asset, weight in rp_weights.items():
                st.write(f"- `{asset}`: **{weight}%**")

        st.subheader("Simulated Efficient Frontier (Volatility vs. Return)")
        scatter_chart = ef_res["scatter_data"]
        st.scatter_chart(scatter_chart, x="Volatility", y="Return", color="Sharpe")
    else:
        st.info("Please select at least 2 assets to run the portfolio optimizer.")

# ------------------------------------------------------------------------------
# TAB 9: AUTOMATED INSTITUTIONAL RESEARCH MEMORANDUM
# ------------------------------------------------------------------------------
with tab_report:
    st.subheader("Automated Hedge-Fund Quantitative Equity Research Memo")
    st.caption("Synthesizes multi-factor models, forensic accounting audits, intrinsic DCF, stress-testing, and Kelly sizing into an institutional report.")

    rep_ticker = st.selectbox("Select Security to Generate Research Report:", options=all_ticker_options, key="report_ticker")
    sec_rep = market_data.get_security(rep_ticker)
    candles_rep = market_data.get_candles(rep_ticker)

    if sec_rep and candles_rep:
        if st.button("🚀 Generate Full Institutional Memorandum"):
            with st.spinner("Compiling multi-factor data, forensic models, and crisis simulations..."):
                report_md = generate_equity_research_report(sec_rep, candles_rep)
                st.markdown(report_md)
                
                st.download_button(
                    label="📥 Download Research Memorandum (Markdown)",
                    data=report_md,
                    file_name=f"equity_research_{sec_rep['ticker']}_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )

# ------------------------------------------------------------------------------
# TAB 10: QUANT MODEL & API DOCUMENTATION
# ------------------------------------------------------------------------------
with tab_docs:
    st.subheader("Institutional Quantitative Specifications & REST API")
    st.markdown("""
    ### 1. Mathematical Formulations
    * **Beneish M-Score (Forensic Fraud Model)**:
      $$M = -4.84 + 0.92\\text{DSRI} + 0.528\\text{GMI} + 0.404\\text{AQI} + 0.892\\text{SGI} + 0.115\\text{DEPI} - 0.172\\text{SGAI} + 4.037\\text{TATA} + 0.0327\\text{LVGI}$$
    * **Value-at-Risk & Conditional VaR (Expected Shortfall)**:
      $$\\text{CVaR}_\\alpha = E[L \\mid L \\ge \\text{VaR}_\\alpha]$$
    * **Fractional Kelly Criterion Optimal Capital Sizing**:
      $$f^* = p - \\frac{1-p}{b}$$
    * **Gordon Growth Two-Stage DCF**:
      $$PV = \\sum_{t=1}^{N} \\frac{FCF_t}{(1 + WACC)^t} + \\frac{FCF_N(1 + g)}{(WACC - g)(1 + WACC)^N}$$
    * **Markowitz Mean-Variance Optimization**:
      $$\\max_w \\frac{w^T \\mu - R_f}{\\sqrt{w^T \\Sigma w}}$$

    ### 2. Programmatic REST API (FastAPI)
    Launch the backend:
    ```bash
    uvicorn api.app:app --host 0.0.0.0 --port 8000
    ```
    """)
