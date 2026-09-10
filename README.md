# OmniScreen: Quantitative Equity Research Terminal

A single-file quantitative equity research and screening platform designed for global multi-asset analysis across 115+ securities in the United States, India, Europe, the United Kingdom, Japan, and macro commodities/crypto.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Architecture & Quantitative Modules

1. **Screener**: AST multi-factor filter engine scanning 115+ assets across market cap tiers (Mega, Large, Mid), valuation multiples, profitability, leverage, and momentum.
2. **Company Profile**: 1-year historical OHLCV price series with SMA overlays (20, 50, 200 days), RSI, and MACD indicators, accompanied by algorithmic regime and pattern detection.
3. **DuPont 5-Way ROE Decomposition**: Deconstructs return on equity into Tax Burden, Interest Burden, Operating Margin, Asset Turnover, and Financial Leverage to identify operational vs. debt-driven profitability.
4. **Forensic Accounting Audit**: Evaluates earnings manipulation risks using the 8-variable Beneish M-Score, bankruptcy risk via the Altman Z-Score, and operational health via the 9-point Piotroski F-Score.
5. **Macro Stress-Testing & Tail Risk**: Historical crisis shock simulations (2008 GFC, 2020 COVID, 2022 Rate Spike, 2000 Dot-Com) alongside parametric and historical Value-at-Risk (95% Daily VaR) and Conditional VaR (Expected Shortfall).
6. **Strategy Backtester**: Vectorized simulation of systematic strategies (Dual Moving Average Crossover, RSI Mean Reversion, Bollinger Volatility Trend) with Strategy vs. Buy & Hold equity curves.
7. **DCF Valuation**: Two-stage Gordon Growth Discounted Cash Flow model with a 2D WACC vs. Terminal Growth sensitivity matrix.
8. **Monte Carlo Price Forecaster (GBM)**: 100 stochastic price paths projected over a 252-day horizon calculating 10th (bearish), 50th (median), and 90th (bullish) percentile price distribution cones.
9. **Portfolio Optimization**: Modern Portfolio Theory (Markowitz Mean-Variance 1,500-portfolio simulation solving for Maximum Sharpe and Minimum Volatility) and Equal Risk Contribution (Inverse Volatility Risk Parity).
10. **Research Memorandum**: One-click generation of formal equity research reports with thesis summary, factor attribution, downside boundaries, and fractional Kelly capital sizing, downloadable in Markdown.

## Global Coverage (115+ Assets)

- **United States**: S&P 500 / Nasdaq leaders across Tech (AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AVGO, CRM, AMD, CSCO, PLTR, PANW, UBER), Financials (JPM, V, MA, BAC, GS, BLK, AXP), Healthcare (LLY, UNH, JNJ, ABBV, ISRG, MRK, TMO, ABT), Consumer (WMT, COST, PG, HD, KO, PEP, MCD, NKE), and Industrials/Energy (CAT, GE, XOM, CVX, LMT).
- **India (NSE)**: Nifty 50 and sectoral leaders (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, BHARTIARTL, ITC, LT, TATAMOTORS, SBIN, BAJFINANCE, MARUTI, HAL, SUNPHARMA, BEL, AXISBANK, KOTAKBANK, M&M, NTPC, ONGC, TATASTEEL).
- **United Kingdom & Europe**: LSE, Euronext, XETRA, and OMX champions (AZN, SHEL, HSBA, ULVR, BP, RR, RIO, GSK, ASML, SAP, MC, NOVO-B, SIE, TTE, BMW, AIR, SAN, RMS).
- **Japan (TSE)**: Nikkei 225 leaders (Toyota, Sony, Keyence, Mitsubishi UFJ, Tokyo Electron, Nintendo).
- **Macro & Digital Assets**: Gold Futures (`GC=F`), Bitcoin (`BTC-USD`), Ethereum (`ETH-USD`).
- **On-Demand Ticker Streaming**: Enter any global ticker in the sidebar to stream live quotes and historical data on demand via `yfinance`.
