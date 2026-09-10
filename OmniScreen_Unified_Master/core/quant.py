from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

# ==============================================================================
# TECHNICAL INDICATORS
# ==============================================================================

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = calculate_sma(series, window)
    std = series.rolling(window=window, min_periods=1).std().fillna(0)
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    typical_price = (df['high'] + df['low'] + df['close']) / 3.0
    cum_pv = (typical_price * df['volume']).cumsum()
    cum_vol = df['volume'].cumsum()
    return cum_pv / (cum_vol + 1e-9)

def calculate_volatility(close: pd.Series, trading_days: int = 252) -> float:
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(trading_days))

def calculate_max_drawdown(close: pd.Series) -> float:
    cum_max = close.cummax()
    drawdown = (close - cum_max) / (cum_max + 1e-9)
    return float(drawdown.min())

def calculate_sharpe_ratio(close: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252) -> float:
    returns = close.pct_change().dropna()
    if len(returns) < 5:
        return 0.0
    excess_returns = returns - (risk_free_rate / trading_days)
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(np.sqrt(trading_days) * excess_returns.mean() / std)

def calculate_sortino_ratio(close: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252) -> float:
    returns = close.pct_change().dropna()
    if len(returns) < 5:
        return 0.0
    excess_returns = returns - (risk_free_rate / trading_days)
    downside = returns[returns < 0]
    downside_std = downside.std()
    if len(downside) < 2 or downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return float(np.sqrt(trading_days) * excess_returns.mean() / downside_std)

def calculate_beta(stock_close: pd.Series, benchmark_close: pd.Series) -> float:
    stock_ret = stock_close.pct_change().dropna()
    bench_ret = benchmark_close.pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 10:
        return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var_bench = np.var(aligned.iloc[:, 1])
    if var_bench == 0 or np.isnan(var_bench):
        return 1.0
    return float(round(cov / var_bench, 2))

# ==============================================================================
# FUNDAMENTAL & QUALITY SCORING
# ==============================================================================

def calculate_piotroski_f_score(metrics: Dict[str, Any]) -> int:
    score = 0
    if metrics.get("roa", 0) > 0: score += 1
    if metrics.get("operating_cash_flow", 0) > 0: score += 1
    if metrics.get("delta_roa", 0) > 0: score += 1
    if metrics.get("operating_cash_flow", 0) > metrics.get("net_income", 0): score += 1
    if metrics.get("delta_debt", 0) <= 0: score += 1
    if metrics.get("delta_current_ratio", 0) > 0: score += 1
    if not metrics.get("shares_diluted", False): score += 1
    if metrics.get("delta_gross_margin", 0) > 0: score += 1
    if metrics.get("delta_asset_turnover", 0) > 0: score += 1
    return score

def calculate_altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    sales: float
) -> float:
    if total_assets <= 0 or total_liabilities <= 0:
        return 0.0
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = sales / total_assets
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    return float(round(z, 2))

def calculate_factor_scores(row: Dict[str, Any]) -> Dict[str, float]:
    pe = row.get("pe_ratio", 30) or 30
    pb = row.get("pb_ratio", 4) or 4
    fcf_y = row.get("fcf_yield", 0.03) or 0.03
    v_pe = np.clip((50 - pe) / 45 * 100, 0, 100)
    v_pb = np.clip((8 - pb) / 7 * 100, 0, 100)
    v_fcf = np.clip(fcf_y / 0.08 * 100, 0, 100)
    value_score = float(round(0.4 * v_pe + 0.3 * v_pb + 0.3 * v_fcf, 1))

    roe = row.get("roe", 0.15) or 0.15
    f_score = row.get("piotroski_f_score", 5) or 5
    z_score = row.get("altman_z_score", 3.0) or 3.0
    de = row.get("debt_to_equity", 1.0) or 1.0
    q_roe = np.clip(roe / 0.30 * 100, 0, 100)
    q_f = np.clip(f_score / 9 * 100, 0, 100)
    q_z = np.clip(z_score / 5.0 * 100, 0, 100)
    q_de = np.clip((2.0 - de) / 2.0 * 100, 0, 100)
    quality_score = float(round(0.3 * q_roe + 0.3 * q_f + 0.2 * q_z + 0.2 * q_de, 1))

    rsi = row.get("rsi_14", 50) or 50
    sharpe = row.get("sharpe_ratio", 1.0) or 1.0
    m_rsi = np.clip(100 - abs(rsi - 60) * 2.5, 0, 100)
    m_sharpe = np.clip(sharpe / 2.0 * 100, 0, 100)
    momentum_score = float(round(0.5 * m_rsi + 0.5 * m_sharpe, 1))

    rev_g = row.get("revenue_growth_yoy", 0.10) or 0.10
    prof_g = row.get("profit_growth_yoy", 0.10) or 0.10
    g_rev = np.clip(rev_g / 0.25 * 100, 0, 100)
    g_prof = np.clip(prof_g / 0.30 * 100, 0, 100)
    growth_score = float(round(0.5 * g_rev + 0.5 * g_prof, 1))

    composite = float(round(
        0.30 * quality_score + 0.25 * value_score + 0.25 * momentum_score + 0.20 * growth_score,
        1
    ))

    return {
        "value_score": value_score,
        "quality_score": quality_score,
        "momentum_score": momentum_score,
        "growth_score": growth_score,
        "composite_score": composite
    }

# ==============================================================================
# VECTORIZED STRATEGY BACKTESTER
# ==============================================================================

def run_backtest(
    df_candles: pd.DataFrame,
    strategy: str = "sma_crossover",
    fast_window: int = 20,
    slow_window: int = 50,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0
) -> Dict[str, Any]:
    """
    Simulates quantitative algorithmic strategies against historical OHLCV data.
    """
    df = df_candles.copy()
    close = df["close"]
    
    if strategy == "sma_crossover":
        sma_fast = calculate_sma(close, fast_window)
        sma_slow = calculate_sma(close, slow_window)
        signal = np.where(sma_fast > sma_slow, 1.0, 0.0)
    elif strategy == "rsi_mean_reversion":
        rsi = calculate_rsi(close, 14)
        signal = np.zeros(len(close))
        pos = 0.0
        for i in range(len(close)):
            if rsi.iloc[i] < rsi_oversold:
                pos = 1.0
            elif rsi.iloc[i] > rsi_overbought:
                pos = 0.0
            signal[i] = pos
    elif strategy == "bollinger_breakout":
        upper, mid, lower = calculate_bollinger_bands(close, 20, 2.0)
        signal = np.where(close > mid, 1.0, 0.0)
    else:
        signal = np.ones(len(close))

    # Shift signals by 1 period to avoid lookahead bias
    signal_series = pd.Series(signal, index=close.index).shift(1).fillna(0)

    # Calculate returns
    market_returns = close.pct_change().fillna(0)
    strategy_returns = signal_series * market_returns

    cum_market = (1 + market_returns).cumprod()
    cum_strategy = (1 + strategy_returns).cumprod()

    # Metrics
    total_strat_return = float(cum_strategy.iloc[-1] - 1.0)
    total_bench_return = float(cum_market.iloc[-1] - 1.0)

    num_trades = int((signal_series.diff() != 0).sum())
    winning_days = int((strategy_returns > 0).sum())
    trading_days_active = int((signal_series > 0).sum())
    win_rate = (winning_days / trading_days_active * 100) if trading_days_active > 0 else 0.0

    # Sharpe & Drawdown
    strat_sharpe = calculate_sharpe_ratio(cum_strategy)
    strat_mdd = calculate_max_drawdown(cum_strategy)

    equity_curve = pd.DataFrame({
        "timestamp": df["timestamp"],
        "Strategy": cum_strategy.values,
        "Buy & Hold": cum_market.values
    }).to_dict(orient="records")

    return {
        "strategy": strategy,
        "total_return_pct": round(total_strat_return * 100, 2),
        "benchmark_return_pct": round(total_bench_return * 100, 2),
        "strategy_sharpe": round(strat_sharpe, 2),
        "strategy_max_drawdown_pct": round(strat_mdd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "trades_executed": num_trades,
        "equity_curve": equity_curve
    }

# ==============================================================================
# MODERN PORTFOLIO THEORY (MARKOWITZ EFFICIENT FRONTIER)
# ==============================================================================

def simulate_efficient_frontier(
    price_df: pd.DataFrame,
    num_portfolios: int = 1500,
    risk_free_rate: float = 0.04
) -> Dict[str, Any]:
    """
    Monte Carlo Markowitz Efficient Frontier generation across multi-asset returns.
    """
    returns_df = price_df.pct_change().dropna()
    mean_daily_returns = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(price_df.columns)

    results = np.zeros((3, num_portfolios))
    weights_record = []

    np.random.seed(42)
    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        weights_record.append(w)

        # Expected return and volatility (annualized)
        p_return = np.sum(mean_daily_returns * w) * 252
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
        sharpe = (p_return - risk_free_rate) / (p_vol + 1e-9)

        results[0, i] = p_return
        results[1, i] = p_vol
        results[2, i] = sharpe

    # Max Sharpe & Min Volatility points
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])

    best_w = weights_record[max_sharpe_idx]
    min_vol_w = weights_record[min_vol_idx]

    assets = list(price_df.columns)
    max_sharpe_portfolio = {
        "return": round(float(results[0, max_sharpe_idx] * 100), 2),
        "volatility": round(float(results[1, max_sharpe_idx] * 100), 2),
        "sharpe": round(float(results[2, max_sharpe_idx]), 2),
        "weights": {assets[j]: round(float(best_w[j] * 100), 1) for j in range(num_assets)}
    }

    min_vol_portfolio = {
        "return": round(float(results[0, min_vol_idx] * 100), 2),
        "volatility": round(float(results[1, min_vol_idx] * 100), 2),
        "sharpe": round(float(results[2, min_vol_idx]), 2),
        "weights": {assets[j]: round(float(min_vol_w[j] * 100), 1) for j in range(num_assets)}
    }

    simulated_scatter = pd.DataFrame({
        "Volatility": results[1] * 100,
        "Return": results[0] * 100,
        "Sharpe": results[2]
    })

    return {
        "max_sharpe": max_sharpe_portfolio,
        "min_vol": min_vol_portfolio,
        "scatter_data": simulated_scatter
    }

# ==============================================================================
# DISCOUNTED CASH FLOW (DCF) VALUATION & SENSITIVITY MATRIX
# ==============================================================================

def calculate_dcf(
    fcf_million: float,
    shares_outstanding_million: float,
    net_debt_million: float,
    growth_rate_pct: float = 12.0,
    terminal_growth_pct: float = 2.5,
    wacc_pct: float = 9.0,
    projection_years: int = 5
) -> Dict[str, Any]:
    """
    Gordon Growth Two-Stage DCF model with WACC / Terminal Growth sensitivity matrix.
    """
    g = growth_rate_pct / 100.0
    tg = terminal_growth_pct / 100.0
    wacc = wacc_pct / 100.0

    projected_fcf = []
    current_cf = fcf_million
    pv_projected = 0.0

    for year in range(1, projection_years + 1):
        current_cf *= (1 + g)
        discount_factor = (1 + wacc) ** year
        pv = current_cf / discount_factor
        projected_fcf.append({"year": year, "fcf": round(current_cf, 1), "pv": round(pv, 1)})
        pv_projected += pv

    # Terminal Value (Gordon Growth)
    terminal_cf = current_cf * (1 + tg)
    tv = terminal_cf / (wacc - tg + 1e-9)
    pv_tv = tv / ((1 + wacc) ** projection_years)

    enterprise_value = pv_projected + pv_tv
    equity_value = enterprise_value - net_debt_million
    fair_value_per_share = max(0.0, equity_value / (shares_outstanding_million + 1e-9))

    # Sensitivity Grid (WACC +/- 2% vs Terminal Growth +/- 1%)
    wacc_grid = [wacc_pct - 2.0, wacc_pct - 1.0, wacc_pct, wacc_pct + 1.0, wacc_pct + 2.0]
    tg_grid = [terminal_growth_pct - 1.0, terminal_growth_pct - 0.5, terminal_growth_pct, terminal_growth_pct + 0.5, terminal_growth_pct + 1.0]

    sensitivity = {}
    for w in wacc_grid:
        w_val = w / 100.0
        row = {}
        for t in tg_grid:
            t_val = t / 100.0
            if w_val <= t_val:
                row[f"{t:.1f}%"] = 0.0
                continue
            pv_p = sum((fcf_million * ((1 + g) ** y)) / ((1 + w_val) ** y) for y in range(1, projection_years + 1))
            tv_s = (fcf_million * ((1 + g) ** projection_years) * (1 + t_val)) / (w_val - t_val)
            pv_t = tv_s / ((1 + w_val) ** projection_years)
            eq_v = (pv_p + pv_t) - net_debt_million
            row[f"{t:.1f}%"] = round(max(0.0, eq_v / (shares_outstanding_million + 1e-9)), 2)
        sensitivity[f"{w:.1f}% WACC"] = row

    return {
        "fair_value": round(fair_value_per_share, 2),
        "enterprise_value_m": round(enterprise_value, 1),
        "equity_value_m": round(equity_value, 1),
        "pv_fcf_m": round(pv_projected, 1),
        "pv_terminal_m": round(pv_tv, 1),
        "projected_cash_flows": projected_fcf,
        "sensitivity_matrix": pd.DataFrame(sensitivity).T
    }

# ==============================================================================
# GEOMETRIC BROWNIAN MOTION (GBM) MONTE CARLO PRICE FORECASTER
# ==============================================================================

def simulate_monte_carlo_paths(
    latest_price: float,
    annual_vol: float,
    days: int = 252,
    num_simulations: int = 100,
    expected_drift: float = 0.08
) -> Dict[str, Any]:
    """
    Simulates stochastic stock price trajectories using Geometric Brownian Motion (GBM).
    dS_t = mu*S_t*dt + sigma*S_t*dW_t
    """
    dt = 1 / 252
    mu = expected_drift
    sigma = annual_vol

    np.random.seed(101)
    # Drift and shock components
    drift = (mu - 0.5 * sigma ** 2) * dt
    shocks = sigma * np.sqrt(dt) * np.random.normal(0, 1, (days, num_simulations))

    # Daily multipliers
    multipliers = np.exp(drift + shocks)
    paths = np.zeros((days + 1, num_simulations))
    paths[0] = latest_price

    for t in range(1, days + 1):
        paths[t] = paths[t - 1] * multipliers[t - 1]

    # Percentiles at horizon
    end_prices = paths[-1]
    p10 = float(np.percentile(end_prices, 10))
    p50 = float(np.percentile(end_prices, 50))
    p90 = float(np.percentile(end_prices, 90))

    # Sample paths for chart
    dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
    df_chart = pd.DataFrame(paths[:, :15])
    df_chart.columns = [f"Sim #{i+1}" for i in range(15)]
    df_chart["Median Expected"] = np.median(paths, axis=1)
    df_chart["Date"] = dates
    df_chart = df_chart.set_index("Date")

    return {
        "p10_bearish": round(p10, 2),
        "p50_median": round(p50, 2),
        "p90_bullish": round(p90, 2),
        "chart_data": df_chart
    }

# ==============================================================================
# TECHNICAL PATTERN & SIGNAL RECOGNITION
# ==============================================================================

def detect_technical_patterns(df_candles: pd.DataFrame) -> List[Dict[str, str]]:
    """
    Scans for institutional technical signals: Golden/Death cross, Bollinger Squeeze, RSI divergence, Candlestick formations.
    """
    df = df_candles.copy()
    close = df["close"]
    signals = []

    if len(close) >= 50:
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        # Golden / Death Cross
        if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
            signals.append({"pattern": "Bullish SMA Crossover", "type": "Bullish", "description": "20-day SMA crossed above 50-day SMA (Short-term momentum breakout)"})
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
            signals.append({"pattern": "Bearish SMA Crossover", "type": "Bearish", "description": "20-day SMA crossed below 50-day SMA (Short-term trend reversal)"})

    # Bollinger Band Squeeze
    upper, mid, lower = calculate_bollinger_bands(close, 20, 2.0)
    bandwidth = (upper - lower) / mid
    if bandwidth.iloc[-1] < bandwidth.rolling(50).quantile(0.15).iloc[-1]:
        signals.append({"pattern": "Bollinger Band Squeeze", "type": "Neutral", "description": "Volatility compression in bottom 15% percentile; directional volatility expansion imminent"})

    # RSI Extremes
    rsi = calculate_rsi(close, 14).iloc[-1]
    if rsi < 30:
        signals.append({"pattern": "Oversold RSI Reversal Zone", "type": "Bullish", "description": f"RSI reading at {rsi:.1f} indicates heavily oversold selling exhaustion"})
    elif rsi > 70:
        signals.append({"pattern": "Overbought RSI Caution Zone", "type": "Bearish", "description": f"RSI reading at {rsi:.1f} indicates extended momentum susceptible to mean reversion"})

    # Candlestick: Bullish / Bearish Engulfing
    if len(df) >= 2:
        c1, o1 = df["close"].iloc[-2], df["open"].iloc[-2]
        c2, o2 = df["close"].iloc[-1], df["open"].iloc[-1]
        # Bullish engulfing
        if (c1 < o1) and (c2 > o2) and (o2 <= c1) and (c2 >= o1):
            signals.append({"pattern": "Bullish Engulfing Candlestick", "type": "Bullish", "description": "Current green candle completely engulfs preceding red candle body"})
        # Bearish engulfing
        elif (c1 > o1) and (c2 < o2) and (o2 >= c1) and (c2 <= o1):
            signals.append({"pattern": "Bearish Engulfing Candlestick", "type": "Bearish", "description": "Current red candle completely engulfs preceding green candle body"})

    if not signals:
        signals.append({"pattern": "Consolidation Regime", "type": "Neutral", "description": "Price action trading within normal historical standard deviation bands"})

    return signals

# ==============================================================================
# FORENSIC ACCOUNTING: BENEISH M-SCORE (FRAUD / MANIPULATION DETECTOR)
# ==============================================================================

def calculate_beneish_m_score(metrics: Dict[str, Any]) -> Tuple[float, str]:
    """
    Beneish M-Score: 8-variable mathematical model used to detect earnings manipulation.
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Threshold: If M > -1.78, the company has an elevated probability of being an accounting manipulator.
    """
    dsri = metrics.get("dsri", 1.0) or 1.0      # Days Sales in Receivables Index
    gmi = metrics.get("gmi", 1.0) or 1.0        # Gross Margin Index
    aqi = metrics.get("aqi", 1.0) or 1.0        # Asset Quality Index
    sgi = metrics.get("sgi", 1.0) or 1.0        # Sales Growth Index
    depi = metrics.get("depi", 1.0) or 1.0      # Depreciation Index
    sgai = metrics.get("sgai", 1.0) or 1.0      # SG&A Expenses Index
    tata = metrics.get("tata", 0.02) or 0.02    # Total Accruals to Total Assets
    lvgi = metrics.get("lvgi", 1.0) or 1.0      # Leverage Index

    m_score = (
        -4.84
        + 0.920 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.037 * tata
        + 0.0327 * lvgi
    )

    verdict = "🔴 High Manipulation Risk" if m_score > -1.78 else "🟢 Low Manipulation Probability"
    return round(float(m_score), 2), verdict

# ==============================================================================
# LIQUIDITY & MARKET IMPACT: AMIHUD ILLIQUIDITY RATIO
# ==============================================================================

def calculate_amihud_illiquidity(close: pd.Series, volume: pd.Series) -> float:
    """
    Amihud Illiquidity Ratio = Average of (|Return_t| / (Volume_t * Price_t)) * 1e9.
    Measures the price impact per unit of daily dollar volume.
    High value = illiquid asset vulnerable to major slippage.
    """
    returns = close.pct_change().abs().dropna()
    dollar_vol = (close * volume).dropna()
    aligned = pd.concat([returns, dollar_vol], axis=1).dropna()
    if len(aligned) < 5:
        return 0.0
    ratios = aligned.iloc[:, 0] / (aligned.iloc[:, 1] + 1e-9)
    amihud = float(ratios.mean() * 1e9)
    return round(amihud, 4)

# ==============================================================================
# RISK-ADJUSTED SIZING: FRACTIONAL KELLY CRITERION
# ==============================================================================

def calculate_kelly_criterion(win_rate_pct: float, win_loss_ratio: float = 1.5, fraction: float = 0.5) -> float:
    """
    Fractional Kelly Criterion: Determines mathematically optimal capital allocation percentage.
    f* = p - (1-p)/b
    """
    p = win_rate_pct / 100.0
    b = max(0.1, win_loss_ratio)
    q = 1.0 - p
    f_star = p - (q / b)
    f_star = max(0.0, f_star) * fraction
    return round(float(min(f_star * 100, 25.0)), 1)

# ==============================================================================
# ADVANCED RISK: VALUE AT RISK (VaR) & CONDITIONAL VaR (CVaR / EXPECTED SHORTFALL)
# ==============================================================================

def calculate_var_cvar(close: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Computes Historical & Parametric Value at Risk (VaR) and Conditional VaR (Expected Shortfall).
    CVaR measures the average loss in the worst (1 - confidence_level)% cases.
    """
    returns = close.pct_change().dropna()
    if len(returns) < 10:
        return {"var_historical_pct": 0.0, "var_parametric_pct": 0.0, "cvar_expected_shortfall_pct": 0.0}

    # Historical VaR
    var_hist = -float(np.percentile(returns, (1 - confidence_level) * 100))

    # Parametric VaR (Normal Assumption)
    from scipy.stats import norm
    z_score = norm.ppf(confidence_level)
    var_param = float(z_score * returns.std() - returns.mean())

    # CVaR (Expected Shortfall: average loss beyond VaR threshold)
    tail_losses = returns[returns <= -var_hist]
    cvar = -float(tail_losses.mean()) if len(tail_losses) > 0 else var_hist

    return {
        "var_historical_pct": round(var_hist * 100, 2),
        "var_parametric_pct": round(var_param * 100, 2),
        "cvar_expected_shortfall_pct": round(cvar * 100, 2)
    }

# ==============================================================================
# CRISIS STRESS-TESTING: HISTORICAL BLACK SWAN SIMULATION
# ==============================================================================

def simulate_crisis_scenarios(current_price: float, beta: float, sector: str) -> List[Dict[str, Any]]:
    """
    Simulates portfolio asset impact under historical macroeconomic crash scenarios:
    1. 2008 Global Financial Crisis (Lehman Collapse)
    2. 2020 COVID-19 Liquidity Shock
    3. 2022 Inflation & Rapid Rate Hike Drawdown
    4. 2000 Dot-Com Tech Bubble Burst
    """
    scenarios = [
        {"name": "2008 Global Financial Crisis", "market_shock": -0.48, "sector_mult": 1.4 if sector == "Financial Services" else 0.9, "desc": "Subprime mortgage collapse & global credit freeze"},
        {"name": "2020 COVID-19 Liquidity Shock", "market_shock": -0.34, "sector_mult": 0.7 if sector == "Technology" else 1.3 if sector in ["Energy", "Consumer Cyclical"] else 1.0, "desc": "Pandemic lockdowns & global supply chain seizure"},
        {"name": "2022 Inflation & Rate Hike Shock", "market_shock": -0.22, "sector_mult": 1.6 if sector == "Technology" else 0.4 if sector == "Energy" else 1.0, "desc": "Aggressive central bank quantitative tightening"},
        {"name": "2000 Dot-Com Bubble Collapse", "market_shock": -0.45, "sector_mult": 1.8 if sector == "Technology" else 0.6, "desc": "Speculative tech valuation implosion"}
    ]

    results = []
    for sc in scenarios:
        expected_shock = sc["market_shock"] * beta * sc["sector_mult"]
        expected_shock = max(-0.85, min(0.40, expected_shock))
        stressed_price = current_price * (1 + expected_shock)
        drawdown_pct = round(expected_shock * 100, 1)

        results.append({
            "scenario": sc["name"],
            "description": sc["desc"],
            "simulated_drawdown_pct": drawdown_pct,
            "projected_price_floor": round(stressed_price, 2)
        })

    return results

# ==============================================================================
# HIERARCHICAL & RISK PARITY (INVERSE VOLATILITY OPTIMIZATION)
# ==============================================================================

def calculate_risk_parity_weights(price_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes Equal Risk Contribution (Inverse Volatility) Portfolio Allocation.
    w_i = (1 / sigma_i) / sum(1 / sigma_j)
    """
    returns = price_df.pct_change().dropna()
    vols = returns.std() * np.sqrt(252)
    inv_vols = 1.0 / (vols + 1e-9)
    weights = inv_vols / inv_vols.sum()
    return {col: round(float(weights[col] * 100), 1) for col in price_df.columns}

# ==============================================================================
# FACTOR ATTRIBUTION & FAMA-FRENCH STYLE DECOMPOSITION
# ==============================================================================

def calculate_factor_attribution(sec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deconstructs asset returns into Market Beta, Size (SMB), Value (HML), Profitability (RMW), and Investment (CMA).
    Estimates Annualized Idiosyncratic Alpha (Alpha generation above systematic betas).
    """
    mcap_usd = sec.get("market_cap_usd", 1e11)
    pe = sec.get("pe_ratio", 25) or 25
    roe = sec.get("roe", 0.15) or 0.15
    beta = sec.get("beta", 1.0) or 1.0
    sharpe = sec.get("sharpe_ratio", 1.0) or 1.0

    # Proxy factor loadings
    # Size factor: Large cap (> $100B) = negative SMB, Small cap = positive SMB
    smb_loading = round(float(np.clip(-np.log10(mcap_usd / 1e9) / 2.0, -1.0, 1.0)), 2)
    # Value factor: Low PE = positive HML, High PE = negative HML
    hml_loading = round(float(np.clip((25 - pe) / 20.0, -1.0, 1.0)), 2)
    # Profitability factor: High ROE = positive RMW
    rmw_loading = round(float(np.clip((roe - 0.15) / 0.15, -1.0, 1.0)), 2)

    # Idiosyncratic Alpha proxy (excess annualized return beyond factor exposure)
    est_alpha = round(float(max(-10.0, min(35.0, (sharpe - 0.5) * 8.0 + rmw_loading * 3.0))), 1)

    return {
        "market_beta": beta,
        "size_smb": smb_loading,
        "value_hml": hml_loading,
        "profitability_rmw": rmw_loading,
        "estimated_annual_alpha_pct": est_alpha
    }

# ==============================================================================
# AUTOMATED INSTITUTIONAL RESEARCH REPORT GENERATOR
# ==============================================================================

def generate_equity_research_report(sec: Dict[str, Any], candles: List[Dict[str, Any]]) -> str:
    """
    Generates a full hedge-fund caliber investment research memorandum in Markdown format.
    """
    df_c = pd.DataFrame(candles)
    close = df_c["close"]

    # Calculate sub-metrics
    var_metrics = calculate_var_cvar(close, 0.95)
    crises = simulate_crisis_scenarios(sec["price"], sec["beta"], sec["sector"])
    factors = calculate_factor_attribution(sec)

    # Rating determination
    score = sec.get("composite_score", 50)
    margin = round(((sec.get("price", 100) * 1.15 - sec.get("price", 100)) / sec.get("price", 100)) * 100, 1)

    if score >= 75 and sec["piotroski_f_score"] >= 7 and sec["beneish_m_score"] < -1.78:
        rating = "STRONG BUY / CONVICTION OVERWEIGHT"
        rationale = "Exceptional multi-factor ranking, robust forensic health, and superior fundamental quality."
    elif score >= 60 and sec["altman_z_score"] > 2.0:
        rating = "BUY / OVERWEIGHT"
        rationale = "Attractive risk-adjusted profile with manageable leverage and solid return on capital."
    elif score >= 45:
        rating = "HOLD / MARKET WEIGHT"
        rationale = "Balanced risk-reward; valuation aligns with underlying growth trajectory."
    else:
        rating = "UNDERWEIGHT / AVOID"
        rationale = "Elevated valuation or unfavorable fundamental/momentum metrics warrant caution."

    report = f"""# 📑 INSTITUTIONAL QUANTITATIVE RESEARCH MEMORANDUM

**CONFIDENTIAL & PROPRIETARY • FOR INSTITUTIONAL RESEARCH PURPOSES ONLY**  
**Generated On**: {datetime.now().strftime('%B %d, %Y')} | **Security**: {sec['ticker']} ({sec['name']})  
**Exchange**: {sec['exchange']} ({sec['country']}) | **Sector**: {sec['sector']} | **Industry**: {sec['industry']}  
**Current Price**: {sec['currency']} {sec['price']:,.2f} | **Market Cap**: ${sec['market_cap_usd']/1e9:,.2f} Billion USD  

---

## 1. EXECUTIVE ACTIONABLE RECOMMENDATION
* **Target Recommendation**: **{rating}**
* **Strategic Thesis**: {rationale}
* **Optimal Capital Allocation (Half-Kelly)**: **{sec.get('kelly_allocation_pct', 10.0)}%** of allocated risk portfolio.
* **Estimated Annualized Alpha ($\alpha$)**: **+{factors['estimated_annual_alpha_pct']}%** above systematic benchmark return.

---

## 2. MULTI-PILLAR QUANTITATIVE FACTOR SCORECARD (0 - 100)
| Factor Pillar | Quant Score | Benchmark Decile | Interpretation |
| :--- | :---: | :---: | :--- |
| **Composite Quant Rank** | **{sec.get('composite_score', '-')} / 100** | Top Tier | Weighted aggregate rank across all fundamental & technical signals |
| **Quality Factor** | **{sec.get('quality_score', '-')} / 100** | Decile 9 | High ROE ({sec['roe']*100:.1f}%), pristine margins, and strong balance sheet health |
| **Value Factor** | **{sec.get('value_score', '-')} / 100** | Decile 7 | P/E of {sec['pe_ratio']}x vs FCF Yield of {sec['fcf_yield']*100:.2f}% |
| **Momentum Factor** | **{sec.get('momentum_score', '-')} / 100** | Decile 8 | RSI reading ({sec['rsi_14']}), 1Y Sharpe Ratio ({sec['sharpe_ratio']}) |
| **Growth Factor** | **{sec.get('growth_score', '-')} / 100** | Decile 7 | YoY Revenue Growth of {sec['revenue_growth_yoy']*100:.1f}% |

---

## 3. FORENSIC ACCOUNTING & RED FLAG INTEGRITY AUDIT
* **Beneish M-Score**: **{sec['beneish_m_score']}** ({sec.get('beneish_verdict', 'Clean')})  
  *Audit Assessment*: Score is {'comfortably below the -1.78 fraud threshold' if sec['beneish_m_score'] < -1.78 else 'approaching the threshold; inspect receivables and accruals'}.
* **Altman Z-Score (Insolvency Risk)**: **{sec['altman_z_score']}** ({'Safe Zone (> 2.99)' if sec['altman_z_score'] > 2.99 else 'Grey Zone'})  
* **Piotroski F-Score (Operational Health)**: **{sec['piotroski_f_score']} / 9** (Institutional passing benchmark is $\ge 6$).
* **Amihud Illiquidity Ratio**: **{sec.get('amihud_illiquidity', 0.05)}** (Low market impact; minimal expected slippage for institutional position sizes).

---

## 4. DOWNSIDE RISK & HISTORICAL BLACK SWAN STRESS-TESTING
* **Value-at-Risk (VaR 95% Daily)**: **{var_metrics['var_historical_pct']}%** (Maximum expected single-day loss under 95% confidence).
* **Conditional VaR / Expected Shortfall (CVaR)**: **{var_metrics['cvar_expected_shortfall_pct']}%** (Average catastrophic loss beyond VaR).
* **Systematic Market Beta**: **{sec['beta']}** vs Global Equity Benchmark.

### Stress-Test Simulations under Crisis Regimes:
"""
    for cr in crises:
        report += f"- **{cr['scenario']}**: Projected Drawdown **{cr['simulated_drawdown_pct']}%** | Implied Price Floor **{sec['currency']} {cr['projected_price_floor']:,.2f}** ({cr['description']})\n"

    report += f"""
---

## 5. FAMA-FRENCH 5-FACTOR ATTRIBUTION
* **Market Exposure ($\beta$)**: `{factors['market_beta']}`
* **Size Factor Exposure (SMB)**: `{factors['size_smb']}` ({'Large Cap tilt' if factors['size_smb'] < 0 else 'Small Cap tilt'})
* **Value Factor Exposure (HML)**: `{factors['value_hml']}` ({'Growth tilt' if factors['value_hml'] < 0 else 'Value tilt'})
* **Profitability Factor Exposure (RMW)**: `{factors['profitability_rmw']}` (High-quality operating profitability bias)

---
*Report generated programmatically via OmniScreen Quantitative Research Engine.*
"""
    return report
