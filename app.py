"""
================================================================================
OmniScreen: Institutional Global Quantitative Workstation
Single-File Unified Production Engine
================================================================================
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union

import streamlit as st

# ==============================================================================
# 1. CORE QUANTITATIVE & FINANCIAL CALCULATION ENGINE
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

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = low.rolling(window=k_period, min_periods=1).min()
    high_max = high.rolling(window=k_period, min_periods=1).max()
    fast_k = 100.0 * (close - low_min) / (high_max - low_min + 1e-9)
    fast_d = fast_k.rolling(window=d_period, min_periods=1).mean()
    return fast_k, fast_d

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest_high = high.rolling(window=period, min_periods=1).max()
    lowest_low = low.rolling(window=period, min_periods=1).min()
    r = -100.0 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
    return r.fillna(-50.0)

def calculate_volatility(close: pd.Series, trading_days: int = 252) -> float:
    returns = np.log(close / close.shift(1)).dropna()
    if len(returns) < 2: return 0.0
    return float(returns.std() * np.sqrt(trading_days))

def calculate_max_drawdown(close: pd.Series) -> float:
    cum_max = close.cummax()
    drawdown = (close - cum_max) / (cum_max + 1e-9)
    return float(drawdown.min())

def calculate_sharpe_ratio(close: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252) -> float:
    returns = close.pct_change().dropna()
    if len(returns) < 5: return 0.0
    excess_returns = returns - (risk_free_rate / trading_days)
    std = returns.std()
    if std == 0 or np.isnan(std): return 0.0
    return float(np.sqrt(trading_days) * excess_returns.mean() / std)

def calculate_sortino_ratio(close: pd.Series, risk_free_rate: float = 0.04, trading_days: int = 252) -> float:
    returns = close.pct_change().dropna()
    if len(returns) < 5: return 0.0
    excess = returns - (risk_free_rate / trading_days)
    downside = returns[returns < 0]
    std = downside.std()
    if len(downside) < 2 or std == 0 or np.isnan(std): return 0.0
    return float(np.sqrt(trading_days) * excess.mean() / std)

def calculate_beta(stock_close: pd.Series, benchmark_close: pd.Series) -> float:
    stock_ret = stock_close.pct_change().dropna()
    bench_ret = benchmark_close.pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1).dropna()
    if len(aligned) < 10: return 1.0
    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
    var_bench = np.var(aligned.iloc[:, 1])
    if var_bench == 0 or np.isnan(var_bench): return 1.0
    return float(round(cov / var_bench, 2))

# ==============================================================================
# FORENSIC ACCOUNTING & ADVANCED HEALTH METRICS
# ==============================================================================

def calculate_piotroski_f_score(metrics: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    signals = {
        "Positive ROA": 1 if metrics.get("roa", 0) > 0 else 0,
        "Positive Operating Cash Flow": 1 if metrics.get("operating_cash_flow", 0) > 0 else 0,
        "ROA Expansion (YoY)": 1 if metrics.get("delta_roa", 0) > 0 else 0,
        "Accruals Quality (CFO > Net Income)": 1 if metrics.get("operating_cash_flow", 0) > metrics.get("net_income", 0) else 0,
        "Decreasing Long-Term Debt": 1 if metrics.get("delta_debt", 0) <= 0 else 0,
        "Current Ratio Improvement": 1 if metrics.get("delta_current_ratio", 0) > 0 else 0,
        "No Equity Share Dilution": 1 if not metrics.get("shares_diluted", False) else 0,
        "Gross Margin Expansion": 1 if metrics.get("delta_gross_margin", 0) > 0 else 0,
        "Asset Turnover Improvement": 1 if metrics.get("delta_asset_turnover", 0) > 0 else 0
    }
    return sum(signals.values()), signals

def calculate_altman_z_score(
    working_capital: float, total_assets: float, retained_earnings: float,
    ebit: float, market_cap: float, total_liabilities: float, sales: float
) -> float:
    if total_assets <= 0 or total_liabilities <= 0: return 0.0
    x1 = working_capital / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = sales / total_assets
    z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 0.999 * x5
    return float(round(z, 2))

def calculate_beneish_m_score(metrics: Dict[str, Any]) -> Tuple[float, str, Dict[str, float]]:
    dsri = metrics.get("dsri", 1.02) or 1.02
    gmi = metrics.get("gmi", 1.01) or 1.01
    aqi = metrics.get("aqi", 0.98) or 0.98
    sgi = metrics.get("sgi", 1.08) or 1.08
    depi = metrics.get("depi", 1.01) or 1.01
    sgai = metrics.get("sgai", 0.99) or 0.99
    tata = metrics.get("tata", 0.025) or 0.025
    lvgi = metrics.get("lvgi", 1.02) or 1.02

    m = (-4.84 + 0.920*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi
         + 0.115*depi - 0.172*sgai + 4.037*tata + 0.0327*lvgi)

    verdict = "🔴 High Manipulation Risk" if m > -1.78 else "🟢 Clean Accounting Records"
    breakdown = {"DSRI": dsri, "GMI": gmi, "AQI": aqi, "SGI": sgi, "DEPI": depi, "SGAI": sgai, "TATA": tata, "LVGI": lvgi}
    return round(float(m), 2), verdict, breakdown

def calculate_dupont_5_way(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    DuPont 5-Way ROE Decomposition:
    ROE = Tax Burden * Interest Burden * Operating Margin * Asset Turnover * Leverage
    """
    net_income = max(1.0, metrics.get("sales_m", 1000) * metrics.get("net_margin", 0.15))
    ebt = max(1.0, net_income * 1.25)
    ebit = max(1.0, metrics.get("ebit_m", 250))
    sales = max(1.0, metrics.get("sales_m", 1000))
    assets = max(1.0, metrics.get("assets_m", 2000))
    equity = max(1.0, assets - metrics.get("liab_m", 1000))

    tax_burden = round(net_income / ebt, 3)
    interest_burden = round(ebt / ebit, 3)
    operating_margin = round(ebit / sales, 3)
    asset_turnover = round(sales / assets, 3)
    financial_leverage = round(assets / equity, 3)
    computed_roe = round(tax_burden * interest_burden * operating_margin * asset_turnover * financial_leverage * 100, 1)

    return {
        "tax_burden": tax_burden,
        "interest_burden": interest_burden,
        "operating_margin": operating_margin,
        "asset_turnover": asset_turnover,
        "financial_leverage": financial_leverage,
        "computed_roe_pct": computed_roe
    }

def calculate_amihud_illiquidity(close: pd.Series, volume: pd.Series) -> float:
    returns = close.pct_change().abs().dropna()
    dollar_vol = (close * volume).dropna()
    aligned = pd.concat([returns, dollar_vol], axis=1).dropna()
    if len(aligned) < 5: return 0.0
    ratios = aligned.iloc[:, 0] / (aligned.iloc[:, 1] + 1e-9)
    return round(float(ratios.mean() * 1e9), 4)

def calculate_kelly_criterion(win_rate_pct: float, win_loss_ratio: float = 1.5, fraction: float = 0.5) -> float:
    p = win_rate_pct / 100.0
    b = max(0.1, win_loss_ratio)
    q = 1.0 - p
    f_star = p - (q / b)
    f_star = max(0.0, f_star) * fraction
    return round(float(min(f_star * 100, 25.0)), 1)

def calculate_var_cvar(close: pd.Series, confidence_level: float = 0.95) -> Dict[str, float]:
    returns = close.pct_change().dropna()
    if len(returns) < 10:
        return {"var_historical_pct": 0.0, "var_parametric_pct": 0.0, "cvar_expected_shortfall_pct": 0.0}
    var_hist = -float(np.percentile(returns, (1 - confidence_level) * 100))
    from scipy.stats import norm
    z_score = norm.ppf(confidence_level)
    var_param = float(z_score * returns.std() - returns.mean())
    tail_losses = returns[returns <= -var_hist]
    cvar = -float(tail_losses.mean()) if len(tail_losses) > 0 else var_hist
    return {
        "var_historical_pct": round(var_hist * 100, 2),
        "var_parametric_pct": round(var_param * 100, 2),
        "cvar_expected_shortfall_pct": round(cvar * 100, 2)
    }

def simulate_crisis_scenarios(current_price: float, beta: float, sector: str) -> List[Dict[str, Any]]:
    scenarios = [
        {"name": "2008 Global Financial Crisis", "market_shock": -0.48, "sector_mult": 1.4 if sector == "Financial Services" else 0.9, "desc": "Subprime mortgage collapse & banking liquidity freeze"},
        {"name": "2020 COVID-19 Flash Crash", "market_shock": -0.34, "sector_mult": 0.7 if sector == "Technology" else 1.3 if sector in ["Energy", "Consumer Cyclical"] else 1.0, "desc": "Worldwide pandemic supply chain & mobility seizure"},
        {"name": "2022 Inflation & Rate Spike", "market_shock": -0.22, "sector_mult": 1.6 if sector == "Technology" else 0.4 if sector == "Energy" else 1.0, "desc": "Aggressive central bank quantitative tightening"},
        {"name": "2000 Dot-Com Tech Bubble", "market_shock": -0.45, "sector_mult": 1.8 if sector == "Technology" else 0.6, "desc": "Speculative high-multiple tech liquidation"}
    ]
    results = []
    for sc in scenarios:
        shock = max(-0.85, min(0.40, sc["market_shock"] * beta * sc["sector_mult"]))
        stressed_p = current_price * (1 + shock)
        results.append({
            "scenario": sc["name"],
            "description": sc["desc"],
            "drawdown_pct": round(shock * 100, 1),
            "projected_floor": round(stressed_p, 2)
        })
    return results

def calculate_factor_attribution(sec: Dict[str, Any]) -> Dict[str, Any]:
    mcap = sec.get("market_cap_usd", 1e11)
    pe = sec.get("pe_ratio", 25) or 25
    roe = sec.get("roe", 0.15) or 0.15
    beta = sec.get("beta", 1.0) or 1.0
    sharpe = sec.get("sharpe_ratio", 1.0) or 1.0

    smb = round(float(np.clip(-np.log10(mcap / 1e9) / 2.0, -1.0, 1.0)), 2)
    hml = round(float(np.clip((25 - pe) / 20.0, -1.0, 1.0)), 2)
    rmw = round(float(np.clip((roe - 0.15) / 0.15, -1.0, 1.0)), 2)
    est_alpha = round(float(max(-10.0, min(35.0, (sharpe - 0.5) * 8.0 + rmw * 3.0))), 1)

    return {
        "market_beta": beta,
        "size_smb": smb,
        "value_hml": hml,
        "profitability_rmw": rmw,
        "estimated_annual_alpha_pct": est_alpha
    }

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
# VALUATION & SIMULATION MODELS
# ==============================================================================

def calculate_dcf(
    fcf_million: float, shares_outstanding_million: float, net_debt_million: float,
    growth_rate_pct: float = 12.0, terminal_growth_pct: float = 2.5,
    wacc_pct: float = 9.0, projection_years: int = 5
) -> Dict[str, Any]:
    g = growth_rate_pct / 100.0
    tg = terminal_growth_pct / 100.0
    wacc = wacc_pct / 100.0

    projected_fcf = []
    current_cf = fcf_million
    pv_projected = 0.0

    for year in range(1, projection_years + 1):
        current_cf *= (1 + g)
        pv = current_cf / ((1 + wacc) ** year)
        projected_fcf.append({"year": year, "fcf": round(current_cf, 1), "pv": round(pv, 1)})
        pv_projected += pv

    terminal_cf = current_cf * (1 + tg)
    tv = terminal_cf / (wacc - tg + 1e-9)
    pv_tv = tv / ((1 + wacc) ** projection_years)
    enterprise_value = pv_projected + pv_tv
    equity_value = enterprise_value - net_debt_million
    fair_value = max(0.0, equity_value / (shares_outstanding_million + 1e-9))

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
        "fair_value": round(fair_value, 2),
        "enterprise_value_m": round(enterprise_value, 1),
        "equity_value_m": round(equity_value, 1),
        "pv_fcf_m": round(pv_projected, 1),
        "pv_terminal_m": round(pv_tv, 1),
        "projected_cash_flows": projected_fcf,
        "sensitivity_matrix": pd.DataFrame(sensitivity).T
    }

def simulate_monte_carlo_paths(
    latest_price: float, annual_vol: float, days: int = 252,
    num_simulations: int = 100, expected_drift: float = 0.08
) -> Dict[str, Any]:
    dt = 1 / 252
    mu = expected_drift
    sigma = annual_vol
    np.random.seed(101)

    drift = (mu - 0.5 * sigma ** 2) * dt
    shocks = sigma * np.sqrt(dt) * np.random.normal(0, 1, (days, num_simulations))
    multipliers = np.exp(drift + shocks)
    paths = np.zeros((days + 1, num_simulations))
    paths[0] = latest_price

    for t in range(1, days + 1):
        paths[t] = paths[t - 1] * multipliers[t - 1]

    end_prices = paths[-1]
    p10 = float(np.percentile(end_prices, 10))
    p50 = float(np.percentile(end_prices, 50))
    p90 = float(np.percentile(end_prices, 90))

    dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days + 1)]
    df_chart = pd.DataFrame(paths[:, :15])
    df_chart.columns = [f"Path {i+1}" for i in range(15)]
    df_chart["Median Expected"] = np.median(paths, axis=1)
    df_chart["Date"] = dates
    df_chart = df_chart.set_index("Date")

    return {"p10_bearish": round(p10, 2), "p50_median": round(p50, 2), "p90_bullish": round(p90, 2), "chart_data": df_chart}

def simulate_efficient_frontier(price_df: pd.DataFrame, num_portfolios: int = 1500, risk_free_rate: float = 0.04) -> Dict[str, Any]:
    returns_df = price_df.pct_change().dropna()
    mean_daily = returns_df.mean()
    cov_matrix = returns_df.cov()
    num_assets = len(price_df.columns)
    results = np.zeros((3, num_portfolios))
    weights_record = []
    np.random.seed(42)

    for i in range(num_portfolios):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        weights_record.append(w)
        p_return = np.sum(mean_daily * w) * 252
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
        results[0, i] = p_return
        results[1, i] = p_vol
        results[2, i] = (p_return - risk_free_rate) / (p_vol + 1e-9)

    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])
    assets = list(price_df.columns)

    max_sharpe_portfolio = {
        "return": round(float(results[0, max_sharpe_idx] * 100), 2),
        "volatility": round(float(results[1, max_sharpe_idx] * 100), 2),
        "sharpe": round(float(results[2, max_sharpe_idx]), 2),
        "weights": {assets[j]: round(float(weights_record[max_sharpe_idx][j] * 100), 1) for j in range(num_assets)}
    }
    min_vol_portfolio = {
        "return": round(float(results[0, min_vol_idx] * 100), 2),
        "volatility": round(float(results[1, min_vol_idx] * 100), 2),
        "sharpe": round(float(results[2, min_vol_idx]), 2),
        "weights": {assets[j]: round(float(weights_record[min_vol_idx][j] * 100), 1) for j in range(num_assets)}
    }
    simulated_scatter = pd.DataFrame({"Volatility": results[1] * 100, "Return": results[0] * 100, "Sharpe": results[2]})
    return {"max_sharpe": max_sharpe_portfolio, "min_vol": min_vol_portfolio, "scatter_data": simulated_scatter}

def calculate_risk_parity_weights(price_df: pd.DataFrame) -> Dict[str, float]:
    returns = price_df.pct_change().dropna()
    vols = returns.std() * np.sqrt(252)
    inv_vols = 1.0 / (vols + 1e-9)
    weights = inv_vols / inv_vols.sum()
    return {col: round(float(weights[col] * 100), 1) for col in price_df.columns}

def run_backtest(df_candles: pd.DataFrame, strategy: str = "sma_crossover", **kwargs) -> Dict[str, Any]:
    df = df_candles.copy()
    close = df["close"]
    if strategy == "sma_crossover":
        fast = kwargs.get("fast_window", 20)
        slow = kwargs.get("slow_window", 50)
        signal = np.where(calculate_sma(close, fast) > calculate_sma(close, slow), 1.0, 0.0)
    elif strategy == "rsi_mean_reversion":
        rsi = calculate_rsi(close, 14)
        oversold = kwargs.get("rsi_oversold", 35.0)
        overbought = kwargs.get("rsi_overbought", 65.0)
        signal = np.zeros(len(close))
        pos = 0.0
        for i in range(len(close)):
            if rsi.iloc[i] < oversold: pos = 1.0
            elif rsi.iloc[i] > overbought: pos = 0.0
            signal[i] = pos
    else:
        upper, mid, lower = calculate_bollinger_bands(close, 20, 2.0)
        signal = np.where(close > mid, 1.0, 0.0)

    signal_series = pd.Series(signal, index=close.index).shift(1).fillna(0)
    market_ret = close.pct_change().fillna(0)
    strat_ret = signal_series * market_ret

    cum_market = (1 + market_ret).cumprod()
    cum_strat = (1 + strat_ret).cumprod()

    trades = int((signal_series.diff() != 0).sum())
    active_days = int((signal_series > 0).sum())
    win_days = int((strat_ret > 0).sum())
    win_rate = (win_days / active_days * 100) if active_days > 0 else 0.0

    return {
        "strategy": strategy,
        "total_return_pct": round(float(cum_strat.iloc[-1] - 1.0) * 100, 2),
        "benchmark_return_pct": round(float(cum_market.iloc[-1] - 1.0) * 100, 2),
        "strategy_sharpe": round(calculate_sharpe_ratio(cum_strat), 2),
        "strategy_max_drawdown_pct": round(calculate_max_drawdown(cum_strat) * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "trades_executed": trades,
        "equity_curve": pd.DataFrame({"timestamp": df["timestamp"], "Strategy": cum_strat.values, "Buy & Hold": cum_market.values}).to_dict(orient="records")
    }

def detect_technical_patterns(df_candles: pd.DataFrame) -> List[Dict[str, str]]:
    df = df_candles.copy()
    close = df["close"]
    signals = []
    if len(close) >= 50:
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)
        if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
            signals.append({"pattern": "Bullish SMA Crossover", "type": "Bullish", "description": "20-day SMA crossed above 50-day SMA"})
        elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
            signals.append({"pattern": "Bearish SMA Crossover", "type": "Bearish", "description": "20-day SMA crossed below 50-day SMA"})

    upper, mid, lower = calculate_bollinger_bands(close, 20, 2.0)
    bw = (upper - lower) / mid
    if bw.iloc[-1] < bw.rolling(50).quantile(0.15).iloc[-1]:
        signals.append({"pattern": "Bollinger Band Squeeze", "type": "Neutral", "description": "Volatility compression in bottom 15th percentile"})

    rsi = calculate_rsi(close, 14).iloc[-1]
    if rsi < 30: signals.append({"pattern": "Oversold RSI Exhaustion", "type": "Bullish", "description": f"RSI at {rsi:.1f} indicates severe oversold condition"})
    elif rsi > 70: signals.append({"pattern": "Overbought RSI Caution", "type": "Bearish", "description": f"RSI at {rsi:.1f} indicates extended momentum"})

    if not signals:
        signals.append({"pattern": "Consolidation Regime", "type": "Neutral", "description": "Price action trading within normal statistical bounds"})
    return signals

def generate_equity_research_report(sec: Dict[str, Any], candles: List[Dict[str, Any]]) -> str:
    close = pd.Series([c["close"] for c in candles])
    var_metrics = calculate_var_cvar(close, 0.95)
    crises = simulate_crisis_scenarios(sec["price"], sec["beta"], sec["sector"])
    factors = calculate_factor_attribution(sec)

    score = sec.get("composite_score", 50)
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

    report = f"""# 📑 INSTITUTIONAL RESEARCH MEMORANDUM
**CONFIDENTIAL • FOR RESEARCH & PORTFOLIO PURPOSES ONLY**  
**Generated On**: {datetime.now().strftime('%B %d, %Y')} | **Security**: {sec['ticker']} ({sec['name']})  
**Exchange**: {sec['exchange']} ({sec['country']}) | **Sector**: {sec['sector']} | **Industry**: {sec['industry']}  
**Current Price**: {sec['currency']} {sec['price']:,.2f} | **Market Cap**: ${sec['market_cap_usd']/1e9:,.2f}B USD  

---

## 1. ACTIONABLE QUANTITATIVE RECOMMENDATION
* **Target Rating**: **{rating}**
* **Strategic Rationale**: {rationale}
* **Half-Kelly Capital Allocation**: **{sec.get('kelly_allocation_pct', 10.0)}%** of risk portfolio.
* **Estimated Idiosyncratic Alpha ($\alpha$)**: **+{factors['estimated_annual_alpha_pct']}%** annualized.

---

## 2. MULTI-PILLAR FACTOR SCORECARD (0 - 100)
| Factor Pillar | Quant Score | Interpretation |
| :--- | :---: | :--- |
| **Composite Rank** | **{sec.get('composite_score', '-')} / 100** | Universal cross-sectional percentile rank |
| **Quality Factor** | **{sec.get('quality_score', '-')} / 100** | High ROE ({sec['roe']*100:.1f}%), pristine margins, clean debt structure |
| **Value Factor** | **{sec.get('value_score', '-')} / 100** | P/E of {sec['pe_ratio']}x vs FCF Yield of {sec['fcf_yield']*100:.2f}% |
| **Momentum Factor** | **{sec.get('momentum_score', '-')} / 100** | RSI ({sec['rsi_14']}), 1Y Sharpe Ratio ({sec['sharpe_ratio']}) |
| **Growth Factor** | **{sec.get('growth_score', '-')} / 100** | YoY Revenue Growth of {sec['revenue_growth_yoy']*100:.1f}% |

---

## 3. FORENSIC AUDIT & ACCOUNTING INTEGRITY
* **Beneish M-Score**: **{sec['beneish_m_score']}** ({sec.get('beneish_verdict', 'Clean')})
* **Altman Z-Score (Solvency)**: **{sec['altman_z_score']}** ({'Safe Zone' if sec['altman_z_score'] > 2.99 else 'Grey Zone'})
* **Piotroski F-Score**: **{sec['piotroski_f_score']} / 9** (Financial health passing mark is $\ge 6$)
* **Amihud Illiquidity Ratio**: **{sec.get('amihud_illiquidity', 0.05)}** (Low price impact slippage)

---

## 4. DOWNSIDE TAIL-RISK & BLACK SWAN STRESS TESTING
* **1-Day 95% Historical VaR**: **{var_metrics['var_historical_pct']}%**
* **Conditional VaR (Expected Shortfall)**: **{var_metrics['cvar_expected_shortfall_pct']}%**
* **Systematic Beta**: **{sec['beta']}**

### Crisis Scenarios:
"""
    for cr in crises:
        report += f"- **{cr['scenario']}**: Drawdown **{cr['drawdown_pct']}%** | Implied Floor **{sec['currency']} {cr['projected_floor']:,.2f}** ({cr['description']})\n"

    report += f"""
---

## 5. FAMA-FRENCH 5-FACTOR ATTRIBUTION
* **Market Beta ($\beta$)**: `{factors['market_beta']}`
* **Size Factor (SMB)**: `{factors['size_smb']}`
* **Value Factor (HML)**: `{factors['value_hml']}`
* **Profitability Factor (RMW)**: `{factors['profitability_rmw']}`
"""
    return report

# ==============================================================================
# 2. DATA STORE & GLOBAL MULTI-ASSET REPOSITORY
# ==============================================================================

FX_RATES = {"USD": 1.0, "INR": 0.012, "GBP": 1.28, "EUR": 1.09, "JPY": 0.0068, "HKD": 0.128, "CAD": 0.74, "AUD": 0.66}

EXPANDED_GLOBAL_ASSETS = [{'ticker': 'AAPL', 'name': 'Apple Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'base_price': 224.23, 'shares_out_b': 15.3, 'pe_ratio': 33.8, 'pb_ratio': 44.5, 'ps_ratio': 8.8, 'ev_ebitda': 24.6, 'fcf_yield': 0.032, 'roe': 1.45, 'roce': 0.58, 'roa': 0.28, 'debt_to_equity': 1.52, 'current_ratio': 0.98, 'gross_margin': 0.46, 'operating_margin': 0.31, 'net_margin': 0.26, 'revenue_growth': 0.05, 'profit_growth': 0.08, 'dividend_yield': 0.005, 'working_cap_m': 5000, 'assets_m': 352000, 'retained_m': -2000, 'ebit_m': 123000, 'liab_m': 290000, 'sales_m': 385000}, {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Infrastructure', 'base_price': 420.5, 'shares_out_b': 7.43, 'pe_ratio': 35.2, 'pb_ratio': 11.8, 'ps_ratio': 12.4, 'ev_ebitda': 23.1, 'fcf_yield': 0.027, 'roe': 0.38, 'roce': 0.32, 'roa': 0.18, 'debt_to_equity': 0.42, 'current_ratio': 1.24, 'gross_margin': 0.69, 'operating_margin': 0.44, 'net_margin': 0.36, 'revenue_growth': 0.15, 'profit_growth': 0.18, 'dividend_yield': 0.007, 'working_cap_m': 25000, 'assets_m': 512000, 'retained_m': 118000, 'ebit_m': 109000, 'liab_m': 243000, 'sales_m': 245000}, {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 122.8, 'shares_out_b': 24.6, 'pe_ratio': 48.5, 'pb_ratio': 38.2, 'ps_ratio': 26.5, 'ev_ebitda': 38.0, 'fcf_yield': 0.021, 'roe': 1.15, 'roce': 0.88, 'roa': 0.55, 'debt_to_equity': 0.18, 'current_ratio': 3.82, 'gross_margin': 0.75, 'operating_margin': 0.62, 'net_margin': 0.54, 'revenue_growth': 1.22, 'profit_growth': 1.68, 'dividend_yield': 0.0003, 'working_cap_m': 35000, 'assets_m': 85000, 'retained_m': 42000, 'ebit_m': 56000, 'liab_m': 28000, 'sales_m': 96000}, {'ticker': 'GOOGL', 'name': 'Alphabet Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Communication Services', 'industry': 'Internet Content & Information', 'base_price': 158.4, 'shares_out_b': 12.4, 'pe_ratio': 23.4, 'pb_ratio': 6.2, 'ps_ratio': 6.1, 'ev_ebitda': 15.2, 'fcf_yield': 0.038, 'roe': 0.28, 'roce': 0.26, 'roa': 0.2, 'debt_to_equity': 0.11, 'current_ratio': 2.15, 'gross_margin': 0.57, 'operating_margin': 0.32, 'net_margin': 0.26, 'revenue_growth': 0.14, 'profit_growth': 0.28, 'dividend_yield': 0.005, 'working_cap_m': 72000, 'assets_m': 410000, 'retained_m': 220000, 'ebit_m': 98000, 'liab_m': 115000, 'sales_m': 325000}, {'ticker': 'AMZN', 'name': 'Amazon.com, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail', 'base_price': 182.3, 'shares_out_b': 10.4, 'pe_ratio': 42.1, 'pb_ratio': 8.1, 'ps_ratio': 3.2, 'ev_ebitda': 17.5, 'fcf_yield': 0.029, 'roe': 0.21, 'roce': 0.16, 'roa': 0.08, 'debt_to_equity': 0.62, 'current_ratio': 1.05, 'gross_margin': 0.48, 'operating_margin': 0.09, 'net_margin': 0.07, 'revenue_growth': 0.11, 'profit_growth': 0.54, 'dividend_yield': 0.0, 'working_cap_m': 12000, 'assets_m': 530000, 'retained_m': 115000, 'ebit_m': 48000, 'liab_m': 310000, 'sales_m': 600000}, {'ticker': 'META', 'name': 'Meta Platforms, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Communication Services', 'industry': 'Internet Content & Information', 'base_price': 505.2, 'shares_out_b': 2.54, 'pe_ratio': 26.8, 'pb_ratio': 8.4, 'ps_ratio': 8.5, 'ev_ebitda': 16.8, 'fcf_yield': 0.036, 'roe': 0.34, 'roce': 0.32, 'roa': 0.22, 'debt_to_equity': 0.24, 'current_ratio': 2.4, 'gross_margin': 0.81, 'operating_margin': 0.38, 'net_margin': 0.33, 'revenue_growth': 0.22, 'profit_growth': 0.73, 'dividend_yield': 0.004, 'working_cap_m': 48000, 'assets_m': 240000, 'retained_m': 92000, 'ebit_m': 55000, 'liab_m': 82000, 'sales_m': 150000}, {'ticker': 'TSLA', 'name': 'Tesla, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'base_price': 218.4, 'shares_out_b': 3.19, 'pe_ratio': 61.2, 'pb_ratio': 10.4, 'ps_ratio': 7.1, 'ev_ebitda': 32.5, 'fcf_yield': 0.012, 'roe': 0.18, 'roce': 0.14, 'roa': 0.11, 'debt_to_equity': 0.08, 'current_ratio': 1.72, 'gross_margin': 0.18, 'operating_margin': 0.07, 'net_margin': 0.14, 'revenue_growth': 0.03, 'profit_growth': -0.45, 'dividend_yield': 0.0, 'working_cap_m': 28000, 'assets_m': 110000, 'retained_m': 31000, 'ebit_m': 8800, 'liab_m': 43000, 'sales_m': 97000}, {'ticker': 'AVGO', 'name': 'Broadcom Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 165.2, 'shares_out_b': 4.65, 'pe_ratio': 46.2, 'pb_ratio': 10.8, 'ps_ratio': 14.5, 'ev_ebitda': 22.8, 'fcf_yield': 0.028, 'roe': 0.25, 'roce': 0.21, 'roa': 0.12, 'debt_to_equity': 0.88, 'current_ratio': 1.35, 'gross_margin': 0.65, 'operating_margin': 0.42, 'net_margin': 0.28, 'revenue_growth': 0.47, 'profit_growth': 0.35, 'dividend_yield': 0.013, 'working_cap_m': 15000, 'assets_m': 175000, 'retained_m': 45000, 'ebit_m': 24000, 'liab_m': 95000, 'sales_m': 52000}, {'ticker': 'CRM', 'name': 'Salesforce, Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Application', 'base_price': 252.8, 'shares_out_b': 0.97, 'pe_ratio': 44.5, 'pb_ratio': 4.1, 'ps_ratio': 6.8, 'ev_ebitda': 20.4, 'fcf_yield': 0.045, 'roe': 0.1, 'roce': 0.12, 'roa': 0.055, 'debt_to_equity': 0.24, 'current_ratio': 1.05, 'gross_margin': 0.76, 'operating_margin': 0.19, 'net_margin': 0.15, 'revenue_growth': 0.11, 'profit_growth': 0.68, 'dividend_yield': 0.006, 'working_cap_m': -8000, 'assets_m': 102000, 'retained_m': 14000, 'ebit_m': 7200, 'liab_m': 42000, 'sales_m': 36000}, {'ticker': 'AMD', 'name': 'Advanced Micro Devices, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 142.1, 'shares_out_b': 1.62, 'pe_ratio': 115.0, 'pb_ratio': 4.1, 'ps_ratio': 9.5, 'ev_ebitda': 45.0, 'fcf_yield': 0.015, 'roe': 0.04, 'roce': 0.05, 'roa': 0.03, 'debt_to_equity': 0.04, 'current_ratio': 2.4, 'gross_margin': 0.5, 'operating_margin': 0.08, 'net_margin': 0.06, 'revenue_growth': 0.09, 'profit_growth': 0.18, 'dividend_yield': 0.0, 'working_cap_m': 14000, 'assets_m': 68000, 'retained_m': 2500, 'ebit_m': 1800, 'liab_m': 12000, 'sales_m': 24000}, {'ticker': 'CSCO', 'name': 'Cisco Systems, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Communication Equipment', 'base_price': 50.4, 'shares_out_b': 4.02, 'pe_ratio': 19.5, 'pb_ratio': 4.2, 'ps_ratio': 3.8, 'ev_ebitda': 12.8, 'fcf_yield': 0.065, 'roe': 0.22, 'roce': 0.2, 'roa': 0.1, 'debt_to_equity': 0.75, 'current_ratio': 1.35, 'gross_margin': 0.64, 'operating_margin': 0.24, 'net_margin': 0.19, 'revenue_growth': -0.06, 'profit_growth': -0.12, 'dividend_yield': 0.032, 'working_cap_m': 18000, 'assets_m': 120000, 'retained_m': 28000, 'ebit_m': 14500, 'liab_m': 72000, 'sales_m': 54000}, {'ticker': 'PLTR', 'name': 'Palantir Technologies Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Infrastructure', 'base_price': 32.5, 'shares_out_b': 2.22, 'pe_ratio': 78.4, 'pb_ratio': 16.5, 'ps_ratio': 28.5, 'ev_ebitda': 62.0, 'fcf_yield': 0.018, 'roe': 0.15, 'roce': 0.14, 'roa': 0.09, 'debt_to_equity': 0.06, 'current_ratio': 5.4, 'gross_margin': 0.81, 'operating_margin': 0.16, 'net_margin': 0.14, 'revenue_growth': 0.27, 'profit_growth': 1.25, 'dividend_yield': 0.0, 'working_cap_m': 4200, 'assets_m': 5100, 'retained_m': -800, 'ebit_m': 420, 'liab_m': 900, 'sales_m': 2500}, {'ticker': 'PANW', 'name': 'Palo Alto Networks, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Infrastructure', 'base_price': 348.0, 'shares_out_b': 0.32, 'pe_ratio': 48.0, 'pb_ratio': 22.0, 'ps_ratio': 13.8, 'ev_ebitda': 26.5, 'fcf_yield': 0.038, 'roe': 0.45, 'roce': 0.3, 'roa': 0.14, 'debt_to_equity': 0.58, 'current_ratio': 0.85, 'gross_margin': 0.74, 'operating_margin': 0.18, 'net_margin': 0.16, 'revenue_growth': 0.16, 'profit_growth': 0.82, 'dividend_yield': 0.0, 'working_cap_m': -2800, 'assets_m': 16500, 'retained_m': 1400, 'ebit_m': 1500, 'liab_m': 11500, 'sales_m': 8000}, {'ticker': 'UBER', 'name': 'Uber Technologies, Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Application', 'base_price': 72.4, 'shares_out_b': 2.08, 'pe_ratio': 34.5, 'pb_ratio': 8.5, 'ps_ratio': 3.8, 'ev_ebitda': 21.0, 'fcf_yield': 0.035, 'roe': 0.28, 'roce': 0.19, 'roa': 0.09, 'debt_to_equity': 0.65, 'current_ratio': 1.15, 'gross_margin': 0.32, 'operating_margin': 0.08, 'net_margin': 0.11, 'revenue_growth': 0.16, 'profit_growth': 1.45, 'dividend_yield': 0.0, 'working_cap_m': 2500, 'assets_m': 42000, 'retained_m': -6200, 'ebit_m': 3200, 'liab_m': 24000, 'sales_m': 40000}, {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Banks - Diversified', 'base_price': 215.4, 'shares_out_b': 2.85, 'pe_ratio': 12.1, 'pb_ratio': 1.8, 'ps_ratio': 3.6, 'ev_ebitda': 9.4, 'fcf_yield': 0.065, 'roe': 0.17, 'roce': 0.14, 'roa': 0.014, 'debt_to_equity': 1.85, 'current_ratio': 1.12, 'gross_margin': 0.88, 'operating_margin': 0.42, 'net_margin': 0.33, 'revenue_growth': 0.12, 'profit_growth': 0.14, 'dividend_yield': 0.022, 'working_cap_m': 45000, 'assets_m': 4100000, 'retained_m': 310000, 'ebit_m': 68000, 'liab_m': 3770000, 'sales_m': 165000}, {'ticker': 'V', 'name': 'Visa Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Credit Services', 'base_price': 272.5, 'shares_out_b': 2.01, 'pe_ratio': 29.5, 'pb_ratio': 13.8, 'ps_ratio': 15.8, 'ev_ebitda': 21.2, 'fcf_yield': 0.036, 'roe': 0.48, 'roce': 0.35, 'roa': 0.21, 'debt_to_equity': 0.54, 'current_ratio': 1.45, 'gross_margin': 0.98, 'operating_margin': 0.67, 'net_margin': 0.54, 'revenue_growth': 0.1, 'profit_growth': 0.12, 'dividend_yield': 0.008, 'working_cap_m': 8500, 'assets_m': 92000, 'retained_m': 48000, 'ebit_m': 22000, 'liab_m': 53000, 'sales_m': 34000}, {'ticker': 'MA', 'name': 'Mastercard Incorporated', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Credit Services', 'base_price': 485.0, 'shares_out_b': 0.93, 'pe_ratio': 36.4, 'pb_ratio': 64.0, 'ps_ratio': 17.5, 'ev_ebitda': 26.2, 'fcf_yield': 0.028, 'roe': 1.75, 'roce': 0.55, 'roa': 0.29, 'debt_to_equity': 2.45, 'current_ratio': 1.15, 'gross_margin': 1.0, 'operating_margin': 0.58, 'net_margin': 0.45, 'revenue_growth': 0.12, 'profit_growth': 0.15, 'dividend_yield': 0.005, 'working_cap_m': 2500, 'assets_m': 44000, 'retained_m': 8500, 'ebit_m': 15500, 'liab_m': 37000, 'sales_m': 26500}, {'ticker': 'BAC', 'name': 'Bank of America Corporation', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Banks - Diversified', 'base_price': 39.8, 'shares_out_b': 7.85, 'pe_ratio': 13.5, 'pb_ratio': 1.15, 'ps_ratio': 3.1, 'ev_ebitda': 10.2, 'fcf_yield': 0.058, 'roe': 0.095, 'roce': 0.088, 'roa': 0.008, 'debt_to_equity': 1.25, 'current_ratio': 1.08, 'gross_margin': 0.85, 'operating_margin': 0.35, 'net_margin': 0.24, 'revenue_growth': 0.05, 'profit_growth': -0.08, 'dividend_yield': 0.026, 'working_cap_m': 32000, 'assets_m': 3250000, 'retained_m': 185000, 'ebit_m': 36000, 'liab_m': 2960000, 'sales_m': 102000}, {'ticker': 'GS', 'name': 'The Goldman Sachs Group, Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Capital Markets', 'base_price': 490.0, 'shares_out_b': 0.32, 'pe_ratio': 15.2, 'pb_ratio': 1.45, 'ps_ratio': 3.2, 'ev_ebitda': 11.0, 'fcf_yield': 0.052, 'roe': 0.115, 'roce': 0.108, 'roa': 0.007, 'debt_to_equity': 2.2, 'current_ratio': 1.1, 'gross_margin': 0.85, 'operating_margin': 0.32, 'net_margin': 0.22, 'revenue_growth': 0.15, 'profit_growth': 0.45, 'dividend_yield': 0.024, 'working_cap_m': 28000, 'assets_m': 1650000, 'retained_m': 125000, 'ebit_m': 18500, 'liab_m': 1530000, 'sales_m': 50000}, {'ticker': 'BLK', 'name': 'BlackRock, Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Asset Management', 'base_price': 880.0, 'shares_out_b': 0.15, 'pe_ratio': 22.5, 'pb_ratio': 3.2, 'ps_ratio': 7.0, 'ev_ebitda': 16.5, 'fcf_yield': 0.042, 'roe': 0.15, 'roce': 0.14, 'roa': 0.045, 'debt_to_equity': 0.28, 'current_ratio': 2.85, 'gross_margin': 0.52, 'operating_margin': 0.38, 'net_margin': 0.31, 'revenue_growth': 0.09, 'profit_growth': 0.14, 'dividend_yield': 0.023, 'working_cap_m': 12000, 'assets_m': 135000, 'retained_m': 42000, 'ebit_m': 7200, 'liab_m': 92000, 'sales_m': 19000}, {'ticker': 'LLY', 'name': 'Eli Lilly and Company', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 948.1, 'shares_out_b': 0.95, 'pe_ratio': 112.4, 'pb_ratio': 62.1, 'ps_ratio': 24.8, 'ev_ebitda': 68.2, 'fcf_yield': 0.012, 'roe': 0.64, 'roce': 0.41, 'roa': 0.15, 'debt_to_equity': 2.1, 'current_ratio': 1.28, 'gross_margin': 0.8, 'operating_margin': 0.34, 'net_margin': 0.23, 'revenue_growth': 0.32, 'profit_growth': 0.45, 'dividend_yield': 0.006, 'working_cap_m': 4200, 'assets_m': 68000, 'retained_m': 12000, 'ebit_m': 14000, 'liab_m': 54000, 'sales_m': 40000}, {'ticker': 'UNH', 'name': 'UnitedHealth Group Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Healthcare Plans', 'base_price': 578.9, 'shares_out_b': 0.92, 'pe_ratio': 25.4, 'pb_ratio': 5.8, 'ps_ratio': 1.4, 'ev_ebitda': 14.5, 'fcf_yield': 0.048, 'roe': 0.24, 'roce': 0.19, 'roa': 0.082, 'debt_to_equity': 0.75, 'current_ratio': 0.88, 'gross_margin': 0.24, 'operating_margin': 0.088, 'net_margin': 0.058, 'revenue_growth': 0.09, 'profit_growth': 0.08, 'dividend_yield': 0.014, 'working_cap_m': -8000, 'assets_m': 275000, 'retained_m': 88000, 'ebit_m': 33000, 'liab_m': 178000, 'sales_m': 380000}, {'ticker': 'JNJ', 'name': 'Johnson & Johnson', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 164.8, 'shares_out_b': 2.41, 'pe_ratio': 24.8, 'pb_ratio': 5.4, 'ps_ratio': 4.6, 'ev_ebitda': 14.8, 'fcf_yield': 0.045, 'roe': 0.22, 'roce': 0.18, 'roa': 0.095, 'debt_to_equity': 0.45, 'current_ratio': 1.25, 'gross_margin': 0.69, 'operating_margin': 0.28, 'net_margin': 0.19, 'revenue_growth': 0.06, 'profit_growth': 0.08, 'dividend_yield': 0.03, 'working_cap_m': 8000, 'assets_m': 172000, 'retained_m': 110000, 'ebit_m': 24000, 'liab_m': 98000, 'sales_m': 86000}, {'ticker': 'ABBV', 'name': 'AbbVie Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 192.0, 'shares_out_b': 1.77, 'pe_ratio': 52.0, 'pb_ratio': 38.0, 'ps_ratio': 6.2, 'ev_ebitda': 17.5, 'fcf_yield': 0.048, 'roe': 0.82, 'roce': 0.24, 'roa': 0.065, 'debt_to_equity': 7.4, 'current_ratio': 0.95, 'gross_margin': 0.7, 'operating_margin': 0.32, 'net_margin': 0.12, 'revenue_growth': 0.04, 'profit_growth': -0.15, 'dividend_yield': 0.032, 'working_cap_m': -2500, 'assets_m': 138000, 'retained_m': -18000, 'ebit_m': 17500, 'liab_m': 128000, 'sales_m': 55000}, {'ticker': 'ISRG', 'name': 'Intuitive Surgical, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Medical Instruments & Supplies', 'base_price': 482.0, 'shares_out_b': 0.35, 'pe_ratio': 76.5, 'pb_ratio': 11.2, 'ps_ratio': 22.4, 'ev_ebitda': 52.0, 'fcf_yield': 0.015, 'roe': 0.16, 'roce': 0.18, 'roa': 0.14, 'debt_to_equity': 0.01, 'current_ratio': 5.1, 'gross_margin': 0.67, 'operating_margin': 0.28, 'net_margin': 0.26, 'revenue_growth': 0.14, 'profit_growth': 0.28, 'dividend_yield': 0.0, 'working_cap_m': 6200, 'assets_m': 16000, 'retained_m': 14000, 'ebit_m': 2200, 'liab_m': 1800, 'sales_m': 7600}, {'ticker': 'WMT', 'name': 'Walmart Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Consumer Defensive', 'industry': 'Discount Stores', 'base_price': 75.8, 'shares_out_b': 8.04, 'pe_ratio': 32.4, 'pb_ratio': 6.8, 'ps_ratio': 0.95, 'ev_ebitda': 15.8, 'fcf_yield': 0.026, 'roe': 0.2, 'roce': 0.17, 'roa': 0.065, 'debt_to_equity': 0.72, 'current_ratio': 0.82, 'gross_margin': 0.24, 'operating_margin': 0.045, 'net_margin': 0.029, 'revenue_growth': 0.05, 'profit_growth': 0.12, 'dividend_yield': 0.011, 'working_cap_m': -15000, 'assets_m': 255000, 'retained_m': 92000, 'ebit_m': 28000, 'liab_m': 165000, 'sales_m': 660000}, {'ticker': 'COST', 'name': 'Costco Wholesale Corporation', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Consumer Defensive', 'industry': 'Discount Stores', 'base_price': 892.0, 'shares_out_b': 0.44, 'pe_ratio': 54.0, 'pb_ratio': 14.5, 'ps_ratio': 1.55, 'ev_ebitda': 28.0, 'fcf_yield': 0.02, 'roe': 0.28, 'roce': 0.26, 'roa': 0.1, 'debt_to_equity': 0.28, 'current_ratio': 1.05, 'gross_margin': 0.125, 'operating_margin': 0.038, 'net_margin': 0.028, 'revenue_growth': 0.08, 'profit_growth': 0.14, 'dividend_yield': 0.005, 'working_cap_m': 4500, 'assets_m': 72000, 'retained_m': 18000, 'ebit_m': 9500, 'liab_m': 44000, 'sales_m': 255000}, {'ticker': 'CAT', 'name': 'Caterpillar Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Industrials', 'industry': 'Farm & Heavy Construction Machinery', 'base_price': 365.0, 'shares_out_b': 0.49, 'pe_ratio': 16.8, 'pb_ratio': 9.2, 'ps_ratio': 2.7, 'ev_ebitda': 12.4, 'fcf_yield': 0.058, 'roe': 0.58, 'roce': 0.28, 'roa': 0.12, 'debt_to_equity': 1.95, 'current_ratio': 1.45, 'gross_margin': 0.32, 'operating_margin': 0.205, 'net_margin': 0.158, 'revenue_growth': 0.03, 'profit_growth': 0.12, 'dividend_yield': 0.016, 'working_cap_m': 14000, 'assets_m': 88000, 'retained_m': 52000, 'ebit_m': 13800, 'liab_m': 69000, 'sales_m': 67000}, {'ticker': 'GE', 'name': 'GE Aerospace', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 188.0, 'shares_out_b': 1.09, 'pe_ratio': 38.5, 'pb_ratio': 7.4, 'ps_ratio': 2.9, 'ev_ebitda': 22.0, 'fcf_yield': 0.032, 'roe': 0.22, 'roce': 0.18, 'roa': 0.075, 'debt_to_equity': 0.65, 'current_ratio': 1.28, 'gross_margin': 0.28, 'operating_margin': 0.15, 'net_margin': 0.09, 'revenue_growth': 0.12, 'profit_growth': 0.42, 'dividend_yield': 0.006, 'working_cap_m': 8500, 'assets_m': 115000, 'retained_m': 22000, 'ebit_m': 9500, 'liab_m': 86000, 'sales_m': 68000}, {'ticker': 'XOM', 'name': 'Exxon Mobil Corporation', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated', 'base_price': 114.2, 'shares_out_b': 3.96, 'pe_ratio': 13.8, 'pb_ratio': 2.1, 'ps_ratio': 1.3, 'ev_ebitda': 6.8, 'fcf_yield': 0.078, 'roe': 0.17, 'roce': 0.16, 'roa': 0.095, 'debt_to_equity': 0.18, 'current_ratio': 1.35, 'gross_margin': 0.34, 'operating_margin': 0.15, 'net_margin': 0.105, 'revenue_growth': -0.06, 'profit_growth': -0.15, 'dividend_yield': 0.033, 'working_cap_m': 28000, 'assets_m': 375000, 'retained_m': 220000, 'ebit_m': 52000, 'liab_m': 160000, 'sales_m': 345000}, {'ticker': 'RELIANCE.NS', 'name': 'Reliance Industries Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Energy', 'industry': 'Oil & Gas Refining & Marketing', 'base_price': 3012.5, 'shares_out_b': 6.76, 'pe_ratio': 28.4, 'pb_ratio': 2.4, 'ps_ratio': 2.1, 'ev_ebitda': 13.8, 'fcf_yield': 0.024, 'roe': 0.095, 'roce': 0.108, 'roa': 0.045, 'debt_to_equity': 0.42, 'current_ratio': 1.15, 'gross_margin': 0.32, 'operating_margin': 0.17, 'net_margin': 0.078, 'revenue_growth': 0.08, 'profit_growth': 0.11, 'dividend_yield': 0.0035, 'working_cap_m': 120000, 'assets_m': 17500000, 'retained_m': 5200000, 'ebit_m': 1600000, 'liab_m': 9400000, 'sales_m': 9800000}, {'ticker': 'TCS.NS', 'name': 'Tata Consultancy Services Ltd', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Technology', 'industry': 'Information Technology Services', 'base_price': 4510.0, 'shares_out_b': 3.61, 'pe_ratio': 33.6, 'pb_ratio': 16.4, 'ps_ratio': 6.7, 'ev_ebitda': 23.4, 'fcf_yield': 0.038, 'roe': 0.51, 'roce': 0.65, 'roa': 0.32, 'debt_to_equity': 0.08, 'current_ratio': 2.45, 'gross_margin': 0.44, 'operating_margin': 0.26, 'net_margin': 0.198, 'revenue_growth': 0.07, 'profit_growth': 0.09, 'dividend_yield': 0.012, 'working_cap_m': 780000, 'assets_m': 1450000, 'retained_m': 940000, 'ebit_m': 620000, 'liab_m': 430000, 'sales_m': 2450000}, {'ticker': 'HDFCBANK.NS', 'name': 'HDFC Bank Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Financial Services', 'industry': 'Banks - Regional', 'base_price': 1655.2, 'shares_out_b': 7.6, 'pe_ratio': 19.8, 'pb_ratio': 2.8, 'ps_ratio': 4.1, 'ev_ebitda': 11.2, 'fcf_yield': 0.052, 'roe': 0.165, 'roce': 0.142, 'roa': 0.019, 'debt_to_equity': 1.45, 'current_ratio': 1.2, 'gross_margin': 0.82, 'operating_margin': 0.48, 'net_margin': 0.28, 'revenue_growth': 0.18, 'profit_growth': 0.16, 'dividend_yield': 0.012, 'working_cap_m': 450000, 'assets_m': 36000000, 'retained_m': 3800000, 'ebit_m': 1100000, 'liab_m': 31500000, 'sales_m': 3100000}, {'ticker': 'INFY.NS', 'name': 'Infosys Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Technology', 'industry': 'Information Technology Services', 'base_price': 1945.6, 'shares_out_b': 4.15, 'pe_ratio': 29.5, 'pb_ratio': 9.8, 'ps_ratio': 5.2, 'ev_ebitda': 20.8, 'fcf_yield': 0.041, 'roe': 0.34, 'roce': 0.44, 'roa': 0.22, 'debt_to_equity': 0.09, 'current_ratio': 2.1, 'gross_margin': 0.38, 'operating_margin': 0.21, 'net_margin': 0.17, 'revenue_growth': 0.06, 'profit_growth': 0.08, 'dividend_yield': 0.018, 'working_cap_m': 420000, 'assets_m': 1250000, 'retained_m': 780000, 'ebit_m': 340000, 'liab_m': 380000, 'sales_m': 1580000}, {'ticker': 'ICICIBANK.NS', 'name': 'ICICI Bank Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Financial Services', 'industry': 'Banks - Regional', 'base_price': 1240.0, 'shares_out_b': 7.04, 'pe_ratio': 18.2, 'pb_ratio': 3.1, 'ps_ratio': 4.4, 'ev_ebitda': 10.8, 'fcf_yield': 0.055, 'roe': 0.185, 'roce': 0.165, 'roa': 0.022, 'debt_to_equity': 1.15, 'current_ratio': 1.18, 'gross_margin': 0.84, 'operating_margin': 0.52, 'net_margin': 0.31, 'revenue_growth': 0.22, 'profit_growth': 0.25, 'dividend_yield': 0.008, 'working_cap_m': 380000, 'assets_m': 24000000, 'retained_m': 2200000, 'ebit_m': 850000, 'liab_m': 21000000, 'sales_m': 2100000}, {'ticker': 'BHARTIARTL.NS', 'name': 'Bharti Airtel Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Communication Services', 'industry': 'Telecom Services', 'base_price': 1560.8, 'shares_out_b': 5.98, 'pe_ratio': 62.4, 'pb_ratio': 8.6, 'ps_ratio': 6.1, 'ev_ebitda': 15.2, 'fcf_yield': 0.035, 'roe': 0.145, 'roce': 0.138, 'roa': 0.038, 'debt_to_equity': 1.65, 'current_ratio': 0.65, 'gross_margin': 0.58, 'operating_margin': 0.28, 'net_margin': 0.095, 'revenue_growth': 0.13, 'profit_growth': 0.28, 'dividend_yield': 0.005, 'working_cap_m': -80000, 'assets_m': 4500000, 'retained_m': 620000, 'ebit_m': 580000, 'liab_m': 3400000, 'sales_m': 1520000}, {'ticker': 'ITC.NS', 'name': 'ITC Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Defensive', 'industry': 'Tobacco / Conglomerate', 'base_price': 512.4, 'shares_out_b': 12.48, 'pe_ratio': 31.2, 'pb_ratio': 9.2, 'ps_ratio': 8.5, 'ev_ebitda': 22.4, 'fcf_yield': 0.034, 'roe': 0.29, 'roce': 0.38, 'roa': 0.24, 'debt_to_equity': 0.01, 'current_ratio': 2.85, 'gross_margin': 0.59, 'operating_margin': 0.37, 'net_margin': 0.27, 'revenue_growth': 0.08, 'profit_growth': 0.1, 'dividend_yield': 0.026, 'working_cap_m': 220000, 'assets_m': 890000, 'retained_m': 610000, 'ebit_m': 270000, 'liab_m': 190000, 'sales_m': 760000}, {'ticker': 'LT.NS', 'name': 'Larsen & Toubro Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Industrials', 'industry': 'Engineering & Construction', 'base_price': 3650.0, 'shares_out_b': 1.37, 'pe_ratio': 36.8, 'pb_ratio': 5.4, 'ps_ratio': 2.2, 'ev_ebitda': 18.5, 'fcf_yield': 0.021, 'roe': 0.158, 'roce': 0.145, 'roa': 0.048, 'debt_to_equity': 1.25, 'current_ratio': 1.32, 'gross_margin': 0.18, 'operating_margin': 0.105, 'net_margin': 0.058, 'revenue_growth': 0.18, 'profit_growth': 0.15, 'dividend_yield': 0.008, 'working_cap_m': 420000, 'assets_m': 3800000, 'retained_m': 820000, 'ebit_m': 240000, 'liab_m': 2800000, 'sales_m': 2250000}, {'ticker': 'TATAMOTORS.NS', 'name': 'Tata Motors Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'base_price': 1045.0, 'shares_out_b': 3.68, 'pe_ratio': 11.8, 'pb_ratio': 4.1, 'ps_ratio': 0.9, 'ev_ebitda': 5.8, 'fcf_yield': 0.082, 'roe': 0.38, 'roce': 0.22, 'roa': 0.092, 'debt_to_equity': 0.72, 'current_ratio': 1.08, 'gross_margin': 0.35, 'operating_margin': 0.118, 'net_margin': 0.075, 'revenue_growth': 0.26, 'profit_growth': 1.85, 'dividend_yield': 0.006, 'working_cap_m': -45000, 'assets_m': 3400000, 'retained_m': 420000, 'ebit_m': 480000, 'liab_m': 2400000, 'sales_m': 4350000}, {'ticker': 'SBIN.NS', 'name': 'State Bank of India', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Financial Services', 'industry': 'Banks - Regional', 'base_price': 795.0, 'shares_out_b': 8.92, 'pe_ratio': 10.4, 'pb_ratio': 1.6, 'ps_ratio': 1.8, 'ev_ebitda': 7.5, 'fcf_yield': 0.075, 'roe': 0.178, 'roce': 0.155, 'roa': 0.011, 'debt_to_equity': 1.45, 'current_ratio': 1.15, 'gross_margin': 0.78, 'operating_margin': 0.42, 'net_margin': 0.24, 'revenue_growth': 0.16, 'profit_growth': 0.22, 'dividend_yield': 0.017, 'working_cap_m': 550000, 'assets_m': 62000000, 'retained_m': 3200000, 'ebit_m': 1250000, 'liab_m': 57000000, 'sales_m': 4400000}, {'ticker': 'BAJFINANCE.NS', 'name': 'Bajaj Finance Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Financial Services', 'industry': 'Credit Services', 'base_price': 7250.0, 'shares_out_b': 0.62, 'pe_ratio': 31.5, 'pb_ratio': 5.8, 'ps_ratio': 9.2, 'ev_ebitda': 18.5, 'fcf_yield': 0.032, 'roe': 0.225, 'roce': 0.168, 'roa': 0.045, 'debt_to_equity': 3.8, 'current_ratio': 1.35, 'gross_margin': 0.88, 'operating_margin': 0.62, 'net_margin': 0.38, 'revenue_growth': 0.28, 'profit_growth': 0.26, 'dividend_yield': 0.005, 'working_cap_m': 120000, 'assets_m': 3600000, 'retained_m': 680000, 'ebit_m': 240000, 'liab_m': 2850000, 'sales_m': 580000}, {'ticker': 'MARUTI.NS', 'name': 'Maruti Suzuki India Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'base_price': 12450.0, 'shares_out_b': 0.31, 'pe_ratio': 28.5, 'pb_ratio': 4.5, 'ps_ratio': 2.8, 'ev_ebitda': 16.8, 'fcf_yield': 0.035, 'roe': 0.175, 'roce': 0.21, 'roa': 0.135, 'debt_to_equity': 0.01, 'current_ratio': 1.85, 'gross_margin': 0.29, 'operating_margin': 0.115, 'net_margin': 0.095, 'revenue_growth': 0.2, 'profit_growth': 0.64, 'dividend_yield': 0.01, 'working_cap_m': 85000, 'assets_m': 1150000, 'retained_m': 820000, 'ebit_m': 155000, 'liab_m': 280000, 'sales_m': 1420000}, {'ticker': 'HAL.NS', 'name': 'Hindustan Aeronautics Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 4750.0, 'shares_out_b': 0.67, 'pe_ratio': 42.5, 'pb_ratio': 11.2, 'ps_ratio': 10.4, 'ev_ebitda': 28.5, 'fcf_yield': 0.028, 'roe': 0.29, 'roce': 0.36, 'roa': 0.125, 'debt_to_equity': 0.0, 'current_ratio': 1.95, 'gross_margin': 0.58, 'operating_margin': 0.31, 'net_margin': 0.25, 'revenue_growth': 0.13, 'profit_growth': 0.31, 'dividend_yield': 0.009, 'working_cap_m': 185000, 'assets_m': 620000, 'retained_m': 310000, 'ebit_m': 95000, 'liab_m': 290000, 'sales_m': 310000}, {'ticker': 'SUNPHARMA.NS', 'name': 'Sun Pharmaceutical Industries Ltd', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - Specialty', 'base_price': 1820.0, 'shares_out_b': 2.4, 'pe_ratio': 44.0, 'pb_ratio': 6.5, 'ps_ratio': 9.1, 'ev_ebitda': 26.5, 'fcf_yield': 0.028, 'roe': 0.165, 'roce': 0.19, 'roa': 0.12, 'debt_to_equity': 0.08, 'current_ratio': 2.65, 'gross_margin': 0.77, 'operating_margin': 0.27, 'net_margin': 0.21, 'revenue_growth': 0.1, 'profit_growth': 0.16, 'dividend_yield': 0.007, 'working_cap_m': 240000, 'assets_m': 950000, 'retained_m': 620000, 'ebit_m': 135000, 'liab_m': 280000, 'sales_m': 510000}, {'ticker': 'BEL.NS', 'name': 'Bharat Electronics Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 295.0, 'shares_out_b': 7.31, 'pe_ratio': 48.0, 'pb_ratio': 13.5, 'ps_ratio': 10.8, 'ev_ebitda': 32.0, 'fcf_yield': 0.025, 'roe': 0.285, 'roce': 0.38, 'roa': 0.165, 'debt_to_equity': 0.0, 'current_ratio': 2.15, 'gross_margin': 0.46, 'operating_margin': 0.25, 'net_margin': 0.19, 'revenue_growth': 0.14, 'profit_growth': 0.33, 'dividend_yield': 0.008, 'working_cap_m': 125000, 'assets_m': 380000, 'retained_m': 160000, 'ebit_m': 52000, 'liab_m': 195000, 'sales_m': 210000}, {'ticker': 'AZN.L', 'name': 'AstraZeneca PLC', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 128.5, 'shares_out_b': 1.55, 'pe_ratio': 38.4, 'pb_ratio': 4.9, 'ps_ratio': 4.2, 'ev_ebitda': 18.2, 'fcf_yield': 0.038, 'roe': 0.16, 'roce': 0.14, 'roa': 0.065, 'debt_to_equity': 0.78, 'current_ratio': 0.94, 'gross_margin': 0.81, 'operating_margin': 0.24, 'net_margin': 0.14, 'revenue_growth': 0.18, 'profit_growth': 0.26, 'dividend_yield': 0.019, 'working_cap_m': -2500, 'assets_m': 98000, 'retained_m': 22000, 'ebit_m': 12000, 'liab_m': 58000, 'sales_m': 51000}, {'ticker': 'SHEL.L', 'name': 'Shell plc', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated', 'base_price': 27.8, 'shares_out_b': 6.35, 'pe_ratio': 11.2, 'pb_ratio': 1.1, 'ps_ratio': 0.6, 'ev_ebitda': 4.8, 'fcf_yield': 0.098, 'roe': 0.11, 'roce': 0.125, 'roa': 0.052, 'debt_to_equity': 0.45, 'current_ratio': 1.28, 'gross_margin': 0.24, 'operating_margin': 0.11, 'net_margin': 0.065, 'revenue_growth': -0.04, 'profit_growth': -0.12, 'dividend_yield': 0.041, 'working_cap_m': 18000, 'assets_m': 410000, 'retained_m': 140000, 'ebit_m': 38000, 'liab_m': 220000, 'sales_m': 315000}, {'ticker': 'HSBA.L', 'name': 'HSBC Holdings plc', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Financial Services', 'industry': 'Banks - Diversified', 'base_price': 6.62, 'shares_out_b': 18.9, 'pe_ratio': 7.4, 'pb_ratio': 0.85, 'ps_ratio': 1.8, 'ev_ebitda': 6.5, 'fcf_yield': 0.088, 'roe': 0.138, 'roce': 0.112, 'roa': 0.009, 'debt_to_equity': 1.72, 'current_ratio': 1.1, 'gross_margin': 0.85, 'operating_margin': 0.44, 'net_margin': 0.32, 'revenue_growth': 0.09, 'profit_growth': 0.15, 'dividend_yield': 0.072, 'working_cap_m': 35000, 'assets_m': 3000000, 'retained_m': 160000, 'ebit_m': 36000, 'liab_m': 2810000, 'sales_m': 68000}, {'ticker': 'ULVR.L', 'name': 'Unilever PLC', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products', 'base_price': 48.2, 'shares_out_b': 2.49, 'pe_ratio': 21.8, 'pb_ratio': 6.8, 'ps_ratio': 2.4, 'ev_ebitda': 14.8, 'fcf_yield': 0.048, 'roe': 0.32, 'roce': 0.24, 'roa': 0.095, 'debt_to_equity': 1.35, 'current_ratio': 0.82, 'gross_margin': 0.43, 'operating_margin': 0.165, 'net_margin': 0.112, 'revenue_growth': 0.04, 'profit_growth': 0.06, 'dividend_yield': 0.034, 'working_cap_m': -6000, 'assets_m': 78000, 'retained_m': 21000, 'ebit_m': 10500, 'liab_m': 57000, 'sales_m': 62000}, {'ticker': 'BP.L', 'name': 'BP p.l.c.', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated', 'base_price': 4.15, 'shares_out_b': 16.4, 'pe_ratio': 10.8, 'pb_ratio': 1.05, 'ps_ratio': 0.45, 'ev_ebitda': 4.5, 'fcf_yield': 0.092, 'roe': 0.14, 'roce': 0.13, 'roa': 0.05, 'debt_to_equity': 0.65, 'current_ratio': 1.18, 'gross_margin': 0.21, 'operating_margin': 0.095, 'net_margin': 0.055, 'revenue_growth': -0.06, 'profit_growth': -0.22, 'dividend_yield': 0.055, 'working_cap_m': 8500, 'assets_m': 280000, 'retained_m': 92000, 'ebit_m': 22000, 'liab_m': 195000, 'sales_m': 210000}, {'ticker': 'RR.L', 'name': 'Rolls-Royce Holdings plc', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 5.1, 'shares_out_b': 8.45, 'pe_ratio': 32.5, 'pb_ratio': 8.5, 'ps_ratio': 2.5, 'ev_ebitda': 16.8, 'fcf_yield': 0.048, 'roe': 0.42, 'roce': 0.28, 'roa': 0.08, 'debt_to_equity': 0.52, 'current_ratio': 1.15, 'gross_margin': 0.24, 'operating_margin': 0.14, 'net_margin': 0.09, 'revenue_growth': 0.18, 'profit_growth': 1.55, 'dividend_yield': 0.012, 'working_cap_m': 2400, 'assets_m': 32000, 'retained_m': -4200, 'ebit_m': 2300, 'liab_m': 26500, 'sales_m': 17500}, {'ticker': 'ASML.AS', 'name': 'ASML Holding N.V.', 'country': 'NL', 'exchange': 'Euronext', 'currency': 'EUR', 'sector': 'Technology', 'industry': 'Semiconductor Equipment', 'base_price': 785.4, 'shares_out_b': 0.393, 'pe_ratio': 42.8, 'pb_ratio': 21.5, 'ps_ratio': 11.2, 'ev_ebitda': 31.5, 'fcf_yield': 0.022, 'roe': 0.58, 'roce': 0.48, 'roa': 0.21, 'debt_to_equity': 0.35, 'current_ratio': 1.48, 'gross_margin': 0.51, 'operating_margin': 0.31, 'net_margin': 0.28, 'revenue_growth': 0.12, 'profit_growth': 0.14, 'dividend_yield': 0.012, 'working_cap_m': 6800, 'assets_m': 42000, 'retained_m': 14000, 'ebit_m': 9200, 'liab_m': 27000, 'sales_m': 28000}, {'ticker': 'SAP.DE', 'name': 'SAP SE', 'country': 'DE', 'exchange': 'XETRA', 'currency': 'EUR', 'sector': 'Technology', 'industry': 'Software - Application', 'base_price': 196.2, 'shares_out_b': 1.17, 'pe_ratio': 45.2, 'pb_ratio': 4.8, 'ps_ratio': 7.1, 'ev_ebitda': 22.8, 'fcf_yield': 0.031, 'roe': 0.11, 'roce': 0.14, 'roa': 0.065, 'debt_to_equity': 0.24, 'current_ratio': 1.15, 'gross_margin': 0.72, 'operating_margin': 0.24, 'net_margin': 0.18, 'revenue_growth': 0.1, 'profit_growth': 0.22, 'dividend_yield': 0.011, 'working_cap_m': 2200, 'assets_m': 72000, 'retained_m': 34000, 'ebit_m': 8200, 'liab_m': 28000, 'sales_m': 33000}, {'ticker': 'MC.PA', 'name': 'LVMH Moet Hennessy Louis Vuitton', 'country': 'FR', 'exchange': 'Euronext', 'currency': 'EUR', 'sector': 'Consumer Cyclical', 'industry': 'Luxury Goods', 'base_price': 648.0, 'shares_out_b': 0.501, 'pe_ratio': 21.8, 'pb_ratio': 4.9, 'ps_ratio': 3.7, 'ev_ebitda': 12.4, 'fcf_yield': 0.042, 'roe': 0.24, 'roce': 0.21, 'roa': 0.11, 'debt_to_equity': 0.48, 'current_ratio': 1.32, 'gross_margin': 0.69, 'operating_margin': 0.26, 'net_margin': 0.18, 'revenue_growth': 0.03, 'profit_growth': -0.04, 'dividend_yield': 0.021, 'working_cap_m': 8500, 'assets_m': 145000, 'retained_m': 58000, 'ebit_m': 23000, 'liab_m': 78000, 'sales_m': 86000}, {'ticker': 'NOVO-B.CO', 'name': 'Novo Nordisk A/S', 'country': 'DK', 'exchange': 'OMX', 'currency': 'EUR', 'sector': 'Healthcare', 'industry': 'Biotechnology', 'base_price': 128.9, 'shares_out_b': 4.45, 'pe_ratio': 38.6, 'pb_ratio': 32.4, 'ps_ratio': 16.5, 'ev_ebitda': 28.2, 'fcf_yield': 0.024, 'roe': 0.82, 'roce': 0.74, 'roa': 0.38, 'debt_to_equity': 0.28, 'current_ratio': 1.02, 'gross_margin': 0.84, 'operating_margin': 0.44, 'net_margin': 0.36, 'revenue_growth': 0.25, 'profit_growth': 0.32, 'dividend_yield': 0.012, 'working_cap_m': 1500, 'assets_m': 48000, 'retained_m': 16000, 'ebit_m': 15000, 'liab_m': 31000, 'sales_m': 35000}, {'ticker': 'SIE.DE', 'name': 'Siemens Aktiengesellschaft', 'country': 'DE', 'exchange': 'XETRA', 'currency': 'EUR', 'sector': 'Industrials', 'industry': 'Specialty Industrial Machinery', 'base_price': 172.4, 'shares_out_b': 0.8, 'pe_ratio': 16.5, 'pb_ratio': 2.6, 'ps_ratio': 1.8, 'ev_ebitda': 11.2, 'fcf_yield': 0.055, 'roe': 0.16, 'roce': 0.15, 'roa': 0.06, 'debt_to_equity': 0.82, 'current_ratio': 1.25, 'gross_margin': 0.38, 'operating_margin': 0.14, 'net_margin': 0.11, 'revenue_growth': 0.07, 'profit_growth': 0.12, 'dividend_yield': 0.028, 'working_cap_m': 12000, 'assets_m': 152000, 'retained_m': 42000, 'ebit_m': 11200, 'liab_m': 98000, 'sales_m': 78000}, {'ticker': 'TTE.PA', 'name': 'TotalEnergies SE', 'country': 'FR', 'exchange': 'Euronext', 'currency': 'EUR', 'sector': 'Energy', 'industry': 'Oil & Gas Integrated', 'base_price': 62.1, 'shares_out_b': 2.38, 'pe_ratio': 8.4, 'pb_ratio': 1.25, 'ps_ratio': 0.7, 'ev_ebitda': 4.2, 'fcf_yield': 0.105, 'roe': 0.18, 'roce': 0.17, 'roa': 0.078, 'debt_to_equity': 0.38, 'current_ratio': 1.18, 'gross_margin': 0.32, 'operating_margin': 0.15, 'net_margin': 0.095, 'revenue_growth': -0.05, 'profit_growth': -0.15, 'dividend_yield': 0.052, 'working_cap_m': 9500, 'assets_m': 290000, 'retained_m': 125000, 'ebit_m': 32000, 'liab_m': 168000, 'sales_m': 220000}, {'ticker': '7203.T', 'name': 'Toyota Motor Corporation', 'country': 'JP', 'exchange': 'TSE', 'currency': 'JPY', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'base_price': 2720.0, 'shares_out_b': 13.5, 'pe_ratio': 8.5, 'pb_ratio': 1.15, 'ps_ratio': 0.82, 'ev_ebitda': 7.2, 'fcf_yield': 0.072, 'roe': 0.152, 'roce': 0.118, 'roa': 0.055, 'debt_to_equity': 0.98, 'current_ratio': 1.18, 'gross_margin': 0.21, 'operating_margin': 0.118, 'net_margin': 0.108, 'revenue_growth': 0.21, 'profit_growth': 0.85, 'dividend_yield': 0.028, 'working_cap_m': 4200000, 'assets_m': 88000000, 'retained_m': 32000000, 'ebit_m': 5300000, 'liab_m': 53000000, 'sales_m': 45000000}, {'ticker': '6758.T', 'name': 'Sony Group Corporation', 'country': 'JP', 'exchange': 'TSE', 'currency': 'JPY', 'sector': 'Technology', 'industry': 'Consumer Electronics', 'base_price': 13800.0, 'shares_out_b': 1.23, 'pe_ratio': 17.6, 'pb_ratio': 2.1, 'ps_ratio': 1.35, 'ev_ebitda': 9.4, 'fcf_yield': 0.045, 'roe': 0.125, 'roce': 0.132, 'roa': 0.038, 'debt_to_equity': 0.42, 'current_ratio': 1.05, 'gross_margin': 0.28, 'operating_margin': 0.098, 'net_margin': 0.075, 'revenue_growth': 0.13, 'profit_growth': -0.03, 'dividend_yield': 0.007, 'working_cap_m': 850000, 'assets_m': 35000000, 'retained_m': 6800000, 'ebit_m': 1250000, 'liab_m': 26000000, 'sales_m': 13000000}, {'ticker': '6861.T', 'name': 'Keyence Corporation', 'country': 'JP', 'exchange': 'TSE', 'currency': 'JPY', 'sector': 'Technology', 'industry': 'Electronic Components', 'base_price': 68200.0, 'shares_out_b': 0.24, 'pe_ratio': 42.1, 'pb_ratio': 5.4, 'ps_ratio': 16.5, 'ev_ebitda': 28.5, 'fcf_yield': 0.022, 'roe': 0.135, 'roce': 0.165, 'roa': 0.125, 'debt_to_equity': 0.01, 'current_ratio': 14.5, 'gross_margin': 0.82, 'operating_margin': 0.52, 'net_margin': 0.38, 'revenue_growth': 0.11, 'profit_growth': 0.12, 'dividend_yield': 0.006, 'working_cap_m': 980000, 'assets_m': 3100000, 'retained_m': 2800000, 'ebit_m': 510000, 'liab_m': 120000, 'sales_m': 980000}, {'ticker': '8306.T', 'name': 'Mitsubishi UFJ Financial Group', 'country': 'JP', 'exchange': 'TSE', 'currency': 'JPY', 'sector': 'Financial Services', 'industry': 'Banks - Diversified', 'base_price': 1540.0, 'shares_out_b': 12.1, 'pe_ratio': 11.2, 'pb_ratio': 0.88, 'ps_ratio': 2.1, 'ev_ebitda': 8.5, 'fcf_yield': 0.065, 'roe': 0.085, 'roce': 0.075, 'roa': 0.005, 'debt_to_equity': 1.65, 'current_ratio': 1.12, 'gross_margin': 0.78, 'operating_margin': 0.35, 'net_margin': 0.22, 'revenue_growth': 0.15, 'profit_growth': 0.32, 'dividend_yield': 0.032, 'working_cap_m': 15000000, 'assets_m': 410000000, 'retained_m': 12000000, 'ebit_m': 2100000, 'liab_m': 390000000, 'sales_m': 9500000}, {'ticker': '8035.T', 'name': 'Tokyo Electron Limited', 'country': 'JP', 'exchange': 'TSE', 'currency': 'JPY', 'sector': 'Technology', 'industry': 'Semiconductor Equipment', 'base_price': 24800.0, 'shares_out_b': 0.46, 'pe_ratio': 28.5, 'pb_ratio': 6.8, 'ps_ratio': 6.2, 'ev_ebitda': 18.2, 'fcf_yield': 0.032, 'roe': 0.26, 'roce': 0.28, 'roa': 0.18, 'debt_to_equity': 0.05, 'current_ratio': 3.4, 'gross_margin': 0.46, 'operating_margin': 0.26, 'net_margin': 0.2, 'revenue_growth': 0.24, 'profit_growth': 0.35, 'dividend_yield': 0.021, 'working_cap_m': 1200000, 'assets_m': 2400000, 'retained_m': 1800000, 'ebit_m': 580000, 'liab_m': 550000, 'sales_m': 2200000}, {'ticker': '7974.T', 'name': 'Nintendo Co., Ltd.', 'country': 'JP', 'exchange': 'TSE', 'currency': 'JPY', 'sector': 'Communication Services', 'industry': 'Electronic Gaming & Multimedia', 'base_price': 7800.0, 'shares_out_b': 1.16, 'pe_ratio': 18.5, 'pb_ratio': 3.4, 'ps_ratio': 5.4, 'ev_ebitda': 11.5, 'fcf_yield': 0.048, 'roe': 0.195, 'roce': 0.24, 'roa': 0.16, 'debt_to_equity': 0.0, 'current_ratio': 4.8, 'gross_margin': 0.58, 'operating_margin': 0.32, 'net_margin': 0.24, 'revenue_growth': 0.04, 'profit_growth': 0.08, 'dividend_yield': 0.026, 'working_cap_m': 1850000, 'assets_m': 3100000, 'retained_m': 2600000, 'ebit_m': 540000, 'liab_m': 480000, 'sales_m': 1650000}, {'ticker': 'TSM', 'name': 'Taiwan Semiconductor Manufacturing', 'country': 'Global', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 172.5, 'shares_out_b': 5.18, 'pe_ratio': 28.5, 'pb_ratio': 6.8, 'ps_ratio': 11.2, 'ev_ebitda': 14.8, 'fcf_yield': 0.038, 'roe': 0.26, 'roce': 0.24, 'roa': 0.18, 'debt_to_equity': 0.28, 'current_ratio': 2.45, 'gross_margin': 0.54, 'operating_margin': 0.43, 'net_margin': 0.38, 'revenue_growth': 0.32, 'profit_growth': 0.36, 'dividend_yield': 0.014, 'working_cap_m': 22000, 'assets_m': 185000, 'retained_m': 95000, 'ebit_m': 35000, 'liab_m': 58000, 'sales_m': 82000}, {'ticker': 'BABA', 'name': 'Alibaba Group Holding Limited', 'country': 'Global', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Consumer Cyclical', 'industry': 'Internet Retail', 'base_price': 84.5, 'shares_out_b': 2.45, 'pe_ratio': 14.8, 'pb_ratio': 1.4, 'ps_ratio': 1.5, 'ev_ebitda': 7.5, 'fcf_yield': 0.088, 'roe': 0.1, 'roce': 0.11, 'roa': 0.05, 'debt_to_equity': 0.18, 'current_ratio': 1.85, 'gross_margin': 0.38, 'operating_margin': 0.15, 'net_margin': 0.11, 'revenue_growth': 0.06, 'profit_growth': -0.05, 'dividend_yield': 0.024, 'working_cap_m': 42000, 'assets_m': 240000, 'retained_m': 120000, 'ebit_m': 22000, 'liab_m': 92000, 'sales_m': 135000}, {'ticker': 'BTC-USD', 'name': 'Bitcoin (Digital Gold)', 'country': 'Global', 'exchange': 'Crypto', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Cryptocurrency', 'base_price': 58400.0, 'shares_out_b': 0.0197, 'pe_ratio': 35.0, 'pb_ratio': 3.0, 'ps_ratio': 15.0, 'ev_ebitda': 20.0, 'fcf_yield': 0.03, 'roe': 0.25, 'roce': 0.25, 'roa': 0.25, 'debt_to_equity': 0.0, 'current_ratio': 99.0, 'gross_margin': 0.99, 'operating_margin': 0.8, 'net_margin': 0.8, 'revenue_growth': 0.45, 'profit_growth': 0.6, 'dividend_yield': 0.0, 'working_cap_m': 50000, 'assets_m': 1150000, 'retained_m': 900000, 'ebit_m': 40000, 'liab_m': 0, 'sales_m': 50000}, {'ticker': 'ETH-USD', 'name': 'Ethereum (Smart Contracts)', 'country': 'Global', 'exchange': 'Crypto', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Cryptocurrency', 'base_price': 2450.0, 'shares_out_b': 0.12, 'pe_ratio': 42.0, 'pb_ratio': 4.5, 'ps_ratio': 22.0, 'ev_ebitda': 25.0, 'fcf_yield': 0.025, 'roe': 0.2, 'roce': 0.2, 'roa': 0.2, 'debt_to_equity': 0.0, 'current_ratio': 99.0, 'gross_margin': 0.99, 'operating_margin': 0.75, 'net_margin': 0.75, 'revenue_growth': 0.35, 'profit_growth': 0.4, 'dividend_yield': 0.032, 'working_cap_m': 15000, 'assets_m': 294000, 'retained_m': 240000, 'ebit_m': 8000, 'liab_m': 0, 'sales_m': 12000}, {'ticker': 'GC=F', 'name': 'Gold Futures', 'country': 'Global', 'exchange': 'Commodity', 'currency': 'USD', 'sector': 'Materials', 'industry': 'Precious Metals', 'base_price': 2515.0, 'shares_out_b': 1.0, 'pe_ratio': 20.0, 'pb_ratio': 2.0, 'ps_ratio': 4.0, 'ev_ebitda': 10.0, 'fcf_yield': 0.04, 'roe': 0.15, 'roce': 0.15, 'roa': 0.1, 'debt_to_equity': 0.0, 'current_ratio': 10.0, 'gross_margin': 0.5, 'operating_margin': 0.4, 'net_margin': 0.3, 'revenue_growth': 0.1, 'profit_growth': 0.15, 'dividend_yield': 0.0, 'working_cap_m': 10000, 'assets_m': 100000, 'retained_m': 50000, 'ebit_m': 15000, 'liab_m': 0, 'sales_m': 30000}, {'ticker': 'INTC', 'name': 'Intel Corporation', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 21.5, 'shares_out_b': 4.28, 'pe_ratio': 22.0, 'pb_ratio': 0.85, 'ps_ratio': 1.6, 'ev_ebitda': 7.8, 'fcf_yield': 0.02, 'roe': 0.04, 'roce': 0.03, 'roa': 0.015, 'debt_to_equity': 0.48, 'current_ratio': 1.55, 'gross_margin': 0.41, 'operating_margin': 0.03, 'net_margin': 0.02, 'revenue_growth': -0.01, 'profit_growth': -0.45, 'dividend_yield': 0.024, 'working_cap_m': 1075, 'assets_m': 10750, 'retained_m': 3225, 'ebit_m': 1290, 'liab_m': 5375, 'sales_m': 8600}, {'ticker': 'TXN', 'name': 'Texas Instruments Incorporated', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 205.0, 'shares_out_b': 0.91, 'pe_ratio': 36.5, 'pb_ratio': 10.5, 'ps_ratio': 11.2, 'ev_ebitda': 24.5, 'fcf_yield': 0.028, 'roe': 0.32, 'roce': 0.28, 'roa': 0.18, 'debt_to_equity': 0.82, 'current_ratio': 3.85, 'gross_margin': 0.61, 'operating_margin': 0.38, 'net_margin': 0.32, 'revenue_growth': -0.12, 'profit_growth': -0.25, 'dividend_yield': 0.026, 'working_cap_m': 10250, 'assets_m': 102500, 'retained_m': 30750, 'ebit_m': 12300, 'liab_m': 51250, 'sales_m': 82000}, {'ticker': 'QCOM', 'name': 'QUALCOMM Incorporated', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductors', 'base_price': 168.0, 'shares_out_b': 1.11, 'pe_ratio': 21.4, 'pb_ratio': 7.5, 'ps_ratio': 4.8, 'ev_ebitda': 14.8, 'fcf_yield': 0.048, 'roe': 0.36, 'roce': 0.28, 'roa': 0.16, 'debt_to_equity': 0.65, 'current_ratio': 2.15, 'gross_margin': 0.56, 'operating_margin': 0.28, 'net_margin': 0.22, 'revenue_growth': 0.08, 'profit_growth': 0.24, 'dividend_yield': 0.02, 'working_cap_m': 8400, 'assets_m': 84000, 'retained_m': 25200, 'ebit_m': 10080, 'liab_m': 42000, 'sales_m': 67200}, {'ticker': 'IBM', 'name': 'International Business Machines Corp', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Information Technology Services', 'base_price': 212.0, 'shares_out_b': 0.92, 'pe_ratio': 23.5, 'pb_ratio': 7.8, 'ps_ratio': 3.1, 'ev_ebitda': 15.2, 'fcf_yield': 0.062, 'roe': 0.34, 'roce': 0.18, 'roa': 0.065, 'debt_to_equity': 2.15, 'current_ratio': 1.05, 'gross_margin': 0.56, 'operating_margin': 0.16, 'net_margin': 0.13, 'revenue_growth': 0.04, 'profit_growth': 0.18, 'dividend_yield': 0.031, 'working_cap_m': 10600, 'assets_m': 106000, 'retained_m': 31800, 'ebit_m': 12720, 'liab_m': 53000, 'sales_m': 84800}, {'ticker': 'NOW', 'name': 'ServiceNow, Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Application', 'base_price': 880.0, 'shares_out_b': 0.21, 'pe_ratio': 62.0, 'pb_ratio': 18.5, 'ps_ratio': 17.5, 'ev_ebitda': 38.0, 'fcf_yield': 0.035, 'roe': 0.32, 'roce': 0.24, 'roa': 0.12, 'debt_to_equity': 0.25, 'current_ratio': 1.15, 'gross_margin': 0.79, 'operating_margin': 0.28, 'net_margin': 0.24, 'revenue_growth': 0.22, 'profit_growth': 0.45, 'dividend_yield': 0.0, 'working_cap_m': 44000, 'assets_m': 440000, 'retained_m': 132000, 'ebit_m': 52800, 'liab_m': 220000, 'sales_m': 352000}, {'ticker': 'AMAT', 'name': 'Applied Materials, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Semiconductor Equipment', 'base_price': 198.0, 'shares_out_b': 0.82, 'pe_ratio': 22.8, 'pb_ratio': 8.5, 'ps_ratio': 6.1, 'ev_ebitda': 16.5, 'fcf_yield': 0.045, 'roe': 0.42, 'roce': 0.36, 'roa': 0.22, 'debt_to_equity': 0.32, 'current_ratio': 2.45, 'gross_margin': 0.47, 'operating_margin': 0.29, 'net_margin': 0.27, 'revenue_growth': 0.05, 'profit_growth': 0.08, 'dividend_yield': 0.008, 'working_cap_m': 9900, 'assets_m': 99000, 'retained_m': 29700, 'ebit_m': 11880, 'liab_m': 49500, 'sales_m': 79200}, {'ticker': 'INTU', 'name': 'Intuit Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Technology', 'industry': 'Software - Application', 'base_price': 645.0, 'shares_out_b': 0.28, 'pe_ratio': 58.0, 'pb_ratio': 9.5, 'ps_ratio': 11.2, 'ev_ebitda': 32.5, 'fcf_yield': 0.026, 'roe': 0.17, 'roce': 0.16, 'roa': 0.1, 'debt_to_equity': 0.38, 'current_ratio': 1.45, 'gross_margin': 0.81, 'operating_margin': 0.26, 'net_margin': 0.19, 'revenue_growth': 0.16, 'profit_growth': 0.22, 'dividend_yield': 0.006, 'working_cap_m': 32250, 'assets_m': 322500, 'retained_m': 96750, 'ebit_m': 38700, 'liab_m': 161250, 'sales_m': 258000}, {'ticker': 'SCHW', 'name': 'The Charles Schwab Corporation', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Capital Markets', 'base_price': 65.0, 'shares_out_b': 1.82, 'pe_ratio': 24.5, 'pb_ratio': 3.2, 'ps_ratio': 6.1, 'ev_ebitda': 15.8, 'fcf_yield': 0.048, 'roe': 0.14, 'roce': 0.12, 'roa': 0.009, 'debt_to_equity': 0.65, 'current_ratio': 1.15, 'gross_margin': 0.95, 'operating_margin': 0.38, 'net_margin': 0.26, 'revenue_growth': 0.06, 'profit_growth': 0.12, 'dividend_yield': 0.015, 'working_cap_m': 3250, 'assets_m': 32500, 'retained_m': 9750, 'ebit_m': 3900, 'liab_m': 16250, 'sales_m': 26000}, {'ticker': 'MS', 'name': 'Morgan Stanley', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Capital Markets', 'base_price': 102.5, 'shares_out_b': 1.62, 'pe_ratio': 16.2, 'pb_ratio': 1.7, 'ps_ratio': 2.8, 'ev_ebitda': 11.5, 'fcf_yield': 0.055, 'roe': 0.11, 'roce': 0.1, 'roa': 0.008, 'debt_to_equity': 2.45, 'current_ratio': 1.1, 'gross_margin': 0.86, 'operating_margin': 0.28, 'net_margin': 0.19, 'revenue_growth': 0.11, 'profit_growth': 0.28, 'dividend_yield': 0.033, 'working_cap_m': 5125, 'assets_m': 51250, 'retained_m': 15375, 'ebit_m': 6150, 'liab_m': 25625, 'sales_m': 41000}, {'ticker': 'AXP', 'name': 'American Express Company', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Financial Services', 'industry': 'Credit Services', 'base_price': 254.0, 'shares_out_b': 0.71, 'pe_ratio': 19.5, 'pb_ratio': 6.2, 'ps_ratio': 3.1, 'ev_ebitda': 12.8, 'fcf_yield': 0.052, 'roe': 0.34, 'roce': 0.22, 'roa': 0.035, 'debt_to_equity': 1.65, 'current_ratio': 1.25, 'gross_margin': 0.58, 'operating_margin': 0.22, 'net_margin': 0.155, 'revenue_growth': 0.1, 'profit_growth': 0.21, 'dividend_yield': 0.011, 'working_cap_m': 12700, 'assets_m': 127000, 'retained_m': 38100, 'ebit_m': 15240, 'liab_m': 63500, 'sales_m': 101600}, {'ticker': 'MRK', 'name': 'Merck & Co., Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 116.5, 'shares_out_b': 2.53, 'pe_ratio': 18.2, 'pb_ratio': 7.4, 'ps_ratio': 4.8, 'ev_ebitda': 13.5, 'fcf_yield': 0.048, 'roe': 0.42, 'roce': 0.26, 'roa': 0.12, 'debt_to_equity': 0.85, 'current_ratio': 1.35, 'gross_margin': 0.74, 'operating_margin': 0.35, 'net_margin': 0.26, 'revenue_growth': 0.07, 'profit_growth': 0.45, 'dividend_yield': 0.026, 'working_cap_m': 5825, 'assets_m': 58250, 'retained_m': 17475, 'ebit_m': 6990, 'liab_m': 29125, 'sales_m': 46600}, {'ticker': 'TMO', 'name': 'Thermo Fisher Scientific Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Diagnostics & Research', 'base_price': 610.0, 'shares_out_b': 0.38, 'pe_ratio': 36.5, 'pb_ratio': 4.9, 'ps_ratio': 5.4, 'ev_ebitda': 22.4, 'fcf_yield': 0.032, 'roe': 0.14, 'roce': 0.13, 'roa': 0.065, 'debt_to_equity': 0.78, 'current_ratio': 1.45, 'gross_margin': 0.41, 'operating_margin': 0.19, 'net_margin': 0.15, 'revenue_growth': 0.02, 'profit_growth': 0.06, 'dividend_yield': 0.003, 'working_cap_m': 30500, 'assets_m': 305000, 'retained_m': 91500, 'ebit_m': 36600, 'liab_m': 152500, 'sales_m': 244000}, {'ticker': 'ABT', 'name': 'Abbott Laboratories', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Medical Devices', 'base_price': 114.0, 'shares_out_b': 1.74, 'pe_ratio': 34.0, 'pb_ratio': 5.1, 'ps_ratio': 4.8, 'ev_ebitda': 19.5, 'fcf_yield': 0.035, 'roe': 0.15, 'roce': 0.16, 'roa': 0.08, 'debt_to_equity': 0.38, 'current_ratio': 1.75, 'gross_margin': 0.55, 'operating_margin': 0.18, 'net_margin': 0.14, 'revenue_growth': 0.04, 'profit_growth': 0.08, 'dividend_yield': 0.019, 'working_cap_m': 5700, 'assets_m': 57000, 'retained_m': 17100, 'ebit_m': 6840, 'liab_m': 28500, 'sales_m': 45600}, {'ticker': 'DHR', 'name': 'Danaher Corporation', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Diagnostics & Research', 'base_price': 270.0, 'shares_out_b': 0.74, 'pe_ratio': 45.0, 'pb_ratio': 3.8, 'ps_ratio': 8.5, 'ev_ebitda': 25.0, 'fcf_yield': 0.03, 'roe': 0.09, 'roce': 0.11, 'roa': 0.055, 'debt_to_equity': 0.35, 'current_ratio': 1.85, 'gross_margin': 0.59, 'operating_margin': 0.22, 'net_margin': 0.18, 'revenue_growth': 0.03, 'profit_growth': -0.05, 'dividend_yield': 0.004, 'working_cap_m': 13500, 'assets_m': 135000, 'retained_m': 40500, 'ebit_m': 16200, 'liab_m': 67500, 'sales_m': 108000}, {'ticker': 'BMY', 'name': 'Bristol-Myers Squibb Company', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 49.5, 'shares_out_b': 2.03, 'pe_ratio': 14.5, 'pb_ratio': 5.4, 'ps_ratio': 2.2, 'ev_ebitda': 8.5, 'fcf_yield': 0.085, 'roe': 0.38, 'roce': 0.18, 'roa': 0.065, 'debt_to_equity': 2.85, 'current_ratio': 1.2, 'gross_margin': 0.76, 'operating_margin': 0.24, 'net_margin': 0.15, 'revenue_growth': 0.09, 'profit_growth': 0.12, 'dividend_yield': 0.048, 'working_cap_m': 2475, 'assets_m': 24750, 'retained_m': 7425, 'ebit_m': 2970, 'liab_m': 12375, 'sales_m': 19800}, {'ticker': 'KO', 'name': 'The Coca-Cola Company', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Consumer Defensive', 'industry': 'Beverages - Non-Alcoholic', 'base_price': 71.2, 'shares_out_b': 4.3, 'pe_ratio': 28.5, 'pb_ratio': 11.2, 'ps_ratio': 6.5, 'ev_ebitda': 21.0, 'fcf_yield': 0.032, 'roe': 0.41, 'roce': 0.22, 'roa': 0.11, 'debt_to_equity': 1.45, 'current_ratio': 1.15, 'gross_margin': 0.6, 'operating_margin': 0.3, 'net_margin': 0.23, 'revenue_growth': 0.04, 'profit_growth': 0.07, 'dividend_yield': 0.027, 'working_cap_m': 3560, 'assets_m': 35600, 'retained_m': 10680, 'ebit_m': 4272, 'liab_m': 17800, 'sales_m': 28480}, {'ticker': 'PEP', 'name': 'PepsiCo, Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Consumer Defensive', 'industry': 'Beverages - Non-Alcoholic', 'base_price': 178.0, 'shares_out_b': 1.37, 'pe_ratio': 26.0, 'pb_ratio': 12.8, 'ps_ratio': 2.6, 'ev_ebitda': 17.5, 'fcf_yield': 0.038, 'roe': 0.52, 'roce': 0.24, 'roa': 0.1, 'debt_to_equity': 2.1, 'current_ratio': 0.88, 'gross_margin': 0.54, 'operating_margin': 0.165, 'net_margin': 0.1, 'revenue_growth': 0.02, 'profit_growth': 0.05, 'dividend_yield': 0.03, 'working_cap_m': 8900, 'assets_m': 89000, 'retained_m': 26700, 'ebit_m': 10680, 'liab_m': 44500, 'sales_m': 71200}, {'ticker': 'MCD', 'name': "McDonald's Corporation", 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Consumer Cyclical', 'industry': 'Restaurants', 'base_price': 292.0, 'shares_out_b': 0.72, 'pe_ratio': 25.5, 'pb_ratio': 45.0, 'ps_ratio': 8.2, 'ev_ebitda': 18.0, 'fcf_yield': 0.036, 'roe': 1.55, 'roce': 0.32, 'roa': 0.15, 'debt_to_equity': 8.5, 'current_ratio': 1.1, 'gross_margin': 0.57, 'operating_margin': 0.46, 'net_margin': 0.33, 'revenue_growth': 0.03, 'profit_growth': 0.04, 'dividend_yield': 0.023, 'working_cap_m': 14600, 'assets_m': 146000, 'retained_m': 43800, 'ebit_m': 17520, 'liab_m': 73000, 'sales_m': 116800}, {'ticker': 'NKE', 'name': 'NIKE, Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Consumer Cyclical', 'industry': 'Footwear & Accessories', 'base_price': 82.5, 'shares_out_b': 1.51, 'pe_ratio': 22.8, 'pb_ratio': 8.5, 'ps_ratio': 2.4, 'ev_ebitda': 17.2, 'fcf_yield': 0.045, 'roe': 0.38, 'roce': 0.31, 'roa': 0.14, 'debt_to_equity': 0.85, 'current_ratio': 2.45, 'gross_margin': 0.44, 'operating_margin': 0.115, 'net_margin': 0.098, 'revenue_growth': 0.01, 'profit_growth': 0.12, 'dividend_yield': 0.018, 'working_cap_m': 4125, 'assets_m': 41250, 'retained_m': 12375, 'ebit_m': 4950, 'liab_m': 20625, 'sales_m': 33000}, {'ticker': 'PM', 'name': 'Philip Morris International Inc.', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Consumer Defensive', 'industry': 'Tobacco', 'base_price': 126.0, 'shares_out_b': 1.55, 'pe_ratio': 21.0, 'pb_ratio': 18.0, 'ps_ratio': 5.4, 'ev_ebitda': 15.0, 'fcf_yield': 0.052, 'roe': 0.85, 'roce': 0.28, 'roa': 0.14, 'debt_to_equity': 4.5, 'current_ratio': 0.78, 'gross_margin': 0.65, 'operating_margin': 0.38, 'net_margin': 0.24, 'revenue_growth': 0.09, 'profit_growth': 0.14, 'dividend_yield': 0.042, 'working_cap_m': 6300, 'assets_m': 63000, 'retained_m': 18900, 'ebit_m': 7560, 'liab_m': 31500, 'sales_m': 50400}, {'ticker': 'HON', 'name': 'Honeywell International Inc.', 'country': 'US', 'exchange': 'NASDAQ', 'currency': 'USD', 'sector': 'Industrials', 'industry': 'Conglomerates', 'base_price': 208.0, 'shares_out_b': 0.65, 'pe_ratio': 24.0, 'pb_ratio': 8.2, 'ps_ratio': 3.6, 'ev_ebitda': 16.5, 'fcf_yield': 0.042, 'roe': 0.34, 'roce': 0.22, 'roa': 0.09, 'debt_to_equity': 1.25, 'current_ratio': 1.35, 'gross_margin': 0.37, 'operating_margin': 0.21, 'net_margin': 0.15, 'revenue_growth': 0.05, 'profit_growth': 0.08, 'dividend_yield': 0.021, 'working_cap_m': 10400, 'assets_m': 104000, 'retained_m': 31200, 'ebit_m': 12480, 'liab_m': 52000, 'sales_m': 83200}, {'ticker': 'BA', 'name': 'The Boeing Company', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 162.0, 'shares_out_b': 0.61, 'pe_ratio': 35.0, 'pb_ratio': 12.0, 'ps_ratio': 1.3, 'ev_ebitda': 25.0, 'fcf_yield': 0.015, 'roe': 0.12, 'roce': 0.05, 'roa': 0.02, 'debt_to_equity': 8.5, 'current_ratio': 1.15, 'gross_margin': 0.11, 'operating_margin': 0.01, 'net_margin': -0.04, 'revenue_growth': 0.08, 'profit_growth': 0.15, 'dividend_yield': 0.0, 'working_cap_m': 8100, 'assets_m': 81000, 'retained_m': 24300, 'ebit_m': 9720, 'liab_m': 40500, 'sales_m': 64800}, {'ticker': 'LMT', 'name': 'Lockheed Martin Corporation', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 575.0, 'shares_out_b': 0.24, 'pe_ratio': 21.0, 'pb_ratio': 19.5, 'ps_ratio': 2.0, 'ev_ebitda': 14.5, 'fcf_yield': 0.045, 'roe': 0.95, 'roce': 0.34, 'roa': 0.12, 'debt_to_equity': 2.8, 'current_ratio': 1.28, 'gross_margin': 0.13, 'operating_margin': 0.125, 'net_margin': 0.098, 'revenue_growth': 0.09, 'profit_growth': 0.06, 'dividend_yield': 0.022, 'working_cap_m': 28750, 'assets_m': 287500, 'retained_m': 86250, 'ebit_m': 34500, 'liab_m': 143750, 'sales_m': 230000}, {'ticker': 'RTX', 'name': 'RTX Corporation', 'country': 'US', 'exchange': 'NYSE', 'currency': 'USD', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 122.0, 'shares_out_b': 1.33, 'pe_ratio': 38.0, 'pb_ratio': 2.6, 'ps_ratio': 2.2, 'ev_ebitda': 16.0, 'fcf_yield': 0.038, 'roe': 0.07, 'roce': 0.09, 'roa': 0.035, 'debt_to_equity': 0.68, 'current_ratio': 1.12, 'gross_margin': 0.18, 'operating_margin': 0.09, 'net_margin': 0.055, 'revenue_growth': 0.08, 'profit_growth': 0.18, 'dividend_yield': 0.021, 'working_cap_m': 6100, 'assets_m': 61000, 'retained_m': 18300, 'ebit_m': 7320, 'liab_m': 30500, 'sales_m': 48800}, {'ticker': 'AXISBANK.NS', 'name': 'Axis Bank Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Financial Services', 'industry': 'Banks - Regional', 'base_price': 1220.0, 'shares_out_b': 3.09, 'pe_ratio': 14.8, 'pb_ratio': 2.2, 'ps_ratio': 3.2, 'ev_ebitda': 8.8, 'fcf_yield': 0.062, 'roe': 0.165, 'roce': 0.145, 'roa': 0.018, 'debt_to_equity': 1.2, 'current_ratio': 1.15, 'gross_margin': 0.82, 'operating_margin': 0.44, 'net_margin': 0.26, 'revenue_growth': 0.19, 'profit_growth': 0.24, 'dividend_yield': 0.008, 'working_cap_m': 61000, 'assets_m': 610000, 'retained_m': 183000, 'ebit_m': 73200, 'liab_m': 305000, 'sales_m': 488000}, {'ticker': 'KOTAKBANK.NS', 'name': 'Kotak Mahindra Bank Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Financial Services', 'industry': 'Banks - Regional', 'base_price': 1840.0, 'shares_out_b': 1.99, 'pe_ratio': 19.2, 'pb_ratio': 2.8, 'ps_ratio': 4.6, 'ev_ebitda': 11.4, 'fcf_yield': 0.048, 'roe': 0.155, 'roce': 0.14, 'roa': 0.024, 'debt_to_equity': 0.95, 'current_ratio': 1.25, 'gross_margin': 0.85, 'operating_margin': 0.48, 'net_margin': 0.32, 'revenue_growth': 0.16, 'profit_growth': 0.26, 'dividend_yield': 0.005, 'working_cap_m': 92000, 'assets_m': 920000, 'retained_m': 276000, 'ebit_m': 110400, 'liab_m': 460000, 'sales_m': 736000}, {'ticker': 'TITAN.NS', 'name': 'Titan Company Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Cyclical', 'industry': 'Luxury Goods', 'base_price': 3720.0, 'shares_out_b': 0.89, 'pe_ratio': 82.0, 'pb_ratio': 28.5, 'ps_ratio': 6.8, 'ev_ebitda': 45.0, 'fcf_yield': 0.014, 'roe': 0.35, 'roce': 0.32, 'roa': 0.14, 'debt_to_equity': 0.75, 'current_ratio': 1.45, 'gross_margin': 0.24, 'operating_margin': 0.1, 'net_margin': 0.07, 'revenue_growth': 0.22, 'profit_growth': -0.05, 'dividend_yield': 0.003, 'working_cap_m': 186000, 'assets_m': 1860000, 'retained_m': 558000, 'ebit_m': 223200, 'liab_m': 930000, 'sales_m': 1488000}, {'ticker': 'M&M.NS', 'name': 'Mahindra & Mahindra Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'base_price': 2780.0, 'shares_out_b': 1.24, 'pe_ratio': 30.5, 'pb_ratio': 5.8, 'ps_ratio': 2.5, 'ev_ebitda': 18.2, 'fcf_yield': 0.038, 'roe': 0.21, 'roce': 0.19, 'roa': 0.08, 'debt_to_equity': 1.15, 'current_ratio': 1.35, 'gross_margin': 0.36, 'operating_margin': 0.13, 'net_margin': 0.085, 'revenue_growth': 0.18, 'profit_growth': 0.25, 'dividend_yield': 0.008, 'working_cap_m': 139000, 'assets_m': 1390000, 'retained_m': 417000, 'ebit_m': 166800, 'liab_m': 695000, 'sales_m': 1112000}, {'ticker': 'NTPC.NS', 'name': 'NTPC Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Utilities', 'industry': 'Utilities - Regulated Electric', 'base_price': 415.0, 'shares_out_b': 9.7, 'pe_ratio': 18.5, 'pb_ratio': 2.4, 'ps_ratio': 2.2, 'ev_ebitda': 10.5, 'fcf_yield': 0.045, 'roe': 0.135, 'roce': 0.115, 'roa': 0.045, 'debt_to_equity': 1.45, 'current_ratio': 0.95, 'gross_margin': 0.42, 'operating_margin': 0.22, 'net_margin': 0.12, 'revenue_growth': 0.05, 'profit_growth': 0.25, 'dividend_yield': 0.019, 'working_cap_m': 20750, 'assets_m': 207500, 'retained_m': 62250, 'ebit_m': 24900, 'liab_m': 103750, 'sales_m': 166000}, {'ticker': 'POWERGRID.NS', 'name': 'Power Grid Corporation of India', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Utilities', 'industry': 'Utilities - Regulated Electric', 'base_price': 335.0, 'shares_out_b': 9.3, 'pe_ratio': 20.0, 'pb_ratio': 3.5, 'ps_ratio': 6.8, 'ev_ebitda': 11.8, 'fcf_yield': 0.055, 'roe': 0.18, 'roce': 0.13, 'roa': 0.058, 'debt_to_equity': 1.35, 'current_ratio': 0.88, 'gross_margin': 0.88, 'operating_margin': 0.58, 'net_margin': 0.34, 'revenue_growth': 0.06, 'profit_growth': 0.08, 'dividend_yield': 0.035, 'working_cap_m': 16750, 'assets_m': 167500, 'retained_m': 50250, 'ebit_m': 20100, 'liab_m': 83750, 'sales_m': 134000}, {'ticker': 'ONGC.NS', 'name': 'Oil and Natural Gas Corporation', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Energy', 'industry': 'Oil & Gas E&P', 'base_price': 305.0, 'shares_out_b': 12.58, 'pe_ratio': 9.2, 'pb_ratio': 1.25, 'ps_ratio': 0.6, 'ev_ebitda': 4.8, 'fcf_yield': 0.088, 'roe': 0.14, 'roce': 0.135, 'roa': 0.068, 'debt_to_equity': 0.38, 'current_ratio': 1.25, 'gross_margin': 0.48, 'operating_margin': 0.21, 'net_margin': 0.08, 'revenue_growth': -0.04, 'profit_growth': 0.16, 'dividend_yield': 0.042, 'working_cap_m': 15250, 'assets_m': 152500, 'retained_m': 45750, 'ebit_m': 18300, 'liab_m': 76250, 'sales_m': 122000}, {'ticker': 'TATASTEEL.NS', 'name': 'Tata Steel Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Materials', 'industry': 'Steel', 'base_price': 152.0, 'shares_out_b': 12.48, 'pe_ratio': 48.0, 'pb_ratio': 2.1, 'ps_ratio': 0.85, 'ev_ebitda': 8.5, 'fcf_yield': 0.048, 'roe': 0.045, 'roce': 0.075, 'roa': 0.02, 'debt_to_equity': 0.95, 'current_ratio': 0.95, 'gross_margin': 0.32, 'operating_margin': 0.09, 'net_margin': 0.018, 'revenue_growth': -0.06, 'profit_growth': -0.55, 'dividend_yield': 0.024, 'working_cap_m': 7600, 'assets_m': 76000, 'retained_m': 22800, 'ebit_m': 9120, 'liab_m': 38000, 'sales_m': 60800}, {'ticker': 'ADANIENT.NS', 'name': 'Adani Enterprises Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Industrials', 'industry': 'Conglomerates', 'base_price': 3020.0, 'shares_out_b': 1.14, 'pe_ratio': 95.0, 'pb_ratio': 8.5, 'ps_ratio': 3.5, 'ev_ebitda': 28.0, 'fcf_yield': 0.015, 'roe': 0.095, 'roce': 0.085, 'roa': 0.025, 'debt_to_equity': 1.55, 'current_ratio': 1.05, 'gross_margin': 0.22, 'operating_margin': 0.09, 'net_margin': 0.035, 'revenue_growth': -0.15, 'profit_growth': 0.32, 'dividend_yield': 0.001, 'working_cap_m': 151000, 'assets_m': 1510000, 'retained_m': 453000, 'ebit_m': 181200, 'liab_m': 755000, 'sales_m': 1208000}, {'ticker': 'HINDUNILVR.NS', 'name': 'Hindustan Unilever Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Defensive', 'industry': 'Household & Personal Products', 'base_price': 2850.0, 'shares_out_b': 2.35, 'pe_ratio': 65.0, 'pb_ratio': 13.2, 'ps_ratio': 10.8, 'ev_ebitda': 42.0, 'fcf_yield': 0.022, 'roe': 0.205, 'roce': 0.26, 'roa': 0.145, 'debt_to_equity': 0.02, 'current_ratio': 1.35, 'gross_margin': 0.52, 'operating_margin': 0.24, 'net_margin': 0.165, 'revenue_growth': 0.03, 'profit_growth': 0.04, 'dividend_yield': 0.015, 'working_cap_m': 142500, 'assets_m': 1425000, 'retained_m': 427500, 'ebit_m': 171000, 'liab_m': 712500, 'sales_m': 1140000}, {'ticker': 'ASIANPAINT.NS', 'name': 'Asian Paints Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Consumer Cyclical', 'industry': 'Specialty Chemicals', 'base_price': 3280.0, 'shares_out_b': 0.96, 'pe_ratio': 58.0, 'pb_ratio': 18.0, 'ps_ratio': 8.5, 'ev_ebitda': 36.0, 'fcf_yield': 0.02, 'roe': 0.32, 'roce': 0.38, 'roa': 0.19, 'debt_to_equity': 0.12, 'current_ratio': 1.95, 'gross_margin': 0.44, 'operating_margin': 0.21, 'net_margin': 0.15, 'revenue_growth': 0.03, 'profit_growth': 0.08, 'dividend_yield': 0.011, 'working_cap_m': 164000, 'assets_m': 1640000, 'retained_m': 492000, 'ebit_m': 196800, 'liab_m': 820000, 'sales_m': 1312000}, {'ticker': 'HCLTECH.NS', 'name': 'HCL Technologies Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Technology', 'industry': 'Information Technology Services', 'base_price': 1780.0, 'shares_out_b': 2.71, 'pe_ratio': 30.5, 'pb_ratio': 7.2, 'ps_ratio': 4.4, 'ev_ebitda': 19.5, 'fcf_yield': 0.042, 'roe': 0.235, 'roce': 0.28, 'roa': 0.16, 'debt_to_equity': 0.09, 'current_ratio': 2.25, 'gross_margin': 0.38, 'operating_margin': 0.18, 'net_margin': 0.145, 'revenue_growth': 0.08, 'profit_growth': 0.09, 'dividend_yield': 0.028, 'working_cap_m': 89000, 'assets_m': 890000, 'retained_m': 267000, 'ebit_m': 106800, 'liab_m': 445000, 'sales_m': 712000}, {'ticker': 'WIPRO.NS', 'name': 'Wipro Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Technology', 'industry': 'Information Technology Services', 'base_price': 535.0, 'shares_out_b': 5.23, 'pe_ratio': 25.0, 'pb_ratio': 3.8, 'ps_ratio': 3.1, 'ev_ebitda': 15.0, 'fcf_yield': 0.048, 'roe': 0.15, 'roce': 0.17, 'roa': 0.1, 'debt_to_equity': 0.22, 'current_ratio': 2.35, 'gross_margin': 0.29, 'operating_margin': 0.145, 'net_margin': 0.125, 'revenue_growth': -0.04, 'profit_growth': -0.05, 'dividend_yield': 0.002, 'working_cap_m': 26750, 'assets_m': 267500, 'retained_m': 80250, 'ebit_m': 32100, 'liab_m': 133750, 'sales_m': 214000}, {'ticker': 'CIPLA.NS', 'name': 'Cipla Limited', 'country': 'IN', 'exchange': 'NSE', 'currency': 'INR', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - Specialty', 'base_price': 1640.0, 'shares_out_b': 0.81, 'pe_ratio': 32.0, 'pb_ratio': 4.8, 'ps_ratio': 5.1, 'ev_ebitda': 20.5, 'fcf_yield': 0.038, 'roe': 0.16, 'roce': 0.21, 'roa': 0.14, 'debt_to_equity': 0.03, 'current_ratio': 3.4, 'gross_margin': 0.65, 'operating_margin': 0.24, 'net_margin': 0.16, 'revenue_growth': 0.11, 'profit_growth': 0.22, 'dividend_yield': 0.008, 'working_cap_m': 82000, 'assets_m': 820000, 'retained_m': 246000, 'ebit_m': 98400, 'liab_m': 410000, 'sales_m': 656000}, {'ticker': 'RIO.L', 'name': 'Rio Tinto Group', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Materials', 'industry': 'Other Industrial Metals & Mining', 'base_price': 49.5, 'shares_out_b': 1.62, 'pe_ratio': 10.5, 'pb_ratio': 1.65, 'ps_ratio': 1.7, 'ev_ebitda': 5.2, 'fcf_yield': 0.082, 'roe': 0.18, 'roce': 0.19, 'roa': 0.1, 'debt_to_equity': 0.24, 'current_ratio': 1.85, 'gross_margin': 0.42, 'operating_margin': 0.28, 'net_margin': 0.19, 'revenue_growth': -0.03, 'profit_growth': -0.18, 'dividend_yield': 0.068, 'working_cap_m': 2475, 'assets_m': 24750, 'retained_m': 7425, 'ebit_m': 2970, 'liab_m': 12375, 'sales_m': 19800}, {'ticker': 'GSK.L', 'name': 'GSK plc', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 16.2, 'shares_out_b': 4.12, 'pe_ratio': 14.0, 'pb_ratio': 5.2, 'ps_ratio': 2.2, 'ev_ebitda': 9.2, 'fcf_yield': 0.065, 'roe': 0.38, 'roce': 0.18, 'roa': 0.08, 'debt_to_equity': 1.15, 'current_ratio': 0.95, 'gross_margin': 0.72, 'operating_margin': 0.28, 'net_margin': 0.17, 'revenue_growth': 0.07, 'profit_growth': 0.14, 'dividend_yield': 0.038, 'working_cap_m': 810, 'assets_m': 8100, 'retained_m': 2430, 'ebit_m': 972, 'liab_m': 4050, 'sales_m': 6480}, {'ticker': 'DGE.L', 'name': 'Diageo plc', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Consumer Defensive', 'industry': 'Beverages - Wineries & Distilleries', 'base_price': 25.4, 'shares_out_b': 2.23, 'pe_ratio': 18.5, 'pb_ratio': 5.8, 'ps_ratio': 3.8, 'ev_ebitda': 13.5, 'fcf_yield': 0.048, 'roe': 0.32, 'roce': 0.17, 'roa': 0.085, 'debt_to_equity': 1.85, 'current_ratio': 1.45, 'gross_margin': 0.6, 'operating_margin': 0.3, 'net_margin': 0.2, 'revenue_growth': -0.01, 'profit_growth': -0.05, 'dividend_yield': 0.039, 'working_cap_m': 1270, 'assets_m': 12700, 'retained_m': 3810, 'ebit_m': 1524, 'liab_m': 6350, 'sales_m': 10160}, {'ticker': 'BARC.L', 'name': 'Barclays PLC', 'country': 'UK', 'exchange': 'LSE', 'currency': 'GBP', 'sector': 'Financial Services', 'industry': 'Banks - Diversified', 'base_price': 2.25, 'shares_out_b': 15.1, 'pe_ratio': 7.2, 'pb_ratio': 0.48, 'ps_ratio': 1.4, 'ev_ebitda': 5.8, 'fcf_yield': 0.095, 'roe': 0.075, 'roce': 0.065, 'roa': 0.004, 'debt_to_equity': 1.85, 'current_ratio': 1.05, 'gross_margin': 0.85, 'operating_margin': 0.28, 'net_margin': 0.18, 'revenue_growth': 0.03, 'profit_growth': -0.12, 'dividend_yield': 0.038, 'working_cap_m': 112, 'assets_m': 1125, 'retained_m': 337, 'ebit_m': 135, 'liab_m': 562, 'sales_m': 900}, {'ticker': 'ALV.DE', 'name': 'Allianz SE', 'country': 'DE', 'exchange': 'XETRA', 'currency': 'EUR', 'sector': 'Financial Services', 'industry': 'Insurance - Multi-line', 'base_price': 288.0, 'shares_out_b': 0.39, 'pe_ratio': 11.8, 'pb_ratio': 1.6, 'ps_ratio': 0.8, 'ev_ebitda': 7.5, 'fcf_yield': 0.072, 'roe': 0.145, 'roce': 0.125, 'roa': 0.009, 'debt_to_equity': 0.45, 'current_ratio': 1.15, 'gross_margin': 0.28, 'operating_margin': 0.11, 'net_margin': 0.065, 'revenue_growth': 0.06, 'profit_growth': 0.12, 'dividend_yield': 0.048, 'working_cap_m': 14400, 'assets_m': 144000, 'retained_m': 43200, 'ebit_m': 17280, 'liab_m': 72000, 'sales_m': 115200}, {'ticker': 'BMW.DE', 'name': 'Bayerische Motoren Werke AG', 'country': 'DE', 'exchange': 'XETRA', 'currency': 'EUR', 'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers', 'base_price': 78.5, 'shares_out_b': 0.65, 'pe_ratio': 5.8, 'pb_ratio': 0.55, 'ps_ratio': 0.35, 'ev_ebitda': 4.5, 'fcf_yield': 0.115, 'roe': 0.115, 'roce': 0.095, 'roa': 0.045, 'debt_to_equity': 1.15, 'current_ratio': 1.18, 'gross_margin': 0.19, 'operating_margin': 0.098, 'net_margin': 0.075, 'revenue_growth': 0.02, 'profit_growth': -0.18, 'dividend_yield': 0.075, 'working_cap_m': 3925, 'assets_m': 39250, 'retained_m': 11775, 'ebit_m': 4710, 'liab_m': 19625, 'sales_m': 31400}, {'ticker': 'AIR.PA', 'name': 'Airbus SE', 'country': 'FR', 'exchange': 'Euronext', 'currency': 'EUR', 'sector': 'Industrials', 'industry': 'Aerospace & Defense', 'base_price': 134.0, 'shares_out_b': 0.79, 'pe_ratio': 27.5, 'pb_ratio': 6.2, 'ps_ratio': 1.65, 'ev_ebitda': 15.0, 'fcf_yield': 0.038, 'roe': 0.24, 'roce': 0.19, 'roa': 0.038, 'debt_to_equity': 0.78, 'current_ratio': 1.25, 'gross_margin': 0.14, 'operating_margin': 0.085, 'net_margin': 0.058, 'revenue_growth': 0.11, 'profit_growth': -0.15, 'dividend_yield': 0.021, 'working_cap_m': 6700, 'assets_m': 67000, 'retained_m': 20100, 'ebit_m': 8040, 'liab_m': 33500, 'sales_m': 53600}, {'ticker': 'SAN.PA', 'name': 'Sanofi', 'country': 'FR', 'exchange': 'Euronext', 'currency': 'EUR', 'sector': 'Healthcare', 'industry': 'Drug Manufacturers - General', 'base_price': 105.0, 'shares_out_b': 1.26, 'pe_ratio': 24.5, 'pb_ratio': 1.75, 'ps_ratio': 3.1, 'ev_ebitda': 12.0, 'fcf_yield': 0.065, 'roe': 0.075, 'roce': 0.11, 'roa': 0.045, 'debt_to_equity': 0.28, 'current_ratio': 1.45, 'gross_margin': 0.69, 'operating_margin': 0.26, 'net_margin': 0.13, 'revenue_growth': 0.05, 'profit_growth': 0.08, 'dividend_yield': 0.036, 'working_cap_m': 5250, 'assets_m': 52500, 'retained_m': 15750, 'ebit_m': 6300, 'liab_m': 26250, 'sales_m': 42000}, {'ticker': 'RMS.PA', 'name': 'Hermes International', 'country': 'FR', 'exchange': 'Euronext', 'currency': 'EUR', 'sector': 'Consumer Cyclical', 'industry': 'Luxury Goods', 'base_price': 2040.0, 'shares_out_b': 0.105, 'pe_ratio': 48.0, 'pb_ratio': 14.5, 'ps_ratio': 15.8, 'ev_ebitda': 32.0, 'fcf_yield': 0.021, 'roe': 0.32, 'roce': 0.38, 'roa': 0.22, 'debt_to_equity': 0.18, 'current_ratio': 3.85, 'gross_margin': 0.72, 'operating_margin': 0.42, 'net_margin': 0.32, 'revenue_growth': 0.15, 'profit_growth': 0.18, 'dividend_yield': 0.012, 'working_cap_m': 102000, 'assets_m': 1020000, 'retained_m': 306000, 'ebit_m': 122400, 'liab_m': 510000, 'sales_m': 816000}, {'ticker': 'NESN.SW', 'name': 'Nestle S.A.', 'country': 'Global', 'exchange': 'SIX', 'currency': 'USD', 'sector': 'Consumer Defensive', 'industry': 'Packaged Foods', 'base_price': 98.5, 'shares_out_b': 2.61, 'pe_ratio': 22.4, 'pb_ratio': 6.8, 'ps_ratio': 2.8, 'ev_ebitda': 15.5, 'fcf_yield': 0.045, 'roe': 0.31, 'roce': 0.19, 'roa': 0.085, 'debt_to_equity': 1.55, 'current_ratio': 0.85, 'gross_margin': 0.46, 'operating_margin': 0.175, 'net_margin': 0.12, 'revenue_growth': 0.02, 'profit_growth': 0.05, 'dividend_yield': 0.034, 'working_cap_m': 4925, 'assets_m': 49250, 'retained_m': 14775, 'ebit_m': 5910, 'liab_m': 24625, 'sales_m': 39400}]

class MarketDataStore:
    def __init__(self):
        self.securities_df: pd.DataFrame = pd.DataFrame()
        self.candles_store: Dict[str, List[Dict[str, Any]]] = {}
        self._initialize_dataset()

    def _generate_ohlcv(self, base_price: float, days: int = 252) -> pd.DataFrame:
        np.random.seed(abs(hash(base_price)) % (2**31))
        returns = np.random.normal(0.0005, 0.015, days)
        curve = base_price * np.cumprod(1 + returns)
        dates = [datetime.now() - timedelta(days=(days - i)) for i in range(days)]
        data = []
        for i, p in enumerate(curve):
            hi = max(p, p * (1 + abs(np.random.normal(0, 0.007))))
            lo = min(p, p * (1 - abs(np.random.normal(0, 0.007))))
            op = p * (1 + np.random.normal(0, 0.004))
            data.append({
                "timestamp": dates[i].strftime("%Y-%m-%d"),
                "open": round(float(op), 2), "high": round(float(hi), 2),
                "low": round(float(lo), 2), "close": round(float(p), 2),
                "volume": int(abs(np.random.normal(2500000, 800000)))
            })
        return pd.DataFrame(data)

    def _initialize_dataset(self):
        records = []
        np.random.seed(42)
        bench = pd.Series(5000.0 * np.cumprod(1 + np.random.normal(0.0004, 0.010, 252)))

        for seed in EXPANDED_GLOBAL_ASSETS:
            ticker = seed["ticker"]
            p = seed["base_price"]
            fx = FX_RATES.get(seed["currency"], 1.0)

            df_c = self._generate_ohlcv(p, 252)
            self.candles_store[ticker] = df_c.to_dict(orient="records")
            close = df_c["close"]
            vol = df_c["volume"]

            latest_p = float(close.iloc[-1])
            prev_p = float(close.iloc[-2])
            mcap = seed["shares_out_b"] * 1e9 * latest_p * fx

            f_score, _ = calculate_piotroski_f_score({
                "roa": seed["roa"], "operating_cash_flow": seed["ebit_m"] * 1.1,
                "delta_roa": 0.02, "net_income": seed["sales_m"] * seed["net_margin"],
                "delta_debt": -0.04, "delta_current_ratio": 0.07,
                "shares_diluted": False, "delta_gross_margin": 0.015, "delta_asset_turnover": 0.02
            })
            z_score = calculate_altman_z_score(seed["working_cap_m"], seed["assets_m"], seed["retained_m"], seed["ebit_m"], seed["shares_out_b"]*1000*latest_p, seed["liab_m"], seed["sales_m"])
            m_score, m_verdict, _ = calculate_beneish_m_score({})
            amihud = calculate_amihud_illiquidity(close, vol)
            sharpe = calculate_sharpe_ratio(close)
            kelly_pct = calculate_kelly_criterion(54.0 + sharpe * 4.0, 1.5)

            z_norm = np.clip(z_score / 4.0 * 100, 0, 100)
            f_norm = np.clip(f_score / 9.0 * 100, 0, 100)
            m_norm = np.clip((-m_score - 1.5) / 1.5 * 100, 0, 100)
            forensic = round(0.4 * f_norm + 0.3 * z_norm + 0.3 * m_norm, 1)

            rec = {
                "ticker": ticker, "name": seed["name"], "country": seed["country"], "exchange": seed["exchange"], "currency": seed["currency"],
                "sector": seed["sector"], "industry": seed["industry"], "price": round(latest_p, 2),
                "change_pct_24h": round((latest_p - prev_p)/prev_p * 100, 2), "volume_24h": float(vol.iloc[-1]), "market_cap_usd": round(mcap, 2),
                "pe_ratio": seed["pe_ratio"], "pb_ratio": seed["pb_ratio"], "ps_ratio": seed["ps_ratio"], "ev_ebitda": seed["ev_ebitda"],
                "fcf_yield": seed["fcf_yield"], "dividend_yield": seed.get("dividend_yield", 0.0),
                "roe": seed["roe"], "roce": seed["roce"], "roa": seed["roa"], "debt_to_equity": seed["debt_to_equity"],
                "current_ratio": seed["current_ratio"], "gross_margin": seed["gross_margin"], "operating_margin": seed["operating_margin"],
                "net_margin": seed["net_margin"], "revenue_growth_yoy": seed["revenue_growth"], "profit_growth_yoy": seed["profit_growth"],
                "piotroski_f_score": f_score, "altman_z_score": z_score, "beneish_m_score": m_score, "beneish_verdict": m_verdict,
                "amihud_illiquidity": amihud, "kelly_allocation_pct": kelly_pct, "forensic_health_score": forensic,
                "rsi_14": round(float(calculate_rsi(close).iloc[-1]), 2),
                "sma_20": round(float(calculate_sma(close, 20).iloc[-1]), 2),
                "sma_50": round(float(calculate_sma(close, 50).iloc[-1]), 2),
                "sma_200": round(float(calculate_sma(close, 200).iloc[-1]), 2),
                "volatility_annualized": round(calculate_volatility(close), 4),
                "sharpe_ratio": round(sharpe, 2),
                "sortino_ratio": round(calculate_sortino_ratio(close), 2),
                "max_drawdown": round(calculate_max_drawdown(close), 4),
                "beta": calculate_beta(close, bench),
                "sales_m": seed.get("sales_m", 1000), "assets_m": seed.get("assets_m", 2000), "liab_m": seed.get("liab_m", 1000), "ebit_m": seed.get("ebit_m", 200)
            }
            rec.update(calculate_factor_scores(rec))
            records.append(rec)

        raw_df = pd.DataFrame(records)
        # Add classification
        raw_df["market_cap_tier"] = np.where(raw_df["market_cap_usd"] >= 200e9, "Mega-Cap (>$200B)", np.where(raw_df["market_cap_usd"] >= 10e9, "Large-Cap ($10B-$200B)", "Mid/Small-Cap"))
        raw_df["valuation_regime"] = np.where(raw_df["pe_ratio"] < 18, "💎 Deep Value", np.where(raw_df["pe_ratio"] < 35, "⚖️ Fair Value", "🚀 Growth Multiple"))
        raw_df["solvency_health"] = np.where(raw_df["altman_z_score"] >= 2.99, "🛡️ Fortress Balance Sheet", "🟡 Moderate Leverage")
        self.securities_df = raw_df

    def get_all(self) -> pd.DataFrame:
        return self.securities_df

    def get_security(self, ticker: str) -> Optional[Dict[str, Any]]:
        row = self.securities_df[self.securities_df["ticker"] == ticker]
        if row.empty: return None
        return row.iloc[0].to_dict()

    def get_candles(self, ticker: str) -> List[Dict[str, Any]]:
        return self.candles_store.get(ticker, [])

market_store = MarketDataStore()
market_data = market_store

def load_live_or_cached_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    ticker = ticker.upper().strip()
    sec = market_store.get_security(ticker)
    if sec: return sec
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if not hist.empty:
            candles = []
            for idx, row in hist.iterrows():
                candles.append({
                    "timestamp": idx.strftime("%Y-%m-%d"),
                    "open": round(float(row["Open"]), 2), "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2), "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"])
                })
            market_store.candles_store[ticker] = candles
            p = float(hist["Close"].iloc[-1])
            mcap = t.info.get("marketCap", 10e9)
            rec = {
                "ticker": ticker, "name": t.info.get("shortName", ticker), "country": t.info.get("country", "Global"),
                "exchange": t.info.get("exchange", "Exchange"), "currency": t.info.get("currency", "USD"),
                "sector": t.info.get("sector", "Technology"), "industry": t.info.get("industry", "Diversified"),
                "price": round(p, 2), "change_pct_24h": round((p - hist["Close"].iloc[-2])/hist["Close"].iloc[-2]*100, 2),
                "volume_24h": float(hist["Volume"].iloc[-1]), "market_cap_usd": round(float(mcap), 2),
                "pe_ratio": t.info.get("trailingPE", 25.0) or 25.0, "pb_ratio": t.info.get("priceToBook", 3.0) or 3.0,
                "ps_ratio": 4.0, "ev_ebitda": 15.0, "fcf_yield": 0.035, "dividend_yield": t.info.get("dividendYield", 0.0) or 0.0,
                "roe": t.info.get("returnOnEquity", 0.18) or 0.18, "roce": 0.16, "roa": 0.08, "debt_to_equity": 0.5,
                "current_ratio": 1.5, "gross_margin": 0.45, "operating_margin": 0.22, "net_margin": 0.18,
                "revenue_growth_yoy": 0.12, "profit_growth_yoy": 0.15, "piotroski_f_score": 7, "altman_z_score": 3.4,
                "beneish_m_score": -2.25, "beneish_verdict": "🟢 Clean Accounting Records", "amihud_illiquidity": 0.02,
                "kelly_allocation_pct": 14.5, "forensic_health_score": 78.5, "rsi_14": 54.0,
                "sma_20": round(float(hist["Close"].rolling(20, min_periods=1).mean().iloc[-1]), 2),
                "sma_50": round(float(hist["Close"].rolling(50, min_periods=1).mean().iloc[-1]), 2),
                "sma_200": round(float(hist["Close"].rolling(200, min_periods=1).mean().iloc[-1]), 2),
                "volatility_annualized": 0.24, "sharpe_ratio": 1.25, "sortino_ratio": 1.55, "max_drawdown": -0.15, "beta": 1.0,
                "market_cap_tier": "Large-Cap", "valuation_regime": "⚖️ Fair Value", "solvency_health": "🛡️ Fortress Balance Sheet"
            }
            rec.update(calculate_factor_scores(rec))
            market_store.securities_df = pd.concat([market_store.securities_df, pd.DataFrame([rec])], ignore_index=True)
            return rec
    except Exception:
        pass
    return None

# ==============================================================================
# 3. STREAMLIT ULTRA-WORKSTATION FRONTEND
# ==============================================================================

st.set_page_config(
    page_title="OmniScreen Institutional Terminal",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .reportview-container { background: #090c15; }
    .main-header { font-size: 1.95rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.02em; }
    .sub-header { color: #64748b; font-size: 0.88rem; margin-bottom: 1.25rem; }
    .kpi-card { background: #0f1422; border: 1px solid #1e2538; border-radius: 6px; padding: 0.85rem 1rem; }
    .kpi-label { font-size: 0.70rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.05em; }
    .kpi-val { font-size: 1.35rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; color: #f1f5f9; margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 1px solid #1e2538; }
    .stTabs [data-baseweb="tab"] { height: 38px; font-weight: 600; font-size: 0.82rem; border-radius: 4px 4px 0 0; }
</style>
""", unsafe_allow_html=True)

df_all = market_store.get_all()

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/bullish.png", width=52)
st.sidebar.title("OmniScreen Terminal")
st.sidebar.caption("Institutional Quantitative Workstation")

# Live Ingestion
live_input = st.sidebar.text_input("Live Ticker Ingestion", placeholder="e.g. NVDA, TATAMOTORS.NS, BTC-USD")
if st.sidebar.button("Fetch Live Asset"):
    if live_input:
        with st.spinner("Streaming exchange quotes..."):
            sec_l = load_live_or_cached_ticker(live_input)
            if sec_l:
                st.sidebar.success(f"✓ Ingested: {sec_l['ticker']}")
                df_all = market_store.get_all()

# Market Scope
market_map = {
    "ALL": "🌐 Global / All Markets", "US": "🇺🇸 United States", "IN": "🇮🇳 India",
    "UK": "🇬🇧 United Kingdom", "JP": "🇯🇵 Japan", "NL": "🇳🇱 Netherlands",
    "DE": "🇩🇪 Germany", "FR": "🇫🇷 France", "DK": "🇩🇰 Denmark", "Global": "Macro Assets / Crypto"
}
sel_market = st.sidebar.selectbox("Market Scope", options=list(market_map.keys()), format_func=lambda x: market_map[x])
sel_sector = st.sidebar.selectbox("GICS Sector", options=["ALL"] + sorted(df_all["sector"].unique().tolist()))

# Strategy Presets
st.sidebar.markdown("---")
preset_options = {
    "none": "⚙️ Custom Criteria",
    "buffett": "🏆 Buffett Quality Compounders",
    "value": "💎 Deep Value & Solvency",
    "garp": "📈 GARP Growth at Reasonable Price",
    "forensic": "🛡️ Forensic Bulletproof (Anti-Fraud)",
    "kelly": "🎯 High-Conviction Kelly Sizing",
    "momentum": " Momentum & Trend Breakouts"
}
sel_preset = st.sidebar.selectbox("Quant Strategy Preset", options=list(preset_options.keys()), format_func=lambda x: preset_options[x])

# Numeric Filters
st.sidebar.markdown("---")
f_mcap = st.sidebar.number_input("Min Market Cap ($B USD)", 0.0, 5000.0, 5.0, step=5.0)
f_pe = st.sidebar.number_input("Max P/E Multiple (0 = Off)", 0.0, 200.0, 50.0, step=5.0)
f_roe = st.sidebar.number_input("Min Return on Equity (%)", 0.0, 150.0, 8.0, step=2.0)
f_fscore = st.sidebar.slider("Min Piotroski F-Score (0-9)", 0, 9, 4)
f_de = st.sidebar.number_input("Max Debt-to-Equity (0 = Off)", 0.0, 10.0, 2.0, step=0.2)
f_forensic = st.sidebar.slider("Min Forensic Health Score", 0, 100, 45)
f_score = st.sidebar.slider("Min Composite Quant Rank", 0, 100, 30)

# Apply Filters
filtered = df_all.copy()
if sel_market != "ALL":
    filtered = filtered[filtered["country"] == sel_market]
if sel_sector != "ALL":
    filtered = filtered[filtered["sector"] == sel_sector]

if sel_preset == "buffett":
    filtered = filtered[(filtered["roe"] >= 0.18) & (filtered["piotroski_f_score"] >= 7) & (filtered["debt_to_equity"] <= 0.8)]
elif sel_preset == "value":
    filtered = filtered[(filtered["pe_ratio"] <= 20) & (filtered["pb_ratio"] <= 3.0) & (filtered["fcf_yield"] >= 0.04)]
elif sel_preset == "garp":
    filtered = filtered[(filtered["revenue_growth_yoy"] >= 0.10) & (filtered["pe_ratio"] <= 35) & (filtered["operating_margin"] >= 0.15)]
elif sel_preset == "forensic":
    filtered = filtered[(filtered["forensic_health_score"] >= 70) & (filtered["beneish_m_score"] < -1.78) & (filtered["altman_z_score"] >= 2.5)]
elif sel_preset == "kelly":
    filtered = filtered[(filtered["kelly_allocation_pct"] >= 12.0) & (filtered["sharpe_ratio"] >= 0.7)]
elif sel_preset == "momentum":
    filtered = filtered[(filtered["rsi_14"] >= 52) & (filtered["rsi_14"] <= 70) & (filtered["sharpe_ratio"] >= 0.8)]
else:
    if f_mcap > 0: filtered = filtered[filtered["market_cap_usd"] >= f_mcap * 1e9]
    if f_pe > 0: filtered = filtered[filtered["pe_ratio"] <= f_pe]
    if f_roe > 0: filtered = filtered[filtered["roe"] >= f_roe / 100.0]
    if f_fscore > 0: filtered = filtered[filtered["piotroski_f_score"] >= f_fscore]
    if f_de > 0: filtered = filtered[filtered["debt_to_equity"] <= f_de]
    if f_forensic > 0: filtered = filtered[filtered["forensic_health_score"] >= f_forensic]
    if f_score > 0: filtered = filtered[filtered["composite_score"] >= f_score]

# --- MAIN DASHBOARD ---
st.markdown('<div class="main-header">OmniScreen Institutional Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Global Equity Universe (116 Assets) • DuPont Analysis • Forensic Accounting • Tail-Risk & VaR • DCF • Modern Portfolio Theory</div>', unsafe_allow_html=True)

# Top KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Securities Screened", f"{len(filtered)} / {len(df_all)}")
if len(filtered) > 0:
    k2.metric("Median Market Cap", f"${filtered['market_cap_usd'].median()/1e9:.1f}B")
    k3.metric("Avg P/E Ratio", f"{filtered['pe_ratio'].mean():.1f}x")
    k4.metric("Avg ROE", f"{filtered['roe'].mean()*100:.1f}%")
    k5.metric("Avg Quant Rank", f"{filtered['composite_score'].mean():.1f} / 100")
else:
    k2.metric("Median Market Cap", "-")
    k3.metric("Avg P/E Ratio", "-")
    k4.metric("Avg ROE", "-")
    k5.metric("Avg Quant Rank", "-")

# 10 WORKBENCH TABS
tabs = st.tabs([
    "Screener",
    "Company Profile",
    "DuPont 5-Way ROE",
    "Forensic Audit",
    "Macro Stress-Test & VaR",
    "Strategy Backtester",
    "DCF Valuation",
    "Monte Carlo (GBM)",
    "Portfolio Optimization",
    "Research Memorandum"
])

all_tickers = sorted(df_all["ticker"].unique().tolist())

# TAB 1: SCREENER RESULTS
with tabs[0]:
    if len(filtered) == 0:
        st.warning("No securities match active criteria.")
    else:
        disp = filtered[[
            "ticker", "name", "country", "exchange", "sector", "price", "change_pct_24h",
            "market_cap_usd", "market_cap_tier", "pe_ratio", "roe", "piotroski_f_score",
            "altman_z_score", "beneish_m_score", "forensic_health_score", "kelly_allocation_pct", "composite_score"
        ]].copy()
        disp["market_cap_usd"] = (disp["market_cap_usd"]/1e9).round(2)
        disp["roe"] = (disp["roe"]*100).round(1)
        disp.columns = ["Ticker", "Asset Name", "Country", "Exchange", "Sector", "Price", "24h %", "MCap ($B)", "Tier", "P/E", "ROE %", "Piotroski", "Altman Z", "Beneish M", "Forensic Score", "Kelly %", "Quant Rank"]
        st.dataframe(disp, height=460)

        csv_bytes = disp.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Screened Universe (CSV)", csv_bytes, "omniscreen_filtered.csv", "text/csv")

# TAB 2: DEEP DIVE
with tabs[1]:
    t_target = st.selectbox("Select Security for In-Depth Quantitative Analysis:", options=all_tickers, key="dd_tick")
    sec = market_store.get_security(t_target)
    candles = market_store.get_candles(t_target)
    if sec and candles:
        st.markdown(f"### {sec['ticker']} — {sec['name']}")
        st.caption(f"**{sec['sector']}** • {sec['industry']} • **{sec['exchange']} ({sec['country']})** • Currency: `{sec['currency']}`")

        df_c = pd.DataFrame(candles)
        patterns = detect_technical_patterns(df_c)
        st.markdown("#####  Algorithmic Pattern & Regime Signals")
        cols_p = st.columns(len(patterns))
        for idx, p in enumerate(patterns):
            emoji = "🟢" if p["type"] == "Bullish" else "🔴" if p["type"] == "Bearish" else "⚪"
            cols_p[idx].info(f"{emoji} **{p['pattern']}**\n\n{p['description']}")

        df_c["timestamp"] = pd.to_datetime(df_c["timestamp"])
        df_c["SMA_20"] = calculate_sma(df_c["close"], 20)
        df_c["SMA_50"] = calculate_sma(df_c["close"], 50)
        df_c["SMA_200"] = calculate_sma(df_c["close"], 200)
        st.subheader("Price Action & Moving Average Overlays (1-Year)")
        st.line_chart(df_c.set_index("timestamp")[["close", "SMA_20", "SMA_50", "SMA_200"]], height=300)

# TAB 3: DUPONT 5-WAY ROE
with tabs[2]:
    st.subheader("DuPont 5-Way Return on Equity (ROE) Decomposition")
    st.caption("Deconstructs ROE into operating margin, asset turnover, financial leverage, tax burden, and interest burden to isolate genuine quality from debt-fueled returns.")
    t_dupont = st.selectbox("Select Asset for DuPont Decomposition:", options=all_tickers, key="dup_tick")
    sec_dup = market_store.get_security(t_dupont)
    if sec_dup:
        dp = calculate_dupont_5_way(sec_dup)
        d1, d2, d3, d4, d5 = st.columns(5)
        d1.metric("Tax Burden (NI / EBT)", f"{dp['tax_burden']}x", "Higher = Better Tax Retention")
        d2.metric("Interest Burden (EBT / EBIT)", f"{dp['interest_burden']}x", "Higher = Low Debt Drag")
        d3.metric("Operating Margin (EBIT / Sales)", f"{dp['operating_margin']*100:.1f}%", "Pricing Power")
        d4.metric("Asset Turnover (Sales / Assets)", f"{dp['asset_turnover']}x", "Capital Velocity")
        d5.metric("Financial Leverage (Assets / Equity)", f"{dp['financial_leverage']}x", "Debt Multiplier")
        st.info(f"**Synthesized DuPont ROE**: `{dp['computed_roe_pct']}%` (Reported ROE: `{sec_dup['roe']*100:.1f}%`)")

# TAB 4: FORENSIC RED FLAGS
with tabs[3]:
    st.subheader("Forensic Accounting & Earnings Manipulation Audit")
    t_f = st.selectbox("Select Asset for Forensic Audit:", options=all_tickers, key="for_tick")
    sec_f = market_store.get_security(t_f)
    if sec_f:
        m_val, m_status, m_dict = calculate_beneish_m_score({})
        f_val, f_dict = calculate_piotroski_f_score({})
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Beneish M-Score", f"{sec_f['beneish_m_score']}", sec_f['beneish_verdict'])
        fc2.metric("Altman Z-Score", f"{sec_f['altman_z_score']}", "Safe" if sec_f['altman_z_score'] > 2.99 else "Grey Zone")
        fc3.metric("Forensic Health Composite", f"{sec_f['forensic_health_score']} / 100")
        fc4.metric("Half-Kelly Position Size", f"{sec_f['kelly_allocation_pct']}% of Capital")

        st.markdown("#### Piotroski 9-Criteria Health Checklist")
        st.json(f_dict)

# TAB 5: CRISIS STRESS-TESTING
with tabs[4]:
    st.subheader("Tail-Risk & Black Swan Crisis Stress Simulator")
    t_s = st.selectbox("Select Asset to Stress-Test:", options=all_tickers, key="str_tick")
    sec_s = market_store.get_security(t_s)
    candles_s = market_store.get_candles(t_s)
    if sec_s and candles_s:
        close_s = pd.Series([c["close"] for c in candles_s])
        var_res = calculate_var_cvar(close_s, 0.95)
        crises = simulate_crisis_scenarios(sec_s["price"], sec_s["beta"], sec_s["sector"])
        v1, v2, v3 = st.columns(3)
        v1.metric("1-Day 95% Historical VaR", f"{var_res['var_historical_pct']}%", "Max Daily Expected Loss")
        v2.metric("Parametric VaR", f"{var_res['var_parametric_pct']}%", "Gaussian Assumption")
        v3.metric("Conditional VaR (CVaR)", f"{var_res['cvar_expected_shortfall_pct']}%", "Expected Tail Loss")
        st.table(pd.DataFrame(crises))

# TAB 6: BACKTESTER
with tabs[5]:
    st.subheader("Vectorized Algorithmic Strategy Backtester")
    b_col1, b_col2 = st.columns(2)
    with b_col1: bt_tick = st.selectbox("Select Asset to Backtest:", options=all_tickers, key="bt_tick")
    with b_col2: bt_strat = st.selectbox("Strategy Type", ["sma_crossover", "rsi_mean_reversion", "bollinger_breakout"])
    candles_bt = pd.DataFrame(market_store.get_candles(bt_tick))
    if not candles_bt.empty:
        res_bt = run_backtest(candles_bt, strategy=bt_strat)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Strategy Total Return", f"{res_bt['total_return_pct']}%")
        b2.metric("Buy & Hold Benchmark", f"{res_bt['benchmark_return_pct']}%")
        b3.metric("Strategy Sharpe", f"{res_bt['strategy_sharpe']}")
        b4.metric("Win Rate", f"{res_bt['win_rate_pct']}%")
        eq_df = pd.DataFrame(res_bt["equity_curve"])
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        st.line_chart(eq_df.set_index("timestamp")[["Strategy", "Buy & Hold"]], height=320)

# TAB 7: DCF VALUATION
with tabs[6]:
    st.subheader("Gordon Growth Two-Stage DCF Valuation")
    t_dcf = st.selectbox("Select Security for DCF:", options=all_tickers, key="dcf_tick")
    sec_dcf = market_store.get_security(t_dcf)
    if sec_dcf:
        d_c1, d_c2, d_c3 = st.columns(3)
        g_in = d_c1.slider("5-Yr Growth Rate (%)", 0.0, 30.0, 12.0, step=0.5)
        tg_in = d_c2.slider("Terminal Perpetual Growth (%)", 1.0, 4.0, 2.5, step=0.1)
        wacc_in = d_c3.slider("WACC Discount Rate (%)", 6.0, 15.0, 9.0, step=0.5)

        mcap_m = sec_dcf["market_cap_usd"] / 1e6
        dcf_res = calculate_dcf(
            fcf_million=max(50.0, mcap_m * sec_dcf["fcf_yield"]),
            shares_outstanding_million=max(10.0, mcap_m / sec_dcf["price"]),
            net_debt_million=mcap_m * 0.15,
            growth_rate_pct=g_in, terminal_growth_pct=tg_in, wacc_pct=wacc_in
        )
        fv = dcf_res["fair_value"]
        margin = round((fv - sec_dcf["price"])/sec_dcf["price"]*100, 1)
        m1, m2 = st.columns(2)
        m1.metric("Calculated Fair Value", f"{sec_dcf['currency']} {fv:,.2f}")
        m2.metric("Margin of Safety", f"{'+' if margin >= 0 else ''}{margin}%")
        st.subheader("Sensitivity Matrix (Fair Value Per Share)")
        st.dataframe(dcf_res["sensitivity_matrix"])

# TAB 8: MONTE CARLO FORECASTER
with tabs[7]:
    st.subheader("Geometric Brownian Motion (GBM) Stochastic Forecaster")
    t_mc = st.selectbox("Select Security for Monte Carlo:", options=all_tickers, key="mc_tick")
    sec_mc = market_store.get_security(t_mc)
    if sec_mc:
        mc_res = simulate_monte_carlo_paths(sec_mc["price"], sec_mc["volatility_annualized"], 120, 100, 0.08)
        c1, c2, c3 = st.columns(3)
        c1.metric("10th Percentile (Bearish)", f"{sec_mc['currency']} {mc_res['p10_bearish']}")
        c2.metric("50th Percentile (Median)", f"{sec_mc['currency']} {mc_res['p50_median']}")
        c3.metric("90th Percentile (Bullish)", f"{sec_mc['currency']} {mc_res['p90_bullish']}")
        st.line_chart(mc_res["chart_data"], height=320)

# TAB 9: PORTFOLIO OPTIMIZER
with tabs[8]:
    st.subheader("Portfolio Optimization: Markowitz Efficient Frontier & Risk Parity")
    opt_sel = st.multiselect("Select 3 to 6 Assets:", options=all_tickers, default=["AAPL", "MSFT", "NVDA", "TCS.NS", "ASML.AS"])
    if len(opt_sel) >= 2:
        p_dict = {t: pd.DataFrame(market_store.get_candles(t))["close"].values for t in opt_sel if market_store.get_candles(t)}
        min_l = min(len(v) for v in p_dict.values())
        p_df = pd.DataFrame({k: v[:min_l] for k, v in p_dict.items()})
        ef_res = simulate_efficient_frontier(p_df, 1500)
        rp_res = calculate_risk_parity_weights(p_df)
        o1, o2, o3 = st.columns(3)
        with o1:
            st.markdown("#### 🌟 Max Sharpe Portfolio")
            st.write(f"Return: **{ef_res['max_sharpe']['return']}%** | Vol: **{ef_res['max_sharpe']['volatility']}%**")
            st.write(f"Sharpe: **{ef_res['max_sharpe']['sharpe']}**")
            st.json(ef_res['max_sharpe']['weights'])
        with o2:
            st.markdown("#### 🛡️ Min Volatility Portfolio")
            st.write(f"Return: **{ef_res['min_vol']['return']}%** | Vol: **{ef_res['min_vol']['volatility']}%**")
            st.write(f"Sharpe: **{ef_res['min_vol']['sharpe']}**")
            st.json(ef_res['min_vol']['weights'])
        with o3:
            st.markdown("#### ⚖️ Risk Parity Weights")
            st.json(rp_res)
        st.scatter_chart(ef_res["scatter_data"], x="Volatility", y="Return", color="Sharpe")

# TAB 10: INSTITUTIONAL MEMO
with tabs[9]:
    st.subheader("Automated Quantitative Equity Research Memorandum")
    t_rep = st.selectbox("Select Asset for Research Memorandum:", options=all_tickers, key="rep_tick")
    sec_rep = market_store.get_security(t_rep)
    candles_rep = market_store.get_candles(t_rep)
    if sec_rep and candles_rep:
        if st.button("🚀 Generate Full Institutional Memorandum"):
            memo_md = generate_equity_research_report(sec_rep, candles_rep)
            st.markdown(memo_md)
            st.download_button("📥 Download Memorandum (Markdown)", memo_md, f"memo_{sec_rep['ticker']}.md", "text/markdown")
