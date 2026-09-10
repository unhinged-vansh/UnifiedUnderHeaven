import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from models.schemas import (
    ComparisonOperator, LeafCondition, LogicalGroup, ScreenerQuery,
    ScreenerResponse, SecuritySummary, MetricMetadata
)

METRIC_CATALOG: List[MetricMetadata] = [
    # General
    MetricMetadata(id="market_cap_usd", name="Market Cap (USD)", category="General", type="currency", description="Total equity valuation normalized to USD", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="price", name="Current Price", category="General", type="currency", description="Latest closing or real-time trading price", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="change_pct_24h", name="24h Change %", category="General", type="percentage", description="Price change percentage over past 24 hours", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="pct_from_52w_high", name="% from 52-Week High", category="General", type="percentage", description="Percentage distance below the 52-week peak", default_operator=ComparisonOperator.GT),
    
    # Valuation & Dividends
    MetricMetadata(id="pe_ratio", name="P/E Ratio", category="Valuation", type="numeric", description="Price to Earnings Multiple (TTM)", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="pb_ratio", name="P/B Ratio", category="Valuation", type="numeric", description="Price to Book Multiple", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="ps_ratio", name="P/S Ratio", category="Valuation", type="numeric", description="Price to Sales Multiple", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="ev_ebitda", name="EV / EBITDA", category="Valuation", type="numeric", description="Enterprise Value over EBITDA", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="fcf_yield", name="Free Cash Flow Yield", category="Valuation", type="percentage", description="Free cash flow as percentage of market cap", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="dividend_yield", name="Dividend Yield", category="Valuation", type="percentage", description="Annual cash dividends paid per share divided by price", default_operator=ComparisonOperator.GT),
    
    # Profitability & Margins
    MetricMetadata(id="roe", name="Return on Equity (ROE)", category="Profitability", type="percentage", description="Net income over shareholders equity", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="roce", name="Return on Capital Employed (ROCE)", category="Profitability", type="percentage", description="EBIT over capital employed", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="roa", name="Return on Assets (ROA)", category="Profitability", type="percentage", description="Net income over total assets", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="gross_margin", name="Gross Margin", category="Profitability", type="percentage", description="Gross profit over total revenues", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="operating_margin", name="Operating Margin", category="Profitability", type="percentage", description="Operating income over total revenues", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="net_margin", name="Net Margin", category="Profitability", type="percentage", description="Net profit over total revenues", default_operator=ComparisonOperator.GT),
    
    # Growth
    MetricMetadata(id="revenue_growth_yoy", name="Revenue Growth (YoY)", category="Growth", type="percentage", description="Year-over-year revenue expansion rate", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="profit_growth_yoy", name="Net Profit Growth (YoY)", category="Growth", type="percentage", description="Year-over-year earnings expansion rate", default_operator=ComparisonOperator.GT),
    
    # Quality & Solvency
    MetricMetadata(id="debt_to_equity", name="Debt to Equity", category="Quality", type="numeric", description="Total financial liabilities divided by total equity", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="current_ratio", name="Current Ratio", category="Quality", type="numeric", description="Current assets divided by current liabilities", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="piotroski_f_score", name="Piotroski F-Score", category="Quality", type="integer", description="Financial health score ranging from 0 (distressed) to 9 (strong)", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="altman_z_score", name="Altman Z-Score", category="Quality", type="numeric", description="Bankruptcy distress indicator (> 2.99 safe, < 1.81 distressed)", default_operator=ComparisonOperator.GT),
    
    # Technical Indicators
    MetricMetadata(id="rsi_14", name="RSI (14)", category="Technical", type="numeric", description="14-period Relative Strength Index", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="volatility_annualized", name="Annualized Volatility", category="Technical", type="percentage", description="Standard deviation of annualized returns", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="sharpe_ratio", name="Sharpe Ratio", category="Technical", type="numeric", description="Risk-adjusted excess return ratio", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="sortino_ratio", name="Sortino Ratio", category="Technical", type="numeric", description="Downside risk-adjusted excess return ratio", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="max_drawdown", name="Max Drawdown (1Y)", category="Technical", type="percentage", description="Maximum peak-to-trough decline over 1 year", default_operator=ComparisonOperator.GT),
    MetricMetadata(id="beta", name="Beta (vs Benchmark)", category="Technical", type="numeric", description="Systematic volatility relative to market benchmark", default_operator=ComparisonOperator.LT),

    # Quant Factor Scores
    MetricMetadata(id="beneish_m_score", name="Beneish M-Score (Fraud Risk)", category="Quality", type="numeric", description="Earnings manipulation risk metric (< -1.78 is safe)", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="forensic_health_score", name="Forensic Health Score (0-100)", category="Quality", type="numeric", description="Combined Piotroski, Altman Z, and Beneish health score", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="amihud_illiquidity", name="Amihud Illiquidity Ratio", category="Technical", type="numeric", description="Measures price impact per traded dollar (lower = more liquid)", default_operator=ComparisonOperator.LT),
    MetricMetadata(id="kelly_allocation_pct", name="Kelly Position Size (%)", category="Factor Models", type="numeric", description="Mathematically optimal capital sizing percentage", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="composite_score", name="Composite Quant Rank (0-100)", category="Factor Models", type="numeric", description="Weighted multi-factor score across quality, value, momentum, growth", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="quality_score", name="Quality Factor Score", category="Factor Models", type="numeric", description="Pillars: ROE, Piotroski, Altman Z, conservative leverage", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="value_score", name="Value Factor Score", category="Factor Models", type="numeric", description="Pillars: P/E, P/B, FCF Yield discount rank", default_operator=ComparisonOperator.GTE),
    MetricMetadata(id="momentum_score", name="Momentum Factor Score", category="Factor Models", type="numeric", description="Pillars: Bullish RSI strength, Sharpe ratio, Trend", default_operator=ComparisonOperator.GTE)
]

PRESET_SCREENS: Dict[str, Dict[str, Any]] = {
    "forensic_bulletproof": {
        "title": "🛡️ Forensic Bulletproof (Anti-Fraud & Solvency)",
        "description": "Companies with the cleanest accounting records (low Beneish M-Score) and safe balance sheets.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "forensic_health_score", "operator": "gte", "value": 70.0},
                {"field": "beneish_m_score", "operator": "lt", "value": -1.78},
                {"field": "altman_z_score", "operator": "gt", "value": 2.5}
            ]
        },
        "sort_by": "forensic_health_score",
        "sort_direction": "desc"
    },
    "kelly_conviction": {
        "title": "🎯 High-Conviction Kelly Sizing",
        "description": "Assets with superior historical risk-reward justifying aggressive fractional Kelly capital allocations.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "kelly_allocation_pct", "operator": "gte", "value": 12.0},
                {"field": "sharpe_ratio", "operator": "gt", "value": 0.7}
            ]
        },
        "sort_by": "kelly_allocation_pct",
        "sort_direction": "desc"
    },
    "buffett_quality": {
        "title": "🏆 Buffett Quality Compounders",
        "description": "High return on equity, pristine Piotroski F-score, and low debt-to-equity leverage.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "roe", "operator": "gt", "value": 0.18},
                {"field": "piotroski_f_score", "operator": "gte", "value": 7},
                {"field": "debt_to_equity", "operator": "lt", "value": 0.80}
            ]
        },
        "sort_by": "roe",
        "sort_direction": "desc"
    },
    "deep_value": {
        "title": "💎 Deep Value & Solvency",
        "description": "Bargain valuation multiples coupled with safe solvency and high free cash flow yield.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "pe_ratio", "operator": "lt", "value": 20.0},
                {"field": "pb_ratio", "operator": "lt", "value": 3.0},
                {"field": "fcf_yield", "operator": "gt", "value": 0.04},
                {"field": "altman_z_score", "operator": "gt", "value": 2.0}
            ]
        },
        "sort_by": "fcf_yield",
        "sort_direction": "desc"
    },
    "garp": {
        "title": "📈 Growth at a Reasonable Price (GARP)",
        "description": "Double-digit revenue growth combined with reasonable earnings multiples and healthy margins.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "revenue_growth_yoy", "operator": "gt", "value": 0.10},
                {"field": "pe_ratio", "operator": "lt", "value": 35.0},
                {"field": "operating_margin", "operator": "gt", "value": 0.15}
            ]
        },
        "sort_by": "revenue_growth_yoy",
        "sort_direction": "desc"
    },
    "momentum_breakout": {
        "title": "⚡ Momentum & Trend Breakout",
        "description": "Stocks displaying strong technical strength, high Sharpe ratio, and superior momentum scores.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "rsi_14", "operator": "gte", "value": 52.0},
                {"field": "rsi_14", "operator": "lte", "value": 70.0},
                {"field": "sharpe_ratio", "operator": "gt", "value": 0.8}
            ]
        },
        "sort_by": "sharpe_ratio",
        "sort_direction": "desc"
    },
    "oversold_reversal": {
        "title": "🎯 Oversold Quality Reversals",
        "description": "High-quality companies temporarily beaten down into oversold RSI territory.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "rsi_14", "operator": "lt", "value": 45.0},
                {"field": "piotroski_f_score", "operator": "gte", "value": 6}
            ]
        },
        "sort_by": "rsi_14",
        "sort_direction": "asc"
    },
    "dividend_income": {
        "title": "💰 High-Yield Cash Cows",
        "description": "Attractive dividend yields supported by strong free cash flow and conservative leverage.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "dividend_yield", "operator": "gte", "value": 0.015},
                {"field": "fcf_yield", "operator": "gte", "value": 0.03},
                {"field": "debt_to_equity", "operator": "lte", "value": 1.5}
            ]
        },
        "sort_by": "dividend_yield",
        "sort_direction": "desc"
    },
    "quant_leaders": {
        "title": "🌟 Institutional Quant Leaders",
        "description": "Securities ranking in the top tier across the composite multi-factor quantitative model.",
        "filters": {
            "logic": "AND",
            "conditions": [
                {"field": "composite_score", "operator": "gte", "value": 65.0}
            ]
        },
        "sort_by": "composite_score",
        "sort_direction": "desc"
    }
}

class ScreenerEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate_node(self, node: Union[LogicalGroup, LeafCondition], sub_df: pd.DataFrame) -> pd.Series:
        if isinstance(node, LeafCondition):
            col = node.field
            if col not in sub_df.columns:
                raise ValueError(f"Unknown metric field: {col}")
            
            series = sub_df[col]
            val = node.value
            op = node.operator

            if op == ComparisonOperator.EQ:
                return series == val
            elif op == ComparisonOperator.NEQ:
                return series != val
            elif op == ComparisonOperator.GT:
                return series > float(val)
            elif op == ComparisonOperator.GTE:
                return series >= float(val)
            elif op == ComparisonOperator.LT:
                return series < float(val)
            elif op == ComparisonOperator.LTE:
                return series <= float(val)
            elif op == ComparisonOperator.IN:
                return series.isin(val if isinstance(val, list) else [val])
            elif op == ComparisonOperator.BETWEEN:
                low, high = val
                return series.between(float(low), float(high))
            elif op == ComparisonOperator.CONTAINS:
                return series.astype(str).str.contains(str(val), case=False, na=False)
            else:
                raise ValueError(f"Unsupported operator: {op}")

        elif isinstance(node, LogicalGroup):
            if not node.conditions:
                return pd.Series(True, index=sub_df.index)

            results = [self.evaluate_node(c, sub_df) for c in node.conditions]

            if node.logic == "AND":
                res = results[0]
                for r in results[1:]:
                    res = res & r
                return res
            elif node.logic == "OR":
                res = results[0]
                for r in results[1:]:
                    res = res | r
                return res
            elif node.logic == "NOT":
                return ~results[0]
            else:
                raise ValueError(f"Unsupported logic: {node.logic}")

        raise ValueError("Invalid AST Node")

    def execute(self, query: ScreenerQuery) -> ScreenerResponse:
        filtered_df = self.df.copy()

        # Market filter
        if query.market and query.market.upper() != "ALL":
            filtered_df = filtered_df[filtered_df["country"].str.upper() == query.market.upper()]

        # Sector filter
        if query.sector and query.sector.upper() != "ALL":
            filtered_df = filtered_df[filtered_df["sector"].str.lower() == query.sector.lower()]

        # AST Filter execution
        if query.filters and query.filters.conditions:
            mask = self.evaluate_node(query.filters, filtered_df)
            filtered_df = filtered_df[mask]

        # Sorting
        sort_col = query.sort_by if query.sort_by in filtered_df.columns else "market_cap_usd"
        ascending = (query.sort_direction.lower() == "asc")
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)

        total_matches = len(filtered_df)
        page_size = query.page_size
        total_pages = max(1, (total_matches + page_size - 1) // page_size)
        page = min(query.page, total_pages)

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_slice = filtered_df.iloc[start_idx:end_idx]

        records = [SecuritySummary(**row) for row in page_slice.to_dict(orient="records")]

        return ScreenerResponse(
            total_matches=total_matches,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            data=records
        )
