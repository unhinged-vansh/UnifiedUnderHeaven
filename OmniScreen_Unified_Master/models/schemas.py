from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ComparisonOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    BETWEEN = "between"
    CONTAINS = "contains"

class LeafCondition(BaseModel):
    field: str
    operator: ComparisonOperator
    value: Any

class LogicalGroup(BaseModel):
    logic: str = "AND"
    conditions: List[Union["LogicalGroup", LeafCondition]]

LogicalGroup.update_forward_refs()

class ScreenerQuery(BaseModel):
    filters: Optional[LogicalGroup] = None
    market: Optional[str] = "ALL"
    sector: Optional[str] = "ALL"
    sort_by: str = "market_cap_usd"
    sort_direction: str = "desc"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=500)

class OHLCVCandle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

class SecuritySummary(BaseModel):
    ticker: str
    name: str
    country: str
    exchange: str
    currency: str
    sector: str
    industry: str
    price: float
    change_pct_24h: float
    volume_24h: float
    market_cap_usd: float
    
    # 52-Week Range
    high_52w: Optional[float] = Field(default=None, alias="52w_high")
    low_52w: Optional[float] = Field(default=None, alias="52w_low")
    pct_from_52w_high: Optional[float] = None

    # Valuation & Yield
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    ev_ebitda: Optional[float] = None
    fcf_yield: Optional[float] = None
    dividend_yield: Optional[float] = None
    
    # Profitability & Efficiency
    roe: Optional[float] = None
    roce: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    profit_growth_yoy: Optional[float] = None
    
    # Solvency & Quality
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    piotroski_f_score: Optional[int] = None
    altman_z_score: Optional[float] = None
    
    # Technical Indicators
    rsi_14: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    volatility_annualized: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    beta: Optional[float] = None

    # Quant Factor Scores (0 - 100)
    value_score: Optional[float] = None
    quality_score: Optional[float] = None
    momentum_score: Optional[float] = None
    growth_score: Optional[float] = None
    composite_score: Optional[float] = None
    beneish_m_score: Optional[float] = None
    amihud_illiquidity: Optional[float] = None
    kelly_allocation_pct: Optional[float] = None
    forensic_health_score: Optional[float] = None

    class Config:
        allow_population_by_field_name = True

class ScreenerResponse(BaseModel):
    total_matches: int
    page: int
    page_size: int
    total_pages: int
    data: List[SecuritySummary]

class MetricMetadata(BaseModel):
    id: str
    name: str
    category: str
    type: str
    description: str
    default_operator: ComparisonOperator
