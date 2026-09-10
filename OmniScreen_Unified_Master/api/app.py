from fastapi import FastAPI, HTTPException, Request, Query, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, List, Optional
import os

from data.market_store import market_store, load_live_or_cached_ticker
from engine.screener_engine import ScreenerEngine, METRIC_CATALOG, PRESET_SCREENS
from models.schemas import (
    ScreenerQuery, ScreenerResponse, SecuritySummary, MetricMetadata
)

app = FastAPI(
    title="OmniScreen: Global Institutional Stock Screener API",
    description="Enterprise-grade quantitative screener, forensic accounting detector, and algorithmic analytics engine.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = ScreenerEngine(market_store.get_all())
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

# ==============================================================================
# CUSTOM ERROR HANDLERS (404 & 500)
# ==============================================================================
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    path = request.url.path
    if path.startswith("/api/"):
        detail_msg = getattr(exc, "detail", "Endpoint or asset not found")
        return JSONResponse(
            status_code=404,
            content={"status": "error", "code": 404, "message": "Resource or security not found", "detail": str(detail_msg)}
        )
    error_file = os.path.join(STATIC_DIR, "404.html")
    if os.path.exists(error_file):
        return FileResponse(error_file, status_code=404)
    return HTMLResponse("<h1>404 Not Found</h1>", status_code=404)

@app.exception_handler(500)
async def custom_500_handler(request: Request, exc: Exception):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        error_file = os.path.join(STATIC_DIR, "500.html")
        if os.path.exists(error_file):
            return FileResponse(error_file, status_code=500)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "code": 500, "message": "Internal quantitative engine exception", "detail": str(exc)}
    )

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/api/v1/health", tags=["System"])
async def health_check():
    df = market_store.get_all()
    return {
        "status": "healthy",
        "universe_size": len(df),
        "markets": sorted(df["country"].unique().tolist()),
        "exchanges": sorted(df["exchange"].unique().tolist()),
        "sectors": sorted(df["sector"].unique().tolist()),
        "quality_report": market_store.quality_report
    }

@app.get("/api/v1/markets", tags=["Metadata"])
async def get_markets():
    df = market_store.get_all()
    return {
        "countries": [
            {"code": "ALL", "name": "Global / Multi-Asset"},
            {"code": "US", "name": "United States (NYSE / NASDAQ)"},
            {"code": "IN", "name": "India (NSE / BSE)"},
            {"code": "UK", "name": "United Kingdom (LSE)"},
            {"code": "JP", "name": "Japan (TSE)"},
            {"code": "NL", "name": "Netherlands (Euronext)"},
            {"code": "DE", "name": "Germany (XETRA)"},
            {"code": "FR", "name": "France (Euronext)"},
            {"code": "DK", "name": "Denmark (OMX)"},
            {"code": "Global", "name": "Macro Indices & Commodities"}
        ],
        "sectors": sorted(df["sector"].unique().tolist()),
        "industries": sorted(df["industry"].unique().tolist()),
        "market_cap_tiers": ["Mega-Cap (>$200B)", "Large-Cap ($10B-$200B)", "Mid-Cap ($2B-$10B)", "Small-Cap (<$2B)"]
    }

@app.get("/api/v1/metrics", response_model=List[MetricMetadata], tags=["Metadata"])
async def get_metrics():
    return METRIC_CATALOG

@app.get("/api/v1/presets", tags=["Screener"])
async def get_presets():
    return PRESET_SCREENS

@app.post("/api/v1/screener/run", response_model=ScreenerResponse, tags=["Screener"])
async def run_screener(query: ScreenerQuery):
    try:
        current_engine = ScreenerEngine(market_store.get_all())
        return current_engine.execute(query)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

@app.get("/api/v1/securities", response_model=List[SecuritySummary], tags=["Securities"])
async def search_securities(
    q: Optional[str] = Query(default=None, description="Search by ticker or name"),
    country: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100)
):
    df = market_store.get_all()
    if country and country.upper() != "ALL":
        df = df[df["country"].str.upper() == country.upper()]
    if q:
        mask = df["ticker"].str.contains(q, case=False) | df["name"].str.contains(q, case=False)
        df = df[mask]
    records = df.head(limit).to_dict(orient="records")
    return [SecuritySummary(**r) for r in records]

@app.get("/api/v1/securities/{ticker}", response_model=SecuritySummary, tags=["Securities"])
async def get_security_detail(ticker: str):
    sec = load_live_or_cached_ticker(ticker.upper())
    if not sec:
        raise HTTPException(status_code=404, detail=f"Security '{ticker}' not found in global database.")
    return SecuritySummary(**sec)

@app.get("/api/v1/securities/{ticker}/candles", tags=["Securities"])
async def get_security_candles(ticker: str):
    candles = market_store.get_candles(ticker.upper())
    if not candles:
        sec = load_live_or_cached_ticker(ticker.upper())
        candles = market_store.get_candles(ticker.upper())
    if not candles:
        raise HTTPException(status_code=404, detail=f"No OHLCV history found for {ticker}")
    return {"ticker": ticker.upper(), "count": len(candles), "candles": candles}

# Mount static files for the full web application
if os.path.exists(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
