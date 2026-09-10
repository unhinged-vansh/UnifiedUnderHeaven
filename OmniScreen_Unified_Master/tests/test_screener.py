import unittest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.quant import (
    calculate_rsi, calculate_sma, calculate_macd,
    calculate_piotroski_f_score, calculate_altman_z_score,
    calculate_volatility, calculate_sharpe_ratio, calculate_factor_scores
)
from data.market_store import market_store
from engine.screener_engine import ScreenerEngine, PRESET_SCREENS
from models.schemas import (
    ScreenerQuery, LogicalGroup, LeafCondition, ComparisonOperator
)
from api.app import app

class TestComprehensiveScreener(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.df = market_store.get_all()
        self.engine = ScreenerEngine(self.df)

    def test_factor_scoring(self):
        sample = self.df.iloc[0].to_dict()
        scores = calculate_factor_scores(sample)
        self.assertIn("composite_score", scores)
        self.assertIn("quality_score", scores)
        self.assertIn("value_score", scores)
        self.assertTrue(0 <= scores["composite_score"] <= 100)

    def test_universe_coverage(self):
        countries = set(self.df["country"].unique())
        self.assertTrue({"US", "IN", "UK", "JP", "NL", "DE", "FR", "DK"}.issubset(countries))
        self.assertGreaterEqual(len(self.df), 25)

    def test_presets_execution(self):
        for preset_id, preset_data in PRESET_SCREENS.items():
            conds = [
                LeafCondition(field=c["field"], operator=ComparisonOperator(c["operator"]), value=c["value"])
                for c in preset_data["filters"]["conditions"]
            ]
            q = ScreenerQuery(
                filters=LogicalGroup(logic="AND", conditions=conds),
                sort_by=preset_data["sort_by"],
                sort_direction=preset_data["sort_direction"]
            )
            res = self.engine.execute(q)
            self.assertIsInstance(res.total_matches, int)

    def test_api_screener(self):
        res = self.client.post("/api/v1/screener/run", json={
            "market": "ALL",
            "sort_by": "composite_score",
            "sort_direction": "desc",
            "filters": {
                "logic": "AND",
                "conditions": [
                    {"field": "market_cap_usd", "operator": "gt", "value": 1e10}
                ]
            }
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreater(data["total_matches"], 0)

if __name__ == "__main__":
    unittest.main()
