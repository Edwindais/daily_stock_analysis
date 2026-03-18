# -*- coding: utf-8 -*-
"""
Contract tests for get_stock_info tool output semantics.
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agent.tools.data_tools import (
    _handle_get_a_share_flow_context,
    _handle_get_stock_event_context,
    _handle_get_stock_info,
)


class _DummyManager:
    def __init__(self):
        self._context = {
            "market": "cn",
            "status": "partial",
            "coverage": {
                "valuation": "ok",
                "growth": "not_supported",
                "earnings": "not_supported",
                "institution": "not_supported",
                "capital_flow": "not_supported",
                "dragon_tiger": "not_supported",
                "boards": "ok",
                "announcements": "ok",
                "northbound": "ok",
                "margin": "ok",
                "shareholder_count": "ok",
            },
            "valuation": {
                "status": "ok",
                "data": {
                    "pe_ratio": 12.3,
                    "pb_ratio": 2.1,
                    "total_mv": 1.0e11,
                    "circ_mv": 7.0e10,
                },
            },
            "growth": {"status": "not_supported", "data": {}},
            "earnings": {"status": "not_supported", "data": {}},
            "institution": {"status": "not_supported", "data": {}},
            "capital_flow": {"status": "not_supported", "data": {}},
            "dragon_tiger": {"status": "ok", "data": {"is_on_list": True, "recent_count": 1, "latest_date": "2026-03-18"}},
            "boards": {
                "status": "ok",
                "data": {
                    "top": [{"name": "白酒", "change_pct": 2.3}],
                    "bottom": [{"name": "煤炭", "change_pct": -1.7}],
                },
            },
            "announcements": {
                "status": "ok",
                "data": {
                    "events": [
                        {"date": "2026-03-18", "category": "earnings", "title": "贵州茅台业绩快报"},
                    ]
                },
            },
            "northbound": {"status": "ok", "data": {"net_buy_direction": "净买入", "change_shares_5d": 12345}},
            "margin": {"status": "ok", "data": {"direction": "明显加杠杆", "balance_change_pct": 5.6}},
            "shareholder_count": {"status": "ok", "data": {"trend": "筹码趋集", "delta_pct": -3.2}},
        }
        self._belong_boards = [{"name": "白酒"}, {"name": "消费"}]

    def get_fundamental_context(self, _stock_code: str):
        return self._context

    def build_failed_fundamental_context(self, _stock_code: str, _reason: str):
        return {}

    def get_belong_boards(self, _stock_code: str):
        return self._belong_boards

    def get_stock_name(self, _stock_code: str):
        return "贵州茅台"


class TestGetStockInfoContract(unittest.TestCase):
    def test_get_stock_info_preserves_board_semantics(self) -> None:
        manager = _DummyManager()
        with patch("src.agent.tools.data_tools._get_fetcher_manager", return_value=manager):
            result = _handle_get_stock_info("600519")

        self.assertEqual(result["name"], "贵州茅台")
        self.assertEqual(result["code"], "600519")
        self.assertEqual(result["pe_ratio"], 12.3)
        self.assertEqual(result["pb_ratio"], 2.1)

        # Contract: boards is compatibility alias of belong_boards.
        self.assertEqual(result["belong_boards"], manager._belong_boards)
        self.assertEqual(result["boards"], result["belong_boards"])

        # Contract: sector_rankings comes from fundamental_context.boards.data.
        self.assertEqual(result["sector_rankings"], manager._context["boards"]["data"])
        self.assertEqual(
            result["fundamental_context"]["boards"]["data"],
            result["sector_rankings"],
        )

    def test_get_stock_event_context_exposes_structured_event_blocks(self) -> None:
        manager = _DummyManager()
        with patch("src.agent.tools.data_tools._get_fetcher_manager", return_value=manager):
            result = _handle_get_stock_event_context("600519")

        self.assertEqual(result["name"], "贵州茅台")
        self.assertEqual(result["recent_announcements"][0]["category"], "earnings")
        self.assertEqual(result["shareholder_count"]["trend"], "筹码趋集")
        self.assertIn("announcements", result["event_context"])
        self.assertIn("shareholder_count", result["event_context"])

    def test_get_a_share_flow_context_exposes_flow_blocks(self) -> None:
        manager = _DummyManager()
        with patch("src.agent.tools.data_tools._get_fetcher_manager", return_value=manager):
            result = _handle_get_a_share_flow_context("600519")

        self.assertEqual(result["name"], "贵州茅台")
        self.assertEqual(result["northbound"]["net_buy_direction"], "净买入")
        self.assertEqual(result["margin"]["direction"], "明显加杠杆")
        self.assertEqual(result["dragon_tiger"]["is_on_list"], True)
        self.assertEqual(result["belong_boards"], manager._belong_boards)


if __name__ == "__main__":
    unittest.main()
