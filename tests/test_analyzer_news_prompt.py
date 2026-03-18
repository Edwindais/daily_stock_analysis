# -*- coding: utf-8 -*-
"""Tests for analyzer news prompt hard constraints (Issue #697)."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    sys.modules["litellm"] = MagicMock()

from src.analyzer import GeminiAnalyzer


class AnalyzerNewsPromptTestCase(unittest.TestCase):
    def test_prompt_contains_time_constraints(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()

        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-16",
            "today": {},
            "fundamental_context": {
                "earnings": {
                    "data": {
                        "financial_report": {"report_date": "2025-12-31", "revenue": 1000},
                        "dividend": {"ttm_cash_dividend_per_share": 1.2, "ttm_dividend_yield_pct": 2.4},
                    }
                }
            },
        }
        fake_cfg = SimpleNamespace(
            news_max_age_days=30,
            news_strategy_profile="medium",  # 7 days
        )
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(context, "贵州茅台", news_context="news")

        self.assertIn("近7日的新闻搜索结果", prompt)
        self.assertIn("每一条都必须带具体日期（YYYY-MM-DD）", prompt)
        self.assertIn("超出近7日窗口的新闻一律忽略", prompt)
        self.assertIn("时间未知、无法确定发布日期的新闻一律忽略", prompt)
        self.assertIn("财报与分红（价值投资口径）", prompt)
        self.assertIn("禁止编造", prompt)

    def test_prompt_prefers_context_news_window_days(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()

        context = {
            "code": "600519",
            "stock_name": "贵州茅台",
            "date": "2026-03-16",
            "today": {},
            "news_window_days": 1,
        }
        fake_cfg = SimpleNamespace(
            news_max_age_days=30,
            news_strategy_profile="long",  # 30 days if fallback is used
        )
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(context, "贵州茅台", news_context="news")

        self.assertIn("近1日的新闻搜索结果", prompt)
        self.assertIn("超出近1日窗口的新闻一律忽略", prompt)

    def test_prompt_includes_a_share_capital_flow_and_momentum_context(self) -> None:
        with patch.object(GeminiAnalyzer, "_init_litellm", return_value=None):
            analyzer = GeminiAnalyzer()

        context = {
            "code": "002214",
            "stock_name": "*ST大立",
            "date": "2026-03-18",
            "today": {"ma5": 10, "ma10": 9, "ma20": 8},
            "belong_boards": [{"name": "军工"}, {"name": "红外热像"}],
            "trend_analysis": {
                "trend_status": "弱势空头",
                "ma_alignment": "MA5<MA10<MA20",
                "trend_strength": 35,
                "ma60": 11.5,
                "bias_ma5": -1.2,
                "bias_ma10": -2.1,
                "bias_ma20": -3.4,
                "volume_status": "缩量回调",
                "volume_ratio_5d": 0.8,
                "volume_trend": "量能回落",
                "buy_signal": "观望",
                "signal_score": 42,
                "signal_reasons": ["量能未放大"],
                "risk_factors": ["趋势偏弱"],
                "support_levels": [9.8],
                "resistance_levels": [10.6],
                "macd_status": "死叉",
                "macd_signal": "MACD走弱",
                "macd_dif": -0.12,
                "macd_dea": -0.08,
                "rsi_status": "弱势",
                "rsi_signal": "RSI偏弱",
                "rsi_6": 32.1,
                "rsi_12": 38.5,
                "rsi_24": 42.0,
            },
            "fundamental_context": {
                "valuation": {"data": {"pe_ratio": 32.5, "pb_ratio": 2.1}},
                "growth": {"data": {"revenue_yoy": 15.2, "net_profit_yoy": 28.1, "gross_margin": 36.2, "roe": 9.8}},
                "capital_flow": {
                    "data": {
                        "stock_flow": {"main_net_inflow": 12345678, "inflow_5d": 23456789, "inflow_10d": 34567890},
                        "sector_rankings": {
                            "top": [{"name": "军工", "net_inflow": 45678901}],
                            "bottom": [{"name": "地产", "net_inflow": -12345678}],
                        },
                    }
                },
                "dragon_tiger": {"data": {"is_on_list": True, "recent_count": 2, "latest_date": "2026-03-17"}},
                "boards": {
                    "data": {
                        "top": [{"name": "军工", "change_pct": "+3.2%"}],
                        "bottom": [{"name": "地产", "change_pct": "-2.1%"}],
                    }
                },
                "earnings": {"data": {}},
            },
        }

        fake_cfg = SimpleNamespace(
            news_max_age_days=7,
            news_strategy_profile="medium",
        )
        with patch("src.analyzer.get_config", return_value=fake_cfg):
            prompt = analyzer._format_prompt(context, "*ST大立", news_context="news")

        self.assertIn("资金流与板块资金", prompt)
        self.assertIn("龙虎榜活跃度", prompt)
        self.assertIn("个股所属板块 / 概念", prompt)
        self.assertIn("MACD状态", prompt)
        self.assertIn("RSI状态", prompt)
        self.assertIn("题材动量股", prompt)
        self.assertIn("同名或相似简称但代码不一致的资讯一律忽略", prompt)


if __name__ == "__main__":
    unittest.main()
