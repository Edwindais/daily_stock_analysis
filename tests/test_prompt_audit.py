import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import prompt_audit


class PromptAuditTests(unittest.TestCase):
    def test_write_prompt_snapshot_creates_timestamped_txt(self):
        messages = [
            {"role": "system", "content": "你是一位A股分析师"},
            {"role": "user", "content": "分析股票\n股票代码: 300369"},
        ]
        logger = logging.getLogger("prompt_audit_test")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(prompt_audit, "_PROMPT_DIR", Path(tmpdir)):
                file_path = prompt_audit.write_prompt_snapshot(
                    logger=logger,
                    source="unit_test",
                    model="gemini-3-flash-preview",
                    messages=messages,
                    metadata={"prompt_kind": "stock_analysis", "stock_code": "300369"},
                )
                self.assertIsNotNone(file_path)
                self.assertTrue(file_path.exists())
                content = file_path.read_text(encoding="utf-8")
                self.assertIn("股票代码: 300369", content)

        self.assertTrue(str(file_path).endswith(".txt"))
        self.assertIn("stock_analysis", file_path.name)
        self.assertIn("300369", file_path.name)
        self.assertRegex(file_path.name, r"^\d{8}_\d{6}_\d{6}_")


if __name__ == "__main__":
    unittest.main()
