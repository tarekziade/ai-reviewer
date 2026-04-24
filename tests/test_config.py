import os
import unittest
from unittest.mock import patch

from config import Config


class ConfigTests(unittest.TestCase):
    def test_defaults_to_serge_trigger(self) -> None:
        with patch.dict(os.environ, {"LLM_API_KEY": "token"}, clear=True):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.mention_trigger, "@serge")

    def test_respects_explicit_trigger_override(self) -> None:
        with patch.dict(
            os.environ,
            {"LLM_API_KEY": "token", "MENTION_TRIGGER": "@custom"},
            clear=True,
        ):
            cfg = Config.from_env(require_app=False)

        self.assertEqual(cfg.mention_trigger, "@custom")


if __name__ == "__main__":
    unittest.main()
