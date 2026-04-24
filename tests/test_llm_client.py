import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *args, **kwargs: None
requests_stub.post = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

from llm_client import ChatCompletionClient


class ChatCompletionClientTests(unittest.TestCase):
    def test_complete_uses_explicit_model_without_discovery(self) -> None:
        with patch("llm_client.requests.get") as mock_get, patch(
            "llm_client.requests.post"
        ) as mock_post:
            mock_post.return_value = Mock(
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com/v1", "token", "fixed-model")
            content = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(content, "ok")
        mock_get.assert_not_called()
        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertEqual(payload["model"], "fixed-model")

    def test_complete_discovers_first_model_once(self) -> None:
        with patch("llm_client.requests.get") as mock_get, patch(
            "llm_client.requests.post"
        ) as mock_post:
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": [{"id": "auto-model"}, {"id": "other"}]}),
            )
            mock_post.return_value = Mock(
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com/v1", "token")
            first = client.complete([{"role": "user", "content": "one"}])
            second = client.complete([{"role": "user", "content": "two"}])

        self.assertEqual(first, "ok")
        self.assertEqual(second, "ok")
        mock_get.assert_called_once_with(
            "https://example.com/v1/models",
            headers={
                "Authorization": "Bearer token",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        first_payload = json.loads(mock_post.call_args_list[0].kwargs["data"])
        second_payload = json.loads(mock_post.call_args_list[1].kwargs["data"])
        self.assertEqual(first_payload["model"], "auto-model")
        self.assertEqual(second_payload["model"], "auto-model")

    def test_complete_adds_v1_when_base_is_unversioned(self) -> None:
        with patch("llm_client.requests.get") as mock_get, patch(
            "llm_client.requests.post"
        ) as mock_post:
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": [{"id": "auto-model"}]}),
            )
            mock_post.return_value = Mock(
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com", "token")
            content = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(content, "ok")
        mock_get.assert_called_once_with(
            "https://example.com/v1/models",
            headers={
                "Authorization": "Bearer token",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        self.assertEqual(
            mock_post.call_args.kwargs["url"]
            if "url" in mock_post.call_args.kwargs
            else mock_post.call_args.args[0],
            "https://example.com/v1/chat/completions",
        )

    def test_complete_raises_when_discovery_returns_no_models(self) -> None:
        with patch("llm_client.requests.get") as mock_get, patch(
            "llm_client.requests.post"
        ) as mock_post:
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": []}),
            )

            client = ChatCompletionClient("https://example.com/v1", "token")
            with self.assertRaisesRegex(RuntimeError, "No models found"):
                client.complete([{"role": "user", "content": "hi"}])

        mock_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
