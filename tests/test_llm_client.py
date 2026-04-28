import json
import unittest
from unittest.mock import Mock, patch

import requests

from reviewbot.llm_client import ChatCompletionClient


def _interrupted_iter_lines(prefix_lines: list[str], exc: Exception):
    def gen(*_args, **_kwargs):
        for line in prefix_lines:
            yield line
        raise exc

    return gen


class ChatCompletionClientTests(unittest.TestCase):
    def test_complete_uses_explicit_model_without_discovery(self) -> None:
        with patch("reviewbot.llm_client.requests.get") as mock_get, patch(
            "reviewbot.llm_client.requests.post"
        ) as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com/v1", "token", "fixed-model")
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        mock_get.assert_not_called()
        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertEqual(payload["model"], "fixed-model")

    def test_complete_discovers_first_model_once(self) -> None:
        with patch("reviewbot.llm_client.requests.get") as mock_get, patch(
            "reviewbot.llm_client.requests.post"
        ) as mock_post:
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": [{"id": "auto-model"}, {"id": "other"}]}),
            )
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com/v1", "token")
            first = client.complete([{"role": "user", "content": "one"}])
            second = client.complete([{"role": "user", "content": "two"}])

        self.assertEqual(first.content, "ok")
        self.assertEqual(second.content, "ok")
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
        with patch("reviewbot.llm_client.requests.get") as mock_get, patch(
            "reviewbot.llm_client.requests.post"
        ) as mock_post:
            mock_get.return_value = Mock(
                ok=True,
                status_code=200,
                json=Mock(return_value={"data": [{"id": "auto-model"}]}),
            )
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com", "token")
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
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

    def test_complete_sends_bill_to_header_when_configured(self) -> None:
        with patch("reviewbot.llm_client.requests.get") as mock_get, patch(
            "reviewbot.llm_client.requests.post"
        ) as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", bill_to="my-org"
            )
            client.complete([{"role": "user", "content": "hi"}])

        mock_get.assert_not_called()
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["X-HF-Bill-To"], "my-org"
        )

    def test_complete_omits_bill_to_header_when_not_configured(self) -> None:
        with patch("reviewbot.llm_client.requests.get"), patch(
            "reviewbot.llm_client.requests.post"
        ) as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
                raise_for_status=Mock(),
            )

            client = ChatCompletionClient("https://example.com/v1", "token", "fixed-model")
            client.complete([{"role": "user", "content": "hi"}])

        self.assertNotIn("X-HF-Bill-To", mock_post.call_args.kwargs["headers"])

    def test_complete_retries_on_5xx_then_succeeds(self) -> None:
        flaky = Mock(
            status_code=504,
            json=Mock(return_value={}),
            raise_for_status=Mock(side_effect=AssertionError("should not be called")),
        )
        ok = Mock(
            status_code=200,
            json=Mock(return_value={"choices": [{"message": {"content": "ok"}}]}),
            raise_for_status=Mock(),
        )
        with patch("reviewbot.llm_client.time.sleep"), patch(
            "reviewbot.llm_client.requests.post", side_effect=[flaky, ok]
        ) as mock_post:
            client = ChatCompletionClient("https://example.com/v1", "token", "fixed-model")
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "ok")
        self.assertEqual(mock_post.call_count, 2)

    def test_complete_consumes_sse_stream_when_streaming_enabled(self) -> None:
        sse_lines = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}',
            "data: [DONE]",
        ]
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                raise_for_status=Mock(),
                iter_lines=Mock(return_value=iter(sse_lines)),
            )

            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "hello")
        self.assertEqual(result.prompt_tokens, 3)
        self.assertEqual(result.completion_tokens, 2)
        payload = json.loads(mock_post.call_args.kwargs["data"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertTrue(mock_post.call_args.kwargs["stream"])

    def test_complete_retries_on_stream_interruption_then_succeeds(self) -> None:
        sse_full = [
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            "data: [DONE]",
        ]
        flaky = Mock(
            status_code=200,
            raise_for_status=Mock(),
            iter_lines=_interrupted_iter_lines(
                [sse_full[0]],
                requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            ),
        )
        ok = Mock(
            status_code=200,
            raise_for_status=Mock(),
            iter_lines=Mock(return_value=iter(sse_full)),
        )
        with patch("reviewbot.llm_client.time.sleep"), patch(
            "reviewbot.llm_client.requests.post", side_effect=[flaky, ok]
        ) as mock_post:
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            result = client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(result.content, "hello")
        self.assertEqual(mock_post.call_count, 2)

    def test_complete_raises_after_exhausting_stream_retries(self) -> None:
        flaky = lambda: Mock(  # noqa: E731
            status_code=200,
            raise_for_status=Mock(),
            iter_lines=_interrupted_iter_lines(
                [],
                requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            ),
        )
        with patch("reviewbot.llm_client.time.sleep"), patch(
            "reviewbot.llm_client.requests.post",
            side_effect=[flaky(), flaky(), flaky()],
        ) as mock_post:
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                client.complete([{"role": "user", "content": "hi"}])

        self.assertEqual(mock_post.call_count, 3)

    def test_stream_assembles_tool_calls_across_chunks(self) -> None:
        # Two tool_calls, each split across multiple chunks. The arguments
        # for index 0 (`read_file`) and index 1 (`grep`) arrive piecewise.
        sse_lines = [
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"id":"call_a","function":{"name":"read_file","arguments":"{\\"pat"}}'
            "]}}]}",
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":0,"function":{"arguments":"h\\":\\"src/main.py\\"}"}}'
            "]}}]}",
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":1,"id":"call_b","function":{"name":"grep","arguments":"{\\"pat"}}'
            "]}}]}",
            'data: {"choices":[{"delta":{"tool_calls":['
            '{"index":1,"function":{"arguments":"tern\\":\\"hello\\"}"}}'
            "]}}]}",
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            "data: [DONE]",
        ]
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                raise_for_status=Mock(),
                iter_lines=Mock(return_value=iter(sse_lines)),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=True
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "read_file"}}],
            )

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].id, "call_a")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(
            json.loads(result.tool_calls[0].arguments), {"path": "src/main.py"}
        )
        self.assertEqual(result.tool_calls[1].id, "call_b")
        self.assertEqual(result.tool_calls[1].name, "grep")
        self.assertEqual(
            json.loads(result.tool_calls[1].arguments), {"pattern": "hello"}
        )

    def test_non_stream_parses_tool_calls(self) -> None:
        with patch("reviewbot.llm_client.requests.post") as mock_post:
            mock_post.return_value = Mock(
                status_code=200,
                json=Mock(
                    return_value={
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call_x",
                                            "type": "function",
                                            "function": {
                                                "name": "list_dir",
                                                "arguments": '{"path":"src"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ),
                raise_for_status=Mock(),
            )
            client = ChatCompletionClient(
                "https://example.com/v1", "token", "fixed-model", stream=False
            )
            result = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "list_dir"}}],
            )

        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "list_dir")
        self.assertEqual(
            json.loads(result.tool_calls[0].arguments), {"path": "src"}
        )

    def test_complete_raises_when_discovery_returns_no_models(self) -> None:
        with patch("reviewbot.llm_client.requests.get") as mock_get, patch(
            "reviewbot.llm_client.requests.post"
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
