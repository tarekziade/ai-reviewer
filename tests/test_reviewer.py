import unittest

from reviewbot.reviewer import (
    _content_preview,
    _extract_json,
    _summarize_rejected_comments,
)


class ExtractJsonTests(unittest.TestCase):
    def test_raw_json_object(self) -> None:
        result = _extract_json('{"summary": "ok", "comments": []}')
        self.assertEqual(result, {"summary": "ok", "comments": []})

    def test_strips_surrounding_whitespace(self) -> None:
        result = _extract_json('   \n  {"summary": "ok"}  \n\n')
        self.assertEqual(result, {"summary": "ok"})

    def test_fenced_block_with_json_tag(self) -> None:
        content = 'Here you go:\n```json\n{"summary": "ok"}\n```\nThanks!'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_fenced_block_without_language_tag(self) -> None:
        content = '```\n{"summary": "ok"}\n```'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_fenced_block_uppercase_tag(self) -> None:
        content = '```JSON\n{"a": 1}\n```'
        self.assertEqual(_extract_json(content), {"a": 1})

    def test_skips_empty_fenced_block_then_recovers(self) -> None:
        content = '```\n\n```\nOr maybe:\n```json\n{"summary": "ok"}\n```'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_json_embedded_in_prose_no_fences(self) -> None:
        content = 'Sure: {"summary": "ok", "event": "COMMENT"} — let me know!'
        self.assertEqual(
            _extract_json(content),
            {"summary": "ok", "event": "COMMENT"},
        )

    def test_json_with_braces_in_prose_before_and_after(self) -> None:
        # Stray braces in surrounding prose used to break the naive
        # find('{') / rfind('}') slicing; raw_decode at every '{' recovers.
        content = 'Note: use { for sets.\n{"summary": "ok"}\nUse } to close.'
        self.assertEqual(_extract_json(content), {"summary": "ok"})

    def test_first_object_wins_when_multiple_candidates(self) -> None:
        content = '{"summary": "first"}\n\nAlso: {"summary": "second"}'
        # Direct parse fails because of trailing data; first raw_decode wins.
        self.assertEqual(_extract_json(content), {"summary": "first"})

    def test_top_level_array_unwraps_to_inner_object(self) -> None:
        # If the model wraps the review in an array (against the contract),
        # the raw_decode pass still recovers the inner object — pragmatic
        # over strict, since downstream code only needs a dict.
        self.assertEqual(_extract_json('[{"summary": "ok"}]'), {"summary": "ok"})

    def test_top_level_array_with_no_inner_object_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("[1, 2, 3]")
        self.assertIn("did not contain a JSON object", str(ctx.exception))

    def test_empty_string_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_none_content_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json(None)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_whitespace_only_raises_with_clear_message(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("   \n\t  \n")
        self.assertIn("whitespace", str(ctx.exception).lower())

    def test_failure_message_includes_length_and_preview(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _extract_json("I cannot help with this request.")
        msg = str(ctx.exception)
        self.assertIn("length=", msg)
        self.assertIn("preview=", msg)
        self.assertIn("cannot help", msg)

    def test_failure_preview_truncates_long_content(self) -> None:
        long = "x" * 5000
        with self.assertRaises(ValueError) as ctx:
            _extract_json(long)
        self.assertIn("length=5000", str(ctx.exception))
        # Full 5000 chars must NOT be in the preview.
        self.assertLess(len(str(ctx.exception)), 1500)

    def test_nested_object_with_braces_inside_strings(self) -> None:
        content = '{"summary": "use { and } carefully", "comments": []}'
        self.assertEqual(
            _extract_json(content),
            {"summary": "use { and } carefully", "comments": []},
        )


class ContentPreviewTests(unittest.TestCase):
    def test_short_content_returned_verbatim(self) -> None:
        self.assertEqual(_content_preview("hello"), "hello")

    def test_long_content_truncated_with_marker(self) -> None:
        out = _content_preview("x" * 1000, limit=100)
        self.assertTrue(out.startswith("x" * 100))
        self.assertIn("+900 chars truncated", out)


class SummarizeRejectedCommentsTests(unittest.TestCase):
    def test_empty_list_renders_empty_string(self) -> None:
        self.assertEqual(_summarize_rejected_comments([]), "")

    def test_renders_path_line_refs(self) -> None:
        out = _summarize_rejected_comments(
            [{"path": "foo.py", "line": 10}, {"path": "bar.py", "line": 20}]
        )
        self.assertEqual(out, "foo.py:10, bar.py:20")

    def test_truncates_after_max_items(self) -> None:
        rejected = [{"path": f"f{i}.py", "line": i} for i in range(10)]
        out = _summarize_rejected_comments(rejected, max_items=3)
        self.assertIn("f0.py:0", out)
        self.assertIn("f2.py:2", out)
        self.assertIn("+7 more", out)
        self.assertNotIn("f9.py:9", out)

    def test_handles_missing_fields_gracefully(self) -> None:
        out = _summarize_rejected_comments([{}, {"path": "foo.py"}])
        self.assertEqual(out, "?:?, foo.py:?")


if __name__ == "__main__":
    unittest.main()
