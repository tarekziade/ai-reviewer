import os
import stat
import tempfile
import textwrap
import unittest

from reviewbot.context_script import run_context_script


def _write_script(dir_: str, body: str, *, executable: bool = True, name: str = "ctx") -> str:
    path = os.path.join(dir_, name)
    with open(path, "w") as f:
        f.write(body)
    if executable:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class ContextScriptTests(unittest.TestCase):
    def test_returns_none_when_path_blank(self) -> None:
        self.assertIsNone(
            run_context_script("", title="t", body="b", files=[], timeout_seconds=5)
        )

    def test_returns_none_when_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(
                run_context_script(
                    "does-not-exist",
                    title="t",
                    body="b",
                    files=[],
                    timeout_seconds=5,
                    cwd=d,
                )
            )

    def test_returns_none_when_not_executable(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\necho hi\n", executable=False)
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_passes_json_on_stdin_and_returns_stdout(self) -> None:
        script = textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json, sys
            data = json.load(sys.stdin)
            print("title=" + data["title"])
            print("nfiles=" + str(len(data["files"])))
            print("first=" + data["files"][0]["path"])
            print("status=" + data["files"][0]["status"])
            """
        )
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, script)
            out = run_context_script(
                "ctx",
                title="My PR",
                body="desc",
                files=[
                    {"filename": "a.py", "status": "modified", "additions": 3, "deletions": 1},
                    {"filename": "b.py", "status": "added"},
                ],
                timeout_seconds=10,
                cwd=d,
            )
        assert out is not None
        self.assertIn("title=My PR", out)
        self.assertIn("nfiles=2", out)
        self.assertIn("first=a.py", out)
        self.assertIn("status=modified", out)

    def test_returns_none_on_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\necho oops\nexit 7\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_returns_none_on_empty_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\nexit 0\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=5, cwd=d
                )
            )

    def test_timeout_is_swallowed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\nsleep 10\n")
            self.assertIsNone(
                run_context_script(
                    "ctx", title="t", body="b", files=[], timeout_seconds=1, cwd=d
                )
            )

    def test_truncates_oversized_output(self) -> None:
        # Print 20k bytes; runner caps at MAX_OUTPUT_CHARS (8000).
        with tempfile.TemporaryDirectory() as d:
            _write_script(d, "#!/bin/sh\npython3 -c 'print(\"x\"*20000)'\n")
            out = run_context_script(
                "ctx", title="t", body="b", files=[], timeout_seconds=10, cwd=d
            )
        assert out is not None
        self.assertIn("truncated", out)
        self.assertLess(len(out), 9000)


if __name__ == "__main__":
    unittest.main()
