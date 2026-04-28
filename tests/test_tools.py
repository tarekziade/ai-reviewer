import os
import shutil
import subprocess
import tempfile
import unittest

from reviewbot.tools import (
    DENY_DIR_NAMES,
    MAX_READ_LINES,
    MAX_TOOL_OUTPUT_CHARS,
    ToolEnv,
    run_tool,
)


class _TempRepo(unittest.TestCase):
    """Tests share a temp git repo under self.repo_root with a small
    fixture tree we know the contents of."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)
        self.repo_root = self.tmpdir

        # Fixture files
        os.makedirs(os.path.join(self.repo_root, "src", "pkg"))
        with open(os.path.join(self.repo_root, "src", "pkg", "main.py"), "w") as f:
            f.write("def hello():\n    return 'world'\n\n\ndef goodbye():\n    return 'bye'\n")
        with open(os.path.join(self.repo_root, "src", "pkg", "util.py"), "w") as f:
            f.write("CONSTANT = 1\n")
        with open(os.path.join(self.repo_root, "README.md"), "w") as f:
            f.write("# fixture repo\n")
        # Pretend there's a .git dir so the denylist has something to bite on
        os.makedirs(os.path.join(self.repo_root, ".git", "refs"))
        # Real git init so `git grep` works in the grep tests
        subprocess.run(
            ["git", "init", "-q"], cwd=self.repo_root, check=True
        )
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=a", "add", "."],
            cwd=self.repo_root,
            check=True,
        )
        subprocess.run(
            ["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "-m", "init"],
            cwd=self.repo_root,
            check=True,
        )

        self.env = ToolEnv(repo_root=self.repo_root)


class ResolvePathTests(_TempRepo):
    def test_read_file_returns_numbered_lines(self) -> None:
        out = run_tool(
            self.env, "read_file", {"path": "src/pkg/main.py", "start_line": 1, "end_line": 3}
        )
        self.assertIn("src/pkg/main.py (lines 1-3)", out)
        self.assertIn("     1\tdef hello():", out)
        self.assertIn("     2\t    return 'world'", out)

    def test_read_file_caps_at_max_read_lines(self) -> None:
        big = os.path.join(self.repo_root, "big.txt")
        with open(big, "w") as f:
            for i in range(1, MAX_READ_LINES + 50):
                f.write(f"line{i}\n")
        out = run_tool(
            self.env,
            "read_file",
            {"path": "big.txt", "start_line": 1, "end_line": MAX_READ_LINES + 100},
        )
        self.assertIn(f"lines 1-{MAX_READ_LINES}", out)

    def test_read_file_rejects_path_escape(self) -> None:
        out = run_tool(self.env, "read_file", {"path": "../etc/passwd"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("escapes the repository root", out)

    def test_read_file_rejects_absolute_path(self) -> None:
        out = run_tool(self.env, "read_file", {"path": "/etc/passwd"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("absolute or home-relative", out)

    def test_read_file_rejects_denylisted_dir(self) -> None:
        out = run_tool(self.env, "read_file", {"path": ".git/HEAD"})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("denylisted", out)

    def test_read_file_errors_on_directory(self) -> None:
        out = run_tool(self.env, "read_file", {"path": "src"})
        self.assertIn("error:", out)
        self.assertIn("not a file", out)

    def test_list_dir_lists_entries_and_hides_denylist(self) -> None:
        out = run_tool(self.env, "list_dir", {"path": "."})
        self.assertIn("README.md", out)
        self.assertIn("src/", out)
        for name in DENY_DIR_NAMES:
            self.assertNotIn(f"\n{name}\n", out)

    def test_list_dir_descends_into_subdirs(self) -> None:
        out = run_tool(self.env, "list_dir", {"path": "src/pkg"})
        self.assertIn("main.py", out)
        self.assertIn("util.py", out)

    def test_grep_finds_match(self) -> None:
        out = run_tool(
            self.env, "grep", {"pattern": "def hello", "path": "src"}
        )
        self.assertIn("src/pkg/main.py:1:def hello", out)

    def test_grep_reports_no_matches(self) -> None:
        out = run_tool(
            self.env, "grep", {"pattern": "definitely_not_there", "path": "src"}
        )
        self.assertIn("no matches", out)

    def test_grep_rejects_non_ascii_pattern(self) -> None:
        out = run_tool(self.env, "grep", {"pattern": "héllo"})
        self.assertTrue(out.startswith("error:"), out)

    def test_unknown_tool_returns_error_message(self) -> None:
        out = run_tool(self.env, "delete_repo", {})
        self.assertTrue(out.startswith("error:"), out)
        self.assertIn("unknown tool", out)

    def test_truncates_oversized_output(self) -> None:
        big = os.path.join(self.repo_root, "huge.txt")
        with open(big, "w") as f:
            f.write("x" * (MAX_TOOL_OUTPUT_CHARS * 2))
        out = run_tool(self.env, "read_file", {"path": "huge.txt"})
        self.assertIn("[... truncated", out)


if __name__ == "__main__":
    unittest.main()
