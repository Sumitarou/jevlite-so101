"""scripts/hooks/pre-commit in a throw-away git repo with made-up secret values."""

import shutil
import subprocess
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parent.parent / "scripts" / "hooks" / "pre-commit"
VALUES = {"hostname": "fakehost42", "personal-email": "someone@example.org", "ipv4": "10.9.8.7"}

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("perl") is None, reason="needs git and perl"
)


def git(repo: Path, *args: str, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "r"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "user.name", "t")
    git(r, "config", "user.email", "t@example.invalid")
    shutil.copy(HOOK, r / ".git" / "hooks" / "pre-commit")
    (r / ".git" / "hooks" / "pre-commit").chmod(0o755)
    lines = ["# test values"] + [f"{k}\t{v}" for k, v in VALUES.items()]
    (r / ".git" / "info" / "secret-patterns").write_text("\n".join(lines) + "\n")
    (r / "base.txt").write_text("base\n")
    git(r, "add", "base.txt")
    git(r, "commit", "-q", "-m", "base", "--no-verify")
    return r


def commit(repo: Path, files: dict[str, bytes | str]) -> subprocess.CompletedProcess:
    for name, content in files.items():
        p = repo / name
        p.write_bytes(content if isinstance(content, bytes) else content.encode())
        git(repo, "add", name)
    return git(repo, "commit", "-q", "-m", "test", check=False)


def assert_no_value_in(text: str):
    for v in VALUES.values():
        assert v.lower() not in text.lower()


def test_harmless_commit_passes(repo):
    assert commit(repo, {"doc.md": "see 192.168.0.1 and /home/alice\n"}).returncode == 0


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("hostname", "built on fakehost42\n"),
        ("personal-email", "mail SOMEONE@EXAMPLE.ORG\n"),  # case-insensitive
        ("ipv4", "ip 10.9.8.7\n"),
    ],
)
def test_value_in_added_line_is_refused_without_printing_it(repo, label, content):
    res = commit(repo, {"a.txt": content})
    assert res.returncode != 0
    assert f"{label}: 1 line(s) in a.txt" in res.stderr
    assert_no_value_in(res.stderr + res.stdout)
    assert git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"  # nothing committed


def test_value_in_binary_file_is_refused(repo):
    res = commit(repo, {"img.bin": b"\x89PNG\x00\x01" + b"fakehost42" + b"\x00\xff"})
    assert res.returncode != 0
    assert "hostname" in res.stderr
    assert_no_value_in(res.stderr)


def test_value_in_file_name_is_refused_and_masked(repo):
    res = commit(repo, {"notes-FakeHost42.txt": "ok\n"})
    assert res.returncode != 0
    assert "notes-<hostname>.txt" in res.stderr
    assert_no_value_in(res.stderr)


def test_removing_a_value_is_allowed(repo):
    (repo / "old.txt").write_text("fakehost42\n")
    git(repo, "add", "old.txt")
    git(repo, "commit", "-q", "-m", "old", "--no-verify")
    assert commit(repo, {"old.txt": "cleaned\n"}).returncode == 0


@pytest.mark.parametrize("content", [None, "", "# only a comment\n"])
def test_missing_or_empty_patterns_fail_closed(repo, content):
    p = repo / ".git" / "info" / "secret-patterns"
    if content is None:
        p.unlink()
    else:
        p.write_text(content)
    res = commit(repo, {"x.txt": "harmless\n"})
    assert res.returncode != 0
    assert "refusing to commit" in res.stderr


def test_works_inside_a_worktree(repo, tmp_path):
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", str(wt), "-b", "wt")
    assert commit(wt, {"w.txt": "fine\n"}).returncode == 0
    assert commit(wt, {"v.txt": "fakehost42\n"}).returncode != 0
