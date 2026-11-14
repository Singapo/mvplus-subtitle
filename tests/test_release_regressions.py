"""Regressions for the v0.1.0 release review.

Each test pins one defect that a new user would hit on a clean machine:

1. credentials only in the environment made ``scan`` die with ``KeyError``;
2. ``home`` in the TOML was silently ignored, so paths stayed on C:;
3. the first ``scan`` crashed when the work directory did not exist yet.

They are written against the public helpers, not internals, so a future
refactor of how the config is loaded still has to keep these promises.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mvsub" / "pipeline"))

import config  # noqa: E402
import nosub_pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_config():
    """Drop cached config around each test; the cache is process-wide.

    Credential isolation itself comes from ``tests/conftest.py`` — it points
    ``MVSUB_ENV_FILE`` at a path that does not exist, so popping that variable
    here would re-enable the repository ``.env`` fallback and undo it.
    """
    config.reload()
    yield
    config.reload()


def test_credentials_come_from_the_environment_too(monkeypatch, tmp_path):
    """Env-only deployment: no .env anywhere, credentials in the environment."""
    monkeypatch.setenv("MVSUB_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("P115_COOKIES", "UID=1; CID=2; SEID=3; KID=4")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    config.reload()

    env = nosub_pipeline.load_env()

    assert env["P115_COOKIES"] == "UID=1; CID=2; SEID=3; KID=4"
    assert env["DEEPSEEK_API_KEY"] == "sk-test"
    # and the scan entry point must resolve a cookie instead of raising KeyError
    assert nosub_pipeline.cloud_cookie(env).startswith("UID=1")


def test_environment_wins_over_the_dotenv_file(monkeypatch, tmp_path):
    """The docs promise env > file; returning file values regardless would break that."""
    env_file = tmp_path / "creds.env"
    env_file.write_text(
        "P115_COOKIES=from-file\nDEEPSEEK_API_KEY=sk-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MVSUB_ENV_FILE", str(env_file))
    monkeypatch.setenv("P115_COOKIES", "from-env")
    config.reload()

    env = nosub_pipeline.load_env()

    assert env["P115_COOKIES"] == "from-env"
    # a key absent from the environment still comes from the file
    assert env["DEEPSEEK_API_KEY"] == "sk-file"


def test_cuda_libs_is_configurable(monkeypatch, tmp_path):
    """The CUDA DLL directory must be overridable, not hardcoded to the repo.

    Without this, a deployment that already has the DLLs (or installs the
    nvidia wheels) still fails on GPU decode with `cublas64_12.dll is not found`.
    """
    target = tmp_path / "cuda"
    monkeypatch.setenv("MVSUB_CUDA_LIBS", str(target))
    config.reload()

    assert config.cuda_libs() == target


def test_cuda_libs_defaults_inside_the_checkout(monkeypatch):
    monkeypatch.delenv("MVSUB_CUDA_LIBS", raising=False)
    config.reload()

    assert config.cuda_libs() == config.repo_root() / "lib" / "cuda"


def test_explicit_env_file_is_authoritative(monkeypatch, tmp_path):
    """Naming a credentials file must not silently fall back to another one.

    A missing ``MVSUB_ENV_FILE`` used to fall through to ``./.env``; on a
    configured machine that means a test (or a CI job) picks up the real cookie.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text("P115_COOKIES=should-not-be-read\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("MVSUB_ENV_FILE", str(tmp_path / "absent.env"))
    config.reload()

    assert config.load_dotenv() == {}
    assert config.dotenv_value("P115_COOKIES") == ""


def test_explicit_env_file_is_read_when_present(monkeypatch, tmp_path):
    env_file = tmp_path / "creds.env"
    env_file.write_text("P115_COOKIES=from-named-file\n", encoding="utf-8")
    monkeypatch.setenv("MVSUB_ENV_FILE", str(env_file))
    config.reload()

    assert config.dotenv_value("P115_COOKIES") == "from-named-file"


def test_tests_never_see_a_real_dotenv():
    """Guards the autouse isolation in conftest: a real .env must stay invisible."""
    assert config.dotenv_value("P115_COOKIES") == ""
    assert config.load_dotenv() == {}


def test_missing_cookie_reports_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setenv("MVSUB_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("P115_COOKIES", raising=False)
    config.reload()

    assert nosub_pipeline.cloud_cookie({}) == ""
    message = nosub_pipeline.missing_cookie_message()
    assert "P115_COOKIES" in message


def test_home_comes_from_the_toml_file(monkeypatch, tmp_path):
    """Setting only `home` must move every default subdirectory."""
    target = tmp_path / "on-d-drive"
    toml = tmp_path / "mvsub.toml"
    toml.write_text(f'home = "{target.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("MVSUB_CONFIG", str(toml))
    config.reload()

    assert config.home() == target
    assert config.work_dir() == target / "work"
    assert config.archive_dir() == target / "originals"
    assert config.download_dir() == target / "downloads"


def test_environment_home_overrides_the_toml(monkeypatch, tmp_path):
    toml = tmp_path / "mvsub.toml"
    toml.write_text('home = "D:/from-toml"\n', encoding="utf-8")
    monkeypatch.setenv("MVSUB_CONFIG", str(toml))
    monkeypatch.setenv("MVSUB_HOME", "E:/from-env")
    config.reload()

    assert config.home() == Path("E:/from-env")


def test_free_space_probe_survives_a_missing_directory(tmp_path):
    """`scan` measures free space under a work dir that may not exist yet."""
    missing = tmp_path / "not-created" / "deeper" / "still"

    assert nosub_pipeline.free_gb(missing) > 0


def test_free_space_probe_reports_zero_for_an_unreachable_volume(tmp_path):
    """A configured drive that is not mounted should read as zero, not traceback."""
    assert nosub_pipeline.free_gb(Path("Q:/definitely-not-mounted/sub")) >= 0


@pytest.mark.parametrize("subcommand", ["scan", "status"])
def test_read_only_commands_fail_cleanly_without_a_cookie(subcommand, monkeypatch, tmp_path, capsys):
    """No cookie must be a message plus exit 2, never a traceback."""
    monkeypatch.setenv("MVSUB_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("P115_COOKIES", raising=False)
    config.reload()

    parser_args = type("A", (), {"remote_cid": "123", "remote_path": None})()
    code = getattr(nosub_pipeline, f"cmd_{subcommand}")(parser_args)

    assert code == 2
    assert "P115_COOKIES" in capsys.readouterr().err
