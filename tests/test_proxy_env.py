"""`NO_PROXY` handling for the 115 client.

A bracketed IPv6 entry is valid in a hosts file and in a URL, but httpx parses
every ``NO_PROXY`` entry as a URL pattern and rejects ``[::1]``, aborting client
construction with ``Invalid port: ':1]'`` — a message that never mentions
proxies. That killed the first real run on a machine whose proxy config is
perfectly normal.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mvsub" / "pipeline"))

import cloud_common  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_proxy_env():
    saved = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_bracketed_ipv6_entry_is_unwrapped(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,[::1],api.deepseek.com")

    changed = cloud_common.normalize_no_proxy()

    assert changed == ["[::1]"]
    assert os.environ["NO_PROXY"] == "localhost,127.0.0.1,::1,api.deepseek.com"


def test_real_proxy_entries_are_preserved(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "localhost,10.0.0.0/8,*.corp.example")

    assert cloud_common.normalize_no_proxy() == []
    assert os.environ["NO_PROXY"] == "localhost,10.0.0.0/8,*.corp.example"


def test_normalization_is_idempotent(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "[::1],[::2]")

    assert cloud_common.normalize_no_proxy() == ["[::1]", "[::2]"]
    assert cloud_common.normalize_no_proxy() == []


def test_missing_variable_is_not_created(monkeypatch):
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)

    assert cloud_common.normalize_no_proxy() == []
    assert "NO_PROXY" not in os.environ


def test_httpx_client_survives_a_bracketed_entry(monkeypatch):
    """The actual symptom: constructing the client raised before any request."""
    import httpx

    monkeypatch.setenv("NO_PROXY", "localhost,[::1]")
    with pytest.raises(Exception):
        httpx.Client(follow_redirects=True)

    cloud_common.normalize_no_proxy()
    client = httpx.Client(follow_redirects=True)
    client.close()
