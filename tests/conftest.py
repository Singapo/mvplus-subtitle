"""Pytest configuration for the mvsub test suite.

The suite is a straight port of the pipeline's own tests. Six of them fail in the
upstream repository as well (verified before the port); they are marked xfail
here instead of being deleted, so the failures stay visible and cited. None of
them is caused by the port itself — each one is a stale expectation against
current code, or an assertion about the *installed* p115client wire format.

Fixing them means changing pipeline behaviour or constants, which is the
maintainer's call, not something a port should quietly decide.

Every test also runs with credentials isolated: a developer following the README
has a real `.env` in the repository root, and a test that reads it would both
reach the network and print the cookie in a failure diff.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mvsub" / "pipeline"))

import config  # noqa: E402

#: Variables a test must never inherit from the developer's shell.
CREDENTIAL_VARS = (
    "P115_COOKIES",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "MVSUB_CONFIG",
    "MVSUB_HOME",
    "MVSUB_ENV_FILE",
    "MVSUB_MODEL_HOME",
)


@pytest.fixture(autouse=True)
def isolate_credentials(tmp_path_factory, monkeypatch):
    """Point the config at an empty environment for the duration of each test.

    ``MVSUB_ENV_FILE`` is set to a path that does not exist, which is now
    authoritative (see ``config.load_dotenv``): no fallback to the repository's
    ``.env`` can happen. ``os.environ`` is restored afterwards because
    ``load_env`` writes resolved values into it with ``setdefault``.
    """
    absent = tmp_path_factory.mktemp("no-env") / "absent.env"
    snapshot = dict(os.environ)

    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    # Set last: MVSUB_ENV_FILE is itself in CREDENTIAL_VARS above.
    monkeypatch.setenv("MVSUB_ENV_FILE", str(absent))
    config.reload()

    yield

    os.environ.clear()
    os.environ.update(snapshot)
    config.reload()

# node id -> reason
KNOWN_UPSTREAM_GAPS = {
    "test_cloud_subtitle_replace.py::ReplacementTests::test_rename_matches_installed_p115_wire_contract":
        "断言的是本机已安装 p115client 的线上契约；换版本即变，不是项目的固定行为",
    "test_content_verification_scope.py::test_alignment_only_keeps_unit_evidence_without_whole_block_decode":
        "期望按 128 帧取块，当前实现取 512 帧（4 倍漂移，测试未跟随）",
    "test_content_verification_scope.py::test_real_content_risk_keeps_whole_block_and_unit_checks":
        "期望 [256,128]，当前实现 [1024,512]（同上漂移）",
    "test_content_verification_scope.py::test_explicit_all_blocks_still_checks_every_block":
        "期望 [256,128]，当前实现 [1024,512]（同上漂移）",
    "test_content_verification_scope.py::test_changed_model_does_not_reuse_legacy_evidence":
        "期望按 128 帧取块，当前实现取 512 帧（同上漂移）",
    "test_neighbor_clock.py::NeighborClockTests::test_weak_zero_vowel_is_fragment_but_clear_prefix_or_positive_sound_is_not":
        "测试要求 kind=='dialogue' 时不算碎片；实现按 confidence<0.6 直接判碎片，口径待作者确认",
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        for node_id, reason in KNOWN_UPSTREAM_GAPS.items():
            if item.nodeid.endswith(node_id):
                item.add_marker(pytest.mark.xfail(reason=reason, strict=False))
                break
