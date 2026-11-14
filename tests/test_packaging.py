"""Packaging contracts that a test run can actually break.

Both checks here map to a defect that shipped once:

* the harness plugin's tarball omitted a module its entry point imports, so an
  installed copy died with ``ERR_MODULE_NOT_FOUND``;
* the main environment never declared ``nagisa``, although the caption stage
  imports it under the *main* interpreter (not the aligner one).
"""
from __future__ import annotations

import fnmatch
import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DSH = ROOT / "dsh"


def _plugin_manifest() -> dict:
    return json.loads((DSH / "package.json").read_text(encoding="utf-8"))


def _relative_imports(entry: Path) -> set[str]:
    """Collect ``./x.js`` specifiers from the entry module."""
    text = entry.read_text(encoding="utf-8")
    found = set()
    for match in re.finditer(r"""from\s+["'](\./[^"']+)["']""", text):
        found.add(match.group(1).removeprefix("./"))
    return found


def test_plugin_ships_every_module_its_entry_imports():
    files = _plugin_manifest().get("files") or []
    assert files, "the plugin must declare what it packs"
    for specifier in _relative_imports(DSH / "index.js"):
        shipped = any(
            entry == specifier or fnmatch.fnmatch(specifier, entry)
            for entry in files
        )
        assert shipped, (
            f"dsh/{specifier} is imported by index.js but not listed in "
            f"package.json files={files}; an installed copy would fail with "
            "ERR_MODULE_NOT_FOUND"
        )


def test_plugin_entry_and_patch_are_packed():
    files = _plugin_manifest()["files"]
    assert "index.js" in files
    assert "cordis.patch.yml" in files


def test_plugin_entry_is_the_declared_main():
    manifest = _plugin_manifest()
    assert manifest["main"] == "index.js"
    assert (DSH / manifest["main"]).is_file()


def test_dependencies_declare_the_caption_stage_tokenizer():
    """`caption.py` runs under the main interpreter and imports nagisa."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = " ".join(data["project"]["dependencies"])
    assert "nagisa" in deps, (
        "the caption stage imports nagisa with the main interpreter; without this "
        "dependency a clean install fails there"
    )
