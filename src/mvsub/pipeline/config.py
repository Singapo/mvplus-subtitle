"""Central configuration for mvsub.

Every path the pipeline needs is resolved here, once. Nothing else in the package
hardcodes a directory.

Resolution order for each key:

1. environment variable (``MVSUB_*``)
2. key in the config file (``$MVSUB_CONFIG``, else ``./mvsub.toml``, else
   ``~/.config/mvsub/config.toml``)
3. default derived from ``home``

The config file is TOML::

    home = "D:/media/mvsub"

    [paths]
    model_home = "D:/models/subtitle"
    archive_dir = "D:/media/originals"
    download_dir = "D:/media/downloads"
    work_dir = "D:/media/work"

    [asr]
    align_python = "D:/envs/align/Scripts/python.exe"

Secrets never live here. They come from the environment or from a ``.env`` file.
"""
from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

__all__ = [
    "home",
    "load_dotenv",
    "dotenv_value",
    "model_home",
    "asr_model",
    "multilingual_model",
    "align_model",
    "align_python",
    "align_libs",
    "model_libs",
    "archive_dir",
    "download_dir",
    "work_dir",
    "repo_root",
    "config_file",
    "reload",
]
# ---------------------------------------------------------------- roots


def home() -> Path:
    """Base directory for every default path.

    Resolution order matters here: a ``home`` in the config file must win over
    the built-in default, otherwise setting it changes nothing and every
    unconfigured subdirectory silently stays under the user's profile.
    """
    return _path("home", Path.home() / ".mvsub")


def repo_root() -> Path:
    """Directory that holds ``.env`` and ``pyproject.toml``.

    A source checkout resolves to the repository itself. A wheel install has no
    such directory, so it falls back to ``$MVSUB_ROOT`` and then the current
    directory — the two places a user would keep their ``.env`` anyway. Assuming
    a fixed ``parents[n]`` here would silently point at ``site-packages``.
    """
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "pyproject.toml").is_file():
        return checkout
    configured = os.environ.get("MVSUB_ROOT")
    if configured:
        return Path(configured).expanduser()
    return Path.cwd()


# ---------------------------------------------------------------- config file


def config_file() -> Path | None:
    """The config file in effect, or ``None`` when no file exists."""
    explicit = os.environ.get("MVSUB_CONFIG")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    for candidate in (
        Path.cwd() / "mvsub.toml",
        Path.home() / ".config" / "mvsub" / "config.toml",
    ):
        if candidate.is_file():
            return candidate
    return None


@lru_cache(maxsize=1)
def _file_config() -> dict:
    path = config_file()
    if path is None:
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _lookup(key: str, section: str = "paths") -> str | None:
    """Return the configured string for ``key``, checking env before the file."""
    env_key = "MVSUB_" + key.upper()
    if value := os.environ.get(env_key):
        return value
    section_values = _file_config().get(section) or {}
    if value := section_values.get(key):
        return value
    if value := _file_config().get(key):
        return value
    return None


def _path(key: str, default: Path, section: str = "paths") -> Path:
    raw = _lookup(key, section)
    if raw is None:
        return default
    return Path(raw).expanduser()


def reload() -> None:
    """Drop cached config values.

    Needed by tests, and by long-lived hosts (the DSH plugin) whose environment
    or config file may change after import.
    """
    _file_config.cache_clear()
    load_dotenv.cache_clear()


# ---------------------------------------------------------------- .env


@lru_cache(maxsize=1)
def load_dotenv() -> dict[str, str]:
    """Parse the first ``.env`` found: ``$MVSUB_ENV_FILE``, repo root, then cwd.

    A deliberately tiny parser: ``KEY=VALUE`` per line, ``#`` comments, optional
    surrounding quotes. It does not touch ``os.environ``; callers decide what to
    export. Missing file yields an empty mapping.
    """
    candidates = [
        Path(os.environ["MVSUB_ENV_FILE"]).expanduser()
        if os.environ.get("MVSUB_ENV_FILE")
        else None,
        repo_root() / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate is None or not candidate.is_file():
            continue
        values: dict[str, str] = {}
        for line in candidate.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, raw = line.partition("=")
            key = key.strip()
            value = raw.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key:
                values[key] = value
        return values
    return {}


def dotenv_value(key: str) -> str:
    """Value of ``key`` from the environment, falling back to the ``.env`` file."""
    return (os.environ.get(key) or load_dotenv().get(key) or "").strip()


# ---------------------------------------------------------------- paths


def model_home() -> Path:
    """Root that holds the model trees (``<root>/models/...`` and ``<root>/align-env``).

    Keeping the upstream layout means an existing deployment can point
    ``MVSUB_MODEL_HOME`` at the directory it already uses, with no move.
    """
    return _path("model_home", home())


def asr_model() -> Path:
    """faster-whisper Japanese model directory (must contain ``model.bin``)."""
    return _path("asr_model", model_home() / "models" / "japanese-turbo")


def multilingual_model() -> Path:
    """Multilingual faster-whisper model used for non-Japanese sources."""
    return _path("multilingual_model", model_home() / "models" / "whisper-large-v3")


def align_model() -> Path:
    """Qwen forced-aligner model directory (must contain ``config.json``)."""
    return _path("align_model", model_home() / "models" / "qwen-aligner")


def align_python() -> Path:
    """Interpreter of the environment that can import the aligner + torch.

    Defaults to ``align-env`` beside the model tree, because the aligner needs a
    different dependency set than the pipeline.
    """
    configured = _lookup("align_python", "asr")
    if configured:
        return Path(configured).expanduser()
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return model_home() / "align-env" / suffix


def align_libs() -> Path:
    """Directory holding the live ``aligned_asr`` package (put on the aligner's path)."""
    return _path("align_libs", Path(__file__).resolve().parent)


def model_libs() -> Path:
    """Extra site-libraries directory shipped beside the models, if any."""
    return _path("model_libs", model_home() / "align-libs")


def archive_dir() -> Path:
    """Where verified Japanese source SRTs are archived locally."""
    return _path("archive_dir", home() / "originals")


def download_dir() -> Path:
    """Default landing directory for downloads from the cloud drive."""
    return _path("download_dir", home() / "downloads")


def work_dir() -> Path:
    """Scratch directory for pipeline state, caches and partial files."""
    return _path("work_dir", home() / "work")
