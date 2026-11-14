"""Unified command line for the subtitle pipeline.

Every subcommand forwards to the module that already owns the behaviour, so the
flags documented in ``docs/`` stay the single source of truth; this file only
maps a short command name onto the right entry point.
"""
from __future__ import annotations

import argparse
import json
import runpy
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from mvsub import __version__

import config  # noqa: E402  (mvsub/__init__ puts src/mvsub/pipeline on sys.path)

# Subcommands that map onto ``nosub_pipeline.py``'s own parser.
NOSUB_COMMANDS = ("scan", "run", "status", "verify")

DESCRIPTION = """\
115 云盘影片补字幕：下载 ∥ 识别 ∥ 翻译 ∥ 回传。

本地单文件、已有字幕重做时间轴、只翻译文字，各用独立子命令；完整参数见 docs/。
"""


def _import(name: str):
    import importlib

    return importlib.import_module(name)


def _call_main(module_name: str, argv: list[str]) -> int:
    """Run ``module_name.main()`` with ``argv`` as its command line."""
    previous = sys.argv
    sys.argv = [module_name, *argv]
    try:
        module = _import(module_name)
        if not hasattr(module, "main"):
            raise SystemExit(f"{module_name} has no main()")
        result = module.main()
        return 0 if result is None else int(result)
    finally:
        sys.argv = previous


def _run_module_as_main(module_name: str, argv: list[str]) -> int:
    """Execute a module's ``if __name__ == '__main__'`` block with ``argv``."""
    previous = sys.argv
    sys.argv = [module_name, *argv]
    try:
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        return 0
    finally:
        sys.argv = previous


# ---------------------------------------------------------------- subcommands


def cmd_nosub(args: argparse.Namespace, argv: list[str]) -> int:
    return _call_main("nosub_pipeline", [args.command, *argv])


def cmd_extract(args: argparse.Namespace, argv: list[str]) -> int:
    return _run_module_as_main("extract_aligned_srt", argv)


def cmd_translate(args: argparse.Namespace, argv: list[str]) -> int:
    return _call_main("translate_srt_batch", argv)


def cmd_cloud(args: argparse.Namespace, argv: list[str]) -> int:
    return _call_main("cloud115", argv)


def cmd_reprocess(args: argparse.Namespace, argv: list[str]) -> int:
    return _run_module_as_main("reprocess_subtitles", argv)


def cmd_review(args: argparse.Namespace, argv: list[str]) -> int:
    return _run_module_as_main("source_review", argv)


def cmd_config(args: argparse.Namespace, argv: list[str]) -> int:
    """Print every resolved path so a broken deployment is diagnosable in one shot."""
    import config

    rows = [
        ("repo_root", config.repo_root()),
        ("config_file", config.config_file() or "(none; defaults + env)"),
        ("model_home", config.model_home()),
        ("asr_model", config.asr_model()),
        ("multilingual_model", config.multilingual_model()),
        ("align_model", config.align_model()),
        ("align_python", config.align_python()),
        ("align_libs", config.align_libs()),
        ("archive_dir", config.archive_dir()),
        ("download_dir", config.download_dir()),
        ("work_dir", config.work_dir()),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:<{width}}  {value}")
    return 0


def cmd_doctor(args: argparse.Namespace, argv: list[str]) -> int:
    """Check the environment the pipeline actually needs, without touching the cloud."""
    import config

    problems: list[str] = []
    notes: list[str] = []

    def check(ok: bool, label: str, detail: str) -> None:
        print(f"{'ok  ' if ok else 'FAIL'}  {label}: {detail}")
        if not ok:
            problems.append(label)

    print(f"mvsub {__version__}")
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    check(sys.version_info >= (3, 12), "python>=3.12", sys.version.split()[0])

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    check(bool(ffmpeg), "ffmpeg", ffmpeg or "not on PATH")
    check(bool(ffprobe), "ffprobe", ffprobe or "not on PATH")

    cookie = config.dotenv_value("P115_COOKIES")
    check(bool(cookie), "P115_COOKIES", "configured" if cookie else "missing (only needed for cloud commands)")
    if cookie:
        parts = {item.split("=", 1)[0].strip().upper() for item in cookie.split(";") if "=" in item}
        missing = {"UID", "CID", "SEID", "KID"} - parts
        if missing:
            notes.append(f"P115_COOKIES 缺少字段: {', '.join(sorted(missing))}")

    key = config.dotenv_value("DEEPSEEK_API_KEY")
    check(bool(key), "DEEPSEEK_API_KEY", "configured" if key else "missing (only needed for paid translation)")

    check(config.asr_model().is_dir(), "asr_model", config.asr_model())
    check((config.asr_model() / "model.bin").is_file(), "asr_model/model.bin",
          config.asr_model() / "model.bin")
    check(config.align_model().is_dir(), "align_model", config.align_model())
    check((config.align_model() / "config.json").is_file(), "align_model/config.json",
          config.align_model() / "config.json")
    check(config.align_python().is_file(), "align_python", config.align_python())
    _check_runtime_imports(check)
    _check_cuda_libraries(check, notes)
    _check_align_environment(check, notes)

    for label, path in (("archive_dir", config.archive_dir()),
                        ("download_dir", config.download_dir()),
                        ("work_dir", config.work_dir())):
        exists = path.is_dir()
        writable = exists and _writable(path)
        check(writable, label, f"{path}{'' if exists else ' (missing)'}")

    if notes:
        print("\nnotes:")
        for note in notes:
            print(f"  - {note}")
    print("\nall checks passed" if not problems else f"\n{len(problems)} check(s) failed: {', '.join(problems)}")
    return 0 if not problems else 1


#: Distribution names the aligner and the verification stages import.
ALIGN_REQUIREMENTS = ("torch", "transformers", "nagisa", "numpy", "scipy")

#: Imported by stages that run under the MAIN interpreter, not the aligner's.
#: `caption` (aligned_asr/caption.py) tokenises Japanese and is one of them; the
#: aligner probe below would never catch a missing nagisa here.
RUNTIME_REQUIREMENTS = ("nagisa",)

#: Models used only by recovery / review paths, not by the shortest Japanese run.
OPTIONAL_MODELS = (
    ("whisper-large-v3", "非日语源、日语时间轴恢复、源文复核的第二识别器"),
    ("qwen-asr-verifier", "局部声学复核"),
    ("audio-event-verifier", "声音事件校验"),
)


def _check_cuda_libraries(check, notes: list[str]) -> None:
    """Find the CUDA DLLs faster-whisper needs before it can decode on GPU.

    They come from ``nvidia-*-cu12`` wheels (site-packages) or from a directory
    you pointed ``MVSUB_CUDA_LIBS`` at. Without them the first extraction dies
    with ``Library cublas64_12.dll is not found or cannot be loaded`` — after the
    download has already been paid for, which is the expensive way to find out.
    """
    site_packages = Path(sysconfig.get_path("purelib")) / "nvidia"
    roots = [site_packages, config.cuda_libs() / "nvidia"]
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for package in root.iterdir():
            bindir = package / "bin"
            if not bindir.is_dir():
                continue
            if bindir.glob("cublas64_*.dll") or bindir.glob("libcublas.so*"):
                found.append(str(bindir))
    if found:
        check(True, "cuda_libs", found[0])
        if len(found) > 1:
            notes.append(f"cuda_libs: {len(found)} candidate dirs; first one is used")
    else:
        note = (
            "未找到 cublas DLL：GPU 提取会报 "
            "`Library cublas64_12.dll is not found`。"
            "解法：安装 nvidia-cublas-cu12 / nvidia-cudnn-cu12，"
            "或用 MVSUB_CUDA_LIBS 指向现有 DLL 目录（nvidia/<pkg>/bin 布局）"
        )
        check(False, "cuda_libs", "not found")
        notes.append(note)


def _check_runtime_imports(check) -> None:
    """Check imports the MAIN interpreter needs, separately from the aligner's.

    The caption stage runs under ``sys.executable``. Checking only the aligner
    environment is how a clean install stayed "green" while the caption stage
    would have failed on ``import nagisa``.
    """
    probe = (
        "import importlib, json\n"
        f"mods = {list(RUNTIME_REQUIREMENTS)!r}\n"
        "missing = []\n"
        "for name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "print(json.dumps(missing))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as error:
        check(False, "runtime_deps", f"cannot probe {sys.executable}: {error}")
        return
    try:
        missing = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        check(False, "runtime_deps", "probe produced no parseable result")
        return
    check(not missing, "runtime_deps",
          "nagisa importable by the main interpreter" if not missing
          else f"missing: {', '.join(missing)} (needed by the caption stage)")


def _check_align_environment(check, notes: list[str]) -> None:
    """Prove the aligner interpreter can import what the stages import.

    Checking only that the file exists is what let a v0.1.0 deployment look
    healthy while every extraction would have died on an ImportError: the
    aligner runs in its own environment, so the main interpreter's packages say
    nothing about it.
    """
    interpreter = config.align_python()
    if not interpreter.is_file():
        return
    probe = (
        "import importlib, json, sys\n"
        f"mods = {list(ALIGN_REQUIREMENTS)!r}\n"
        "missing = []\n"
        "versions = {}\n"
        "for name in mods:\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "        versions[name] = getattr(module, '__version__', '?')\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "print(json.dumps({'missing': missing, 'versions': versions}))\n"
    )
    try:
        result = subprocess.run(
            [str(interpreter), "-c", probe],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError) as error:
        check(False, "align_env", f"cannot run {interpreter}: {error}")
        return
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        check(False, "align_env", detail[-1] if detail else f"probe failed ({result.returncode})")
        return
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        check(False, "align_env", "probe produced no parseable result")
        return
    missing = payload.get("missing") or []
    check(not missing, "align_env", "torch/transformers/nagisa/numpy/scipy importable"
          if not missing else f"missing: {', '.join(missing)}")
    if not missing:
        versions = payload.get("versions") or {}
        notes.append("align_env versions: " + ", ".join(f"{k} {v}" for k, v in versions.items()))

    for model_dir, purpose in OPTIONAL_MODELS:
        path = config.model_home() / "models" / model_dir
        present = path.is_dir() and any(path.iterdir())
        print(f"{'ok  ' if present else 'warn'}  optional {model_dir}: "
              f"{'present' if present else f'absent — {purpose}将会失败'}")
        if not present:
            notes.append(f"{model_dir} 缺失：只影响「{purpose}」，最短日语路径仍可跑")


def _writable(path: Path) -> bool:
    probe = path / ".mvsub-write-probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


# ---------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvsub",
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"mvsub {__version__}")
    sub = parser.add_subparsers(dest="group", metavar="<command>")

    for name, help_text in (
        ("scan", "只读预检：列出目标目录里缺字幕的视频"),
        ("run", "整线：下载 ∥ 提取 ∥ 翻译 ∥ 回传"),
        ("status", "查看批次进度"),
        ("verify", "只读复核一对字幕（本地、不联网）"),
    ):
        sp = sub.add_parser(name, help=help_text, add_help=False)
        sp.set_defaults(handler=cmd_nosub, command=name)

    for name, handler, help_text in (
        ("extract", cmd_extract, "本地单文件：取音 → 识别 → 对齐，产出原文 SRT"),
        ("translate", cmd_translate, "只翻译已有原文 SRT"),
        ("reprocess", cmd_reprocess, "已有字幕时间轴有问题：从源视频重新识别"),
        ("review", cmd_review, "源文声学复核（送翻译前的证据）"),
        ("cloud", cmd_cloud, "115 直连：ls / download / upload"),
    ):
        sp = sub.add_parser(name, help=help_text, add_help=False)
        sp.set_defaults(handler=handler)

    for name, handler, help_text in (
        ("config", cmd_config, "打印全部解析后的路径"),
        ("doctor", cmd_doctor, "体检：解释器、ffmpeg、模型、密钥、目录"),
    ):
        sub.add_parser(name, help=help_text).set_defaults(handler=handler)

    return parser


HANDLERS = {
    "scan": cmd_nosub,
    "run": cmd_nosub,
    "status": cmd_nosub,
    "verify": cmd_nosub,
    "extract": cmd_extract,
    "translate": cmd_translate,
    "reprocess": cmd_reprocess,
    "review": cmd_review,
    "cloud": cmd_cloud,
    "config": cmd_config,
    "doctor": cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"mvsub {__version__}")
        return 0
    command, rest = argv[0], argv[1:]
    handler = HANDLERS.get(command)
    if handler is None:
        print(f"未知子命令: {command}\n", file=sys.stderr)
        build_parser().print_help()
        return 2
    try:
        return handler(argparse.Namespace(command=command), rest)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
