#!/usr/bin/env python3
"""无字幕影片加字幕流水线（下载 ∥ 提取 ∥ 翻译 ∥ 回传）

SOP 与锁定参数见 subtitle/docs/README.md —— 要改参数请改那份文档，
不要在这里写一次性硬编码。

用法:
    # 只读预检：列出目标目录里缺字幕的视频，不下载
    python subtitle/tools/nosub_pipeline.py scan   --remote-cid 3462599370287875569

    # 正式跑（断点续跑：中断后原样重跑同一命令）
    python subtitle/tools/nosub_pipeline.py run    --remote-cid 3462599370287875569

    # 看进度
    python subtitle/tools/nosub_pipeline.py status --remote-cid 3462599370287875569

    # 只跑指定几部（试跑验收）
    python subtitle/tools/nosub_pipeline.py run --remote-cid <cid> --only IPVR-008 --limit 1

    # 只读复核一对字幕（本地、不联网、不写状态）——SOP §3.1 步骤⑥ 的可自动化判据
    python subtitle/tools/nosub_pipeline.py verify <video|-> <原文.ja.srt> <中文.srt>
    # 退出码：0 = 硬判据全过；1 = 有硬判据失败；2 = 参数/文件缺失

115 接口事实（本机实测，别再踩，详见 subtitle/docs/问题与解决方案.md §6）:
    * 下载直链：P115FileSystem.get_download_info_from_pickcode(pc, user_agent="Mozilla/5.0 115disk/30.5.1")
      走 app/chrome 加密接口；纯 web 接口对大文件报「文件大小超出限制，请使用115电脑端下载」；
      app 接口不带该 UA 会报「请重新登录」。
    * 上传：P115UploadHelper(client).upload_file_sample(path, pid=cid, filename=名字)，
      同名不会覆盖，会并存两份同名文件。
    * upload_gettoken 必须打补丁：legacy 接口 uplb.115.com/3.0/gettoken.php 返回扁平体
      （无 state / 无 data），p115sync 的信封校验会把它误判为失败 -> P115APIError: 115 API 请求失败。
      该 bug 只在 status=1（真传字节）时触发，秒传命中(status=2)时根本不调用它，所以极具误导性。
    * 改名：client.fs_rename((fid, 新名))；删除：client.fs_delete(int(fid))。

115 补丁与接口知识统一在 `cloud_common.py`（subtitle 与 demosaic 两条线共用，只实现一次）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ------------------------------------------------- 固定参数（与 SOP §4「锁定参数」表一一对应）
DL_WORKERS = 3
EXTRACT_WORKERS = 1
TRANSLATE_WORKERS = 2
UPLOAD_WORKERS = 2
MAX_QUEUED_DL = 3

MODEL_ASR = "efwkjn/faster-whisper-ja-760M + Qwen3-ForcedAligner-0.6B-hf"

MODEL_TRANSLATE = "deepseek-flash"
TRANSLATE_URL = "https://api.deepseek.com"
TRANSLATE_EXTRA = ["--grounded", "--batch-size", "16", "--max-workers", "2",
                   "--max-output-tokens", "16384", "--reasoning-effort", "high",
                   "--timeout", "660", "--off-peak-only"]
PAID_ARGS = []

# 收尾看门狗的各级超时（秒）：超过才认为该级真卡住并补终止符
REAPER_SLACK = {
    "download": 1800,   # 下载可能很慢（单部数 GB），给足时间
    "extract": 1800,    # GPU 提取长片可到十几分钟
    "translate": 900,   # 单部字幕翻译几分钟
    "upload": 600,
}

DL_CHUNK = 1 << 20
DISK_FLOOR_GB = 15          # 下载途中的磁盘硬下限，低于此值暂停等提取腾空间
MIN_SRT_BYTES = 100
ORIG_ARCHIVE_DIR = config.archive_dir()   # 本地原文归档目录（不上云端）

# ---- 验收判据（H2，2026-09-22 新增；改这些值必须先读 SOP §3.1 步骤⑥ 与问题文档 §13.17）
HASH_CHUNK = 1 << 20            # 流式 sha1 块大小（不把 srt 读进内存；原文归档件也只有几十 KB，但保持统一写法）
CUE_TAIL_TOLERANCE = 5.0        # 字幕起止时间允许超出视频时长的容差（秒）：编码/尾帧延迟的经验值，未逐部实测校准
TIMELINE_TOL_MS = 0             # 中/日时间轴允许偏差（毫秒）：现行实现原样拷贝时间戳，实测 31479 条 cue 无偏差

# ---- `verify` 子命令的判据阈值（口径唯一权威 = SOP §3.1 步骤⑥；下面每个常量对应该节的一个数字，别另立一套）
KANA_RE = r"[\u3040-\u30ff]"    # ⑥.2：假名/日文残留（判据：必须为 0）
SINGLE_CHAR_RUN_RE = r"(.)\1{7,}"   # ⑥.3：单字长重复（≥8 次）——**提示级**，需人工确认
CONSEC_DUP_MIN_CHARS = 4        # ⑥.3：连续重复（弱信号）——只提示长度 ≥ 4 字的连续同句
CPS_MIN_CUE_SEC = 1.0           # ⑥.4：只判时长 ≥ 1.0s 的 cue（更短的另计数、不参与 CPS 判定）
CPS_WARN = 17.0                 # ⑥.4：> 17 CPS → 人工看一眼
CPS_MUST_LOOK = 25.0            # ⑥.4：> 25 CPS → 必看

# 批次收尾汇总的状态顺序（`cmd_run` 结束时的「本轮结果」那一行；固定顺序 = 每次长一样，便于 diff/截图）
BATCH_STATUS_ORDER = ("uploaded", "skipped_verify_failed", "skipped_archive_failed",
                      "skipped_missing_translation", "skipped_missing_video",
                      "skipped_download", "skipped_disk", "skipped_extract_failed",
                      "skipped_empty_srt", "skipped_translate_failed",
                      "skipped_name_exists", "skipped_upload_failed")


def now() -> str:
    return time.strftime("%m-%d %H:%M:%S")


log_lock = threading.Lock()
log_fh = None


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    with log_lock:
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")
            log_fh.flush()


def log_flush() -> None:
    with log_lock:
        sys.stdout.flush()
        if log_fh:
            log_fh.flush()


# ------------------------------------------------- 路径
TOOLS_DIR = Path(__file__).resolve().parent
SUBTITLE = config.repo_root()
PROJECT = SUBTITLE
PY = sys.executable
PROMPT = TOOLS_DIR / "prompts" / "system_prompt_v9_grounded.txt"
WORK_ROOT = config.work_dir() / "nosub"
VIDEO_EXT = [".mp4", ".mkv", ".ts", ".wmv", ".avi", ".mov", ".m2ts"]
SUB_EXT = [".srt", ".ass", ".vtt", ".ssa"]


def load_env() -> dict[str, str]:
    """Resolve credentials and export them for child processes.

    Priority is environment variable first, then the `.env` file, matching
    `config.dotenv_value()` and what the docs promise. Returning only the file
    contents (the earlier behaviour) meant an env-only deployment died with a
    bare `KeyError` on the very first scan.
    """
    env: dict[str, str] = dict(config.load_dotenv())
    for key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "P115_COOKIES"):
        value = (os.environ.get(key) or env.get(key) or "").strip()
        if value:
            env[key] = value
            os.environ.setdefault(key, value)
    return env


def cloud_cookie(env: dict[str, str] | None = None) -> str:
    """The 115 cookie, or a message-free empty string the caller must handle."""
    resolved = env if env is not None else load_env()
    return (resolved.get("P115_COOKIES") or config.dotenv_value("P115_COOKIES") or "").strip()


def missing_cookie_message() -> str:
    return (
        "缺少 P115_COOKIES：设成环境变量，或写进 "
        f"{config.repo_root() / '.env'}（或 $MVSUB_ENV_FILE 指向的文件）"
    )


def safe_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(". ") or "root"


def stem_of(name: str) -> str:
    """去掉扩展名：IPVR-008.mp4 -> IPVR-008（字幕与视频同名就靠它配对）。"""
    return re.sub(r"\.[^.]+$", "", name)


def human(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def free_gb(path: Path) -> float:
    """Free space for the volume holding ``path``.

    ``path`` often does not exist yet on a fresh deployment (the work directory is
    created on the first run). ``disk_usage`` on a missing path raises
    ``FileNotFoundError``, so walk up to the nearest existing ancestor: that is
    the same volume, and it keeps read-only commands like ``scan`` read-only
    instead of creating directories just to measure them.
    """
    probe = Path(path)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free / 1024 ** 3
    except OSError:
        # Unreachable volume (e.g. a configured drive that is not mounted):
        # report zero rather than traceback, so the caller's threshold produces
        # a message a user can act on.
        return 0.0


# ------------------------------------------------- 115 客户端（补丁与接口知识都在 cloud_common.py）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud_common import Cloud115  # noqa: E402


class Cloud(Cloud115):
    """在本流水线里加一层带 TTL 的目录缓存（减少重复列目录请求）。"""

    def __init__(self, cookie: str):
        super().__init__(cookie, log=log)
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._cache_lock = threading.Lock()

    def list_dir(self, cid, ttl: float = 300.0) -> list[dict]:
        key = str(cid)
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit and time.time() - hit[0] < ttl:
                return hit[1]
        items = super().list_dir(cid)
        with self._cache_lock:
            self._cache[key] = (time.time(), items)
        return items

    def invalidate(self, cid) -> None:
        with self._cache_lock:
            self._cache.pop(str(cid), None)


# ------------------------------------------------- 状态（断点续跑；key 全是字符串，保证 JSON 往返一致）
class State:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.items: dict[str, dict] = {}
        self.uploaded: dict[str, dict] = {}     # "zh::名字" / "orig::名字" -> {fid, cid}
        if path.exists():
            try:
                blob = json.loads(path.read_text(encoding="utf-8"))
                self.items = blob.get("items", {})
                self.uploaded = blob.get("uploaded", {})
            except (json.JSONDecodeError, OSError):
                pass

    def save(self) -> None:
        with self.lock:
            blob = {"updated": now(), "items": self.items, "uploaded": self.uploaded}
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)

    def set(self, stem: str, status: str, **extra) -> None:
        with self.lock:
            rec = self.items.setdefault(stem, {})
            rec["status"] = status
            rec["at"] = now()
            rec.update(extra)
        self.save()

    def get(self, stem: str) -> dict:
        return self.items.get(stem, {})


# ------------------------------------------------- 盘点
def survey(cloud: Cloud, cid) -> dict:
    """列目录 → 分出视频/字幕/子目录，配对出「缺字幕」的视频。"""
    items = cloud.list_dir(cid, ttl=0)
    videos, subs, dirs = [], [], []
    for it in items:
        if it.get("fid") is None:
            dirs.append(it)
            continue
        ext = Path(it["n"]).suffix.lower()
        if ext in VIDEO_EXT:
            videos.append(it)
        elif ext in SUB_EXT:
            subs.append(it)
    sub_by_stem: dict[str, list[dict]] = {}
    for s in subs:
        sub_by_stem.setdefault(stem_of(s["n"]), []).append(s)
    paired, missing = [], []
    for v in videos:
        st = stem_of(v["n"])
        (paired if st in sub_by_stem else missing).append(v)
    video_stems = {stem_of(v["n"]) for v in videos}
    orphans = [s for st, g in sub_by_stem.items() if st not in video_stems for s in g]
    return {"all": items, "videos": videos, "subs": subs, "dirs": dirs,
            "paired": paired, "missing": missing, "orphans": orphans}


def find_on_disk(workdir: Path, stem: str) -> Path | None:
    for ext in VIDEO_EXT:
        p = workdir / f"{stem}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


# ------------------------------------------------- 四段工人
def do_download(item, workdir, cloud, state, extract_q, stop_event) -> None:
    stem = item["stem"]
    existing = find_on_disk(workdir, stem)
    if existing:
        log(f"[{stem}] 本地已有 {existing.name}（{human(existing.stat().st_size)}），跳过下载")
        state.set(stem, "downloaded", file=existing.name, size=existing.stat().st_size)
        extract_q.put(item)
        return

    url, headers = cloud.download_url(item["pc"])
    size = int(item.get("s") or 0)
    need_gb = size * 1.5 / 1024 ** 3
    if free_gb(workdir) < need_gb:
        log(f"[{stem}] 磁盘水位不足（可用 {free_gb(workdir):.1f} GB < {need_gb:.1f} GB），等 60s")
        for _ in range(20):
            if stop_event.wait(30):
                return
            if free_gb(workdir) >= need_gb:
                break
        else:
            state.set(stem, "skipped_disk", size=size)
            log(f"[{stem}] 磁盘持续不足，跳过（重跑会再试）")
            return

    dst = workdir / item["n"]
    part = dst.with_suffix(dst.suffix + ".part")
    for attempt in range(3):
        # 断点续传：.part 已有内容就带 Range 从断点继续，失败不再从头重下。
        # 115 的签名地址每次重新获取，Range 按字节偏移续传。
        have = part.stat().st_size if part.exists() else 0
        if size and have >= size * 0.99:
            log(f"[{stem}] .part 已完整（{human(have)}），跳过下载")
        else:
            if have:
                log(f"[{stem}] 断点续传 {human(have)} / {human(size)}")
            else:
                log(f"[{stem}] 下载(第{attempt + 1}次) {item['n']} {human(size)}")
            req_headers = dict(headers)
            if have:
                req_headers["Range"] = f"bytes={have}-"
            try:
                done = 0
                with httpx.stream("GET", url, headers=req_headers,
                                  timeout=httpx.Timeout(30, read=120), follow_redirects=True) as r:
                    r.raise_for_status()
                    # 服务器不认 Range 会回 200 + 全量，此时必须覆盖写而非追加
                    resume = bool(have) and r.status_code == 206
                    if have and not resume:
                        log(f"[{stem}] 服务器未支持续传（HTTP {r.status_code}），从头下载")
                    mode = "ab" if resume else "wb"
                    done = have if resume else 0
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(DL_CHUNK):
                            if stop_event.is_set():
                                log(f"[{stem}] 收到停止信号，下载中断（.part 保留）")
                                return
                            f.write(chunk)
                            done += len(chunk)
                            if done % (1024 << 20) < DL_CHUNK:
                                log(f"[{stem}]   已下 {human(done)} / {human(size)}")
                                # 下载途中也要守住磁盘水位：多路并发时不会等新文件
                                # 才开始检查，必须在这里拦，否则会把盘写满。
                                while free_gb(workdir) < DISK_FLOOR_GB:
                                    log(f"[{stem}] 磁盘仅剩 {free_gb(workdir):.1f} GB，"
                                        f"暂停下载等提取腾空间")
                                    if stop_event.wait(60):
                                        log(f"[{stem}] 收到停止信号，下载中断（.part 保留）")
                                        return
                                    if part.stat().st_size >= (size or 0) * 0.99:
                                        break
            except Exception as e:  # noqa: BLE001
                log(f"[{stem}] 下载出错: {type(e).__name__}: {str(e)[:150]}")
                continue

        got = part.stat().st_size if part.exists() else 0
        if size and got < size * 0.99:
            log(f"[{stem}] 大小不足（{human(got)} vs {human(size)}），重下")
            continue
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                str(part)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if probe.returncode != 0:
            log(f"[{stem}] ffprobe 校验失败（moov 缺失/损坏），重下")
            continue
        part.replace(dst)
        state.set(stem, "downloaded", file=dst.name, size=dst.stat().st_size)
        log(f"[{stem}] 下载完成 {human(dst.stat().st_size)}")
        extract_q.put(item)
        return
    state.set(stem, "skipped_download", size=size)
    log(f"[{stem}] 下载 3 次均失败，跳过（重跑会再试）")


def do_extract(item, workdir, state, translate_q) -> None:
    stem = item["stem"]
    video = find_on_disk(workdir, stem)
    if not video:
        log(f"[{stem}] 提取阶段找不到视频，跳过")
        state.set(stem, "skipped_missing_video")
        return
    srt = video.with_suffix(".srt")
    from reprocess_subtitles import require_current_timing
    alignment_work=srt.parent/(srt.stem+'.alignment')
    try:
        require_current_timing(srt,work=alignment_work)
        current_source=True
    except (OSError,ValueError,RuntimeError):current_source=False
    if not current_source:
        if srt.exists():
            prior=alignment_work/'prior_sources'/(sha1_of(srt)[0]+'.srt')
            prior.parent.mkdir(parents=True,exist_ok=True)
            if not prior.exists():shutil.copy2(srt,prior)
            log(f'[{stem}] 旧原文已留存，重新验证提取')
        cmd = [PY, str(TOOLS_DIR / "extract_aligned_srt.py"), str(video), "--output", str(srt)]
        log(f"[{stem}] 提取日文字幕（{MODEL_ASR}）")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            log(f"[{stem}] 提取失败 rc={r.returncode} {(r.stderr or '')[-300:]}")
            state.set(stem, "skipped_extract_failed", rc=r.returncode)
            return
    if not srt.exists() or not parse_srt_cues(srt):
        log(f"[{stem}] 提取为空（音频异常），保留视频与现场等待复核")
        state.set(stem, "skipped_empty_srt")
        return
    require_current_timing(srt,work=alignment_work)
    # Keep the extracted full audio before any later upload/video cleanup.
    import wave
    from audio_archive import archive_audio
    audio=srt.parent/(srt.stem+'.alignment')/'full_audio.wav'
    if audio.exists():
        try:
            archived_audio,meta=archive_audio(audio,stem,item.get('fid'))
            state.set(stem,'extracted',audio_archive=str(archived_audio),audio_archive_sha1=meta['sha1'])
        except (OSError,ValueError,EOFError,wave.Error) as error:
            state.set(stem,'skipped_archive_failed',reason=str(error)[:200])
            return
        from source_review import ensure_review
        ensure_review(srt,audio,audio.parent,audio.parent/'raw_asr.json')
    else:
        state.set(stem,'waiting_source_review',reason='Missing original audio for source review')
        return
    state.set(stem, "extracted", srt_bytes=srt.stat().st_size)
    log(f"[{stem}] 提取完成 {srt.stat().st_size} B")
    translate_q.put(item)


def do_translate(item, workdir, state, upload_q) -> None:
    stem = item["stem"]
    video = find_on_disk(workdir, stem)
    if not video:
        state.set(stem, "skipped_missing_video")
        return
    srt = video.with_suffix(".srt")
    zh = srt.with_name(srt.stem + ".zh-cn.srt")
    if not PAID_ARGS:
        raise RuntimeError('Paid workflow requires explicit shared budget configuration')
    review=srt.parent/(srt.stem+'.alignment')/'source_review.json'
    if not zh.exists():
        cmd = [PY, str(TOOLS_DIR / "translate_srt_batch.py"), str(srt),
               "--model", MODEL_TRANSLATE, "--source", "ja", "--target", "zh-cn",
               "--url", TRANSLATE_URL, "--system-prompt", str(PROMPT),
               '--source-review',str(review),*TRANSLATE_EXTRA,*PAID_ARGS]
        log(f"[{stem}] 翻译（{MODEL_TRANSLATE}）")
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode==3:
            state.set(stem,'waiting_source_review');return
        if r.returncode != 0 or not zh.exists():
            tail = (r.stderr or r.stdout or "")[-300:]
            log(f"[{stem}] 翻译失败 rc={r.returncode} {tail}")
            state.set(stem, "skipped_translate_failed", rc=r.returncode)
            return
    from grounded_translation import require_audit
    require_audit(srt,zh,review)
    item['_grounded_audit_required']=True
    state.set(stem, "translated", zh=zh.name)
    log(f"[{stem}] 翻译完成 {zh.stat().st_size} B")
    upload_q.put(item)


# ------------------------------------------------- 校验（验收判据，H1/H2 共用）
def sha1_of(path: Path, chunk: int = HASH_CHUNK) -> tuple[str, int]:
    """流式算 sha1，并返回**实际读到的字节数**（不把文件读进内存）。"""
    h = hashlib.sha1()
    size = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def parse_srt_cues(path: Path) -> list[tuple[str, float, float, str]]:
    """解析 SRT → [(序号文本, 开始秒, 结束秒, 文本)]；解析不出的块直接丢弃（丢光 = cue 数 0 = 校验失败）。

    与 `translate_srt_batch.parse_srt()` 的差别：**保留空文本 cue**（只裁掉块首/块尾空行，不吞块内结构），
    否则「无空文本 cue」这条判据会被解析层吃掉、永远查不出来。
    """
    from subtitle_pipeline_core import _read_srt
    content = _read_srt(path).replace("\r\n", "\n").replace("\r", "\n")
    cues: list[tuple[str, float, float, str]] = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [ln.rstrip() for ln in block.split("\n")]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if len(lines) < 2:
            continue
        m = re.match(r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})\s*-->\s*"
                     r"(\d+):(\d\d):(\d\d)[,.](\d{1,3})", lines[1].strip())
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        cues.append((lines[0].strip(),
                     g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000,
                     g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000,
                     " ".join(ln.strip() for ln in lines[2:]).strip()))
    return cues


def probe_duration(video: Path | None) -> float | None:
    """ffprobe 取视频时长（秒）；取不到返回 None（调用方把「时长范围」这一项记为**未做**，不是通过）。"""
    if video is None:
        log("    ⚠ 未提供视频 →「起止在片长内」这一项记为**未做**")
        return None
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nw=1:nk=1", str(video)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError) as e:  # noqa: BLE001
        log(f"    ⚠ ffprobe 取时长失败: {type(e).__name__}: {str(e)[:120]}")
        return None
    if r.returncode != 0:
        log(f"    ⚠ ffprobe rc={r.returncode}，本次不做「时长范围」校验: {(r.stderr or '')[:120]}")
        return None
    try:
        return float((r.stdout or "").strip())
    except ValueError:
        log(f"    ⚠ ffprobe 输出无法解析为时长: {(r.stdout or '')[:60]!r}")
        return None


def _check_srt(path: Path, label: str, duration: float | None) -> tuple[str, list[tuple[str, float, float, str]]]:
    """单个 SRT 的结构校验；返回 ("" 或失败原因, cues)。"""
    if not path.exists():
        return f"{label} 不存在: {path}", []
    try:
        cues = parse_srt_cues(path)
    except OSError as e:  # noqa: BLE001
        return f"{label} 读取失败 {path}: {type(e).__name__}: {str(e)[:100]}", []
    if not cues:
        return f"{label} 解析不出任何 cue（文件不是合法 SRT 或为空）: {path.name}", []
    prev_start = -1.0
    for idx, start, end, text in cues:
        if not text.strip():
            return f"{label} 第 {idx} 条 cue 文本为空", cues
        if end <= start:
            return f"{label} 第 {idx} 条 cue 结束时间 <= 开始时间（{start:.3f} → {end:.3f}）", cues
        if start < prev_start:
            return f"{label} 第 {idx} 条 cue 时间轴非单调递增（{start:.3f} < 上一条 {prev_start:.3f}）", cues
        prev_start = start
        if duration is not None and (start > duration + CUE_TAIL_TOLERANCE
                                     or end > duration + CUE_TAIL_TOLERANCE):
            return (f"{label} 第 {idx} 条 cue 超出视频时长（{start:.3f}→{end:.3f}，"
                    f"片长 {duration:.3f}s + 容差 {CUE_TAIL_TOLERANCE}s）"), cues
    return "", cues


def validate_subtitles(srt: Path, zh: Path, video: Path | None) -> tuple[bool, str]:
    """**删除本地文件之前**的验收（H2）。返回 (是否通过, 判据摘要 / 失败原因)。

    判据（全部本地完成，不联网；做不到的项在摘要里明确写「未做」）：
      ① SRT 可解析、cue 数 > 0；
      ② 无空文本 cue；end > start；start 单调递增；
      ③ 起止时间都在源视频时长内（容差 `CUE_TAIL_TOLERANCE` 秒；ffprobe 取不到时长 → 该项「未做」）；
      ④ 中文 cue 数与时间轴与原文 `*.ja.srt` 逐条一致（容差 `TIMELINE_TOL_MS` 毫秒）；
      ⑤ **中文里没有假名/日文残留**（`KANA_RE`，与 SOP §3.1 步骤⑥.2 同一条判据：必须为 0）。
         —— 2026-09-24 补：这一项原先只在 `verify`（事后复核）里查，**删本地文件的闸门不查**，
         结果某部片有一行 `ひぶきちゃん…` 漏译仍发成了成品（整批里唯一一处）。模型偶发会把
         日文专名原样留下（提示词第 17 条本就禁止），只靠事后复核等于让它出厂，所以在
         **上传与清理之前**就拦下。
    云端一致性在 `do_upload()` 里上传后另做（体积 + 能拿到 sha1 就一起核对）。
    """
    duration = probe_duration(video)
    if video is not None and duration is None:
        # 2026-09-24 收口径：生产路径（do_upload）一定拿得到视频文件，所以「取不到片长」
        # 不是"跳过"，而是异常 —— 按失败处理（保留本地、不上传），避免整段时长范围无人校验
        # 却仍记成"验收通过"。只想复核字幕结构时用 `verify -`（显式不提供视频）。
        return False, (f"ffprobe 取不到片长（{video}）→「起止在片长内」无法执行，"
                       f"生产路径按失败处理（不复核跳过；如只想校对字幕结构请用 `verify -`）")
    why, ja_cues = _check_srt(srt, "原文", duration)
    if why:
        return False, why
    why, zh_cues = _check_srt(zh, "中文", duration)
    if why:
        return False, why
    kana = [i for i, ln in enumerate(zh.read_text(encoding="utf-8", errors="replace").splitlines(), 1)
            if re.search(KANA_RE, ln)]
    if kana:
        return False, (f"中文里仍有假名/日文残留 {len(kana)} 行（⑥.2 判据：必须为 0；"
                       f"首处在第 {kana[0]} 行）——模型偶发把日文专名原样留下，需处理后再传")
    if len(ja_cues) != len(zh_cues):
        return False, f"原/译条数不一致（原文 {len(ja_cues)} 条，中文 {len(zh_cues)} 条）"
    tol = TIMELINE_TOL_MS / 1000.0
    for (ji, js, je, _), (zi, zs, ze, _) in zip(ja_cues, zh_cues):
        if abs(js - zs) > tol or abs(je - ze) > tol:
            return False, (f"时间轴与原/译不一致（原文第 {ji} 条 {js:.3f}→{je:.3f}，"
                           f"中文第 {zi} 条 {zs:.3f}→{ze:.3f}，容差 {TIMELINE_TOL_MS}ms）")
    note = (f"cue {len(zh_cues)} / 时间轴一致（ja↔zh 逐条，容差 {TIMELINE_TOL_MS}ms）"
            f" / 无空文本、单调递增")
    note += (f" / 起止均在片长内（{duration:.1f}s + 容差 {CUE_TAIL_TOLERANCE}s）"
             if duration is not None else
             " / 时长范围校验：**未做**（未提供视频 → 显式跳过；生产路径不允许跳过）")
    return True, note


def verify_subtitles(video: Path | None, orig: Path, zh: Path) -> dict:
    """SOP §3.1 步骤⑥ 的**本地可自动化判据**：纯只读、不联网、不写状态（`nosub_pipeline.py verify` 用它）。

    判据与阈值**唯一权威在 SOP §3.1 步骤⑥**（⑥.2 假名残留 / ⑥.3 重复幻觉 / ⑥.4 CPS / ⑥.5 首中尾人工项），
    本函数只负责执行，本文件顶部的常量与那节的数字一一对应 —— 改阈值必须同时改文档。

    返回 dict：
      ok        —— 硬判据是否全过（假名残留>0、或结构/时间轴/片长不过 → False）
      summary   —— 一行摘要（通过时复用 `validate_subtitles()` 的判据串）
      failures  —— 硬判据失败原因列表
      hints     —— **提示项**（重复/CPS/超短 cue/与原文同句）：需人工确认，**不影响 ok**
      details   —— 明细行（每条带 cue 序号与内容片段，便于直接抽查）
      stats     —— 计数
    """
    failures: list[str] = []
    hints: list[str] = []
    details: list[str] = []
    stats: dict[str, int] = {}

    ok_struct, verdict = validate_subtitles(orig, zh, video)
    if not ok_struct:
        failures.append(verdict)

    try:
        zh_cues = parse_srt_cues(zh)
        orig_cues = parse_srt_cues(orig)
        raw = zh.read_text(encoding="utf-8", errors="replace")
    except OSError as e:  # noqa: BLE001
        return {"ok": False, "summary": f"读取失败: {type(e).__name__}: {str(e)[:120]}",
                "failures": [f"读取字幕失败: {type(e).__name__}: {str(e)[:120]}"],
                "hints": [], "details": [], "stats": {}}

    # ⑥.2 假名/日文残留 —— 硬判据：必须为 0
    kana = [(i, ln.strip()) for i, ln in enumerate(raw.splitlines(), 1) if re.search(KANA_RE, ln)]
    stats["kana_lines"] = len(kana)
    if kana:
        failures.append(f"假名/日文残留 {len(kana)} 行（⑥.2 判据：必须为 0）")
        details += [f"  假名残留 行{i}: {t[:60]}" for i, t in kana[:5]]

    # ⑥.3 单字长重复（提示级）与连续重复（弱信号）
    runs = [(i, t) for i, (_idx, _s, _e, t) in enumerate(zh_cues, 1)
            if re.search(SINGLE_CHAR_RUN_RE, t)]
    consec = [(i, t) for i, (_idx, _s, _e, t) in enumerate(zh_cues[1:], 2)
              if t and t == zh_cues[i - 2][3] and len(t) >= CONSEC_DUP_MIN_CHARS]
    stats["single_char_runs"] = len(runs)
    stats["consec_dups"] = len(consec)
    if runs:
        hints.append(f"单字长重复 {len(runs)} 条（≥8 次，⑥.3 提示级 → 人工确认是幻觉还是「啊啊啊」类台词）")
        details += [f"  单字长重复 #{i}: {t[:60]}" for i, t in runs[:10]]
    if consec:
        hints.append(f"连续重复 {len(consec)} 条（≥{CONSEC_DUP_MIN_CHARS} 字，⑥.3 弱信号）")
        details += [f"  连续重复 #{i}: {t[:60]}" for i, t in consec[:5]]

    # ⑥.2 附带：整条与原文完全相同的 cue（脚本层本会拦，这里报数便于抽查老成品）
    same = [(i, t) for i, (_idx, _s, _e, t) in enumerate(zh_cues, 1)
            if i <= len(orig_cues) and t and t == orig_cues[i - 1][3]]
    stats["identical_to_orig"] = len(same)
    if same:
        hints.append(f"与原文完全相同的 cue {len(same)} 条（可能未译，⑥.2 附带项）")
        details += [f"  与原文相同 #{i}: {t[:60]}" for i, t in same[:5]]

    # ⑥.4 阅读速度：只判时长 ≥ CPS_MIN_CUE_SEC 的 cue
    cps: list[tuple[float, int, str]] = []
    short: list[tuple[int, float]] = []
    for i, (_idx, start, end, text) in enumerate(zh_cues, 1):
        dur = end - start
        if dur < CPS_MIN_CUE_SEC:
            short.append((i, dur))
            continue
        if not text:
            continue
        cps.append((len(re.sub(r"\s", "", text)) / dur, i, text))
    over_warn = sorted([c for c in cps if c[0] > CPS_WARN], reverse=True)
    stats["cps_judged"] = len(cps)
    stats["cps_over_warn"] = len(over_warn)
    stats["cps_over_must_look"] = sum(1 for c in cps if c[0] > CPS_MUST_LOOK)
    stats["short_cues"] = len(short)
    if over_warn:
        hints.append(f"> {CPS_WARN:g} CPS 的 cue {len(over_warn)} 条"
                     f"（其中 > {CPS_MUST_LOOK:g} CPS {stats['cps_over_must_look']} 条 → 必看，⑥.4）")
        details += [f"  {c:.1f} CPS #{i}: {t[:40]}" for c, i, t in over_warn[:5]]
    if short:
        hints.append(f"超短 cue {len(short)} 条（< {CPS_MIN_CUE_SEC:g}s，不参与 CPS 判定，⑥.4）")

    hints.append("人工抽查项（无法自动化）：首/中/尾内容与画面对照、称谓与口气（SOP §3.1 步骤⑥.5）")
    ok = not failures
    summary = verdict if ok_struct else "；".join(failures)
    return {"ok": ok, "summary": summary, "failures": failures, "hints": hints,
            "details": details, "stats": stats}


def archive_original(srt: Path, archive_dir: Path | None) -> tuple[Path | None, str]:
    """把提取出的日文原文字幕归档到**本地**目录（不上传云端），并做「sha1 + 体积」校验。

    2026-09-22 流程变更：用户要求「只上传中文的，日文保存本地一个文件夹里」。
    默认落点由 `config.archive_dir()` 决定，可用 `--orig-dir` 或 `MVSUB_ARCHIVE_DIR` 覆盖。

    返回 `(归档件路径, 说明)`；**路径为 None 表示归档未确认成功**，说明里带失败原因。
    调用方（`do_upload()`）必须把 None 当失败：不清理本地原文、不上传，记 `skipped_archive_failed`。

    同名策略（写清，不静默）：
      * 同名**且内容一致**（sha1 与体积都比）→ 复用，不复制、不产生 `.prev`，日志写明「内容一致」；
      * 同名**但内容不同** → 旧版改名成 `<名>.prev`（上一版 `.prev` 会被覆盖），新版就位；
      * 复制走 `<名>.part` 临时件：复制/校验失败只删这个临时件，**既有归档件与本地原文都不动**。
    """
    if archive_dir is None:
        return None, "未启用本地归档目录（--orig-dir 缺失或默认目录不可用）"
    if not srt.exists():
        return None, f"本地原文不存在: {srt}"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:  # noqa: BLE001
        return None, f"归档目录不可用 {archive_dir}: {type(e).__name__}: {str(e)[:120]}"
    if not archive_dir.is_dir():
        return None, f"归档目标不是目录: {archive_dir}"

    target = archive_dir / f"{srt.stem}.ja.srt"
    try:
        src_sha, src_size = sha1_of(srt)
    except OSError as e:  # noqa: BLE001
        return None, f"读取本地原文失败 {srt}: {type(e).__name__}: {str(e)[:120]}"
    if src_size <= 0:
        return None, f"本地原文为空（0 B）: {srt}"

    if target.exists():
        try:
            tgt_sha, tgt_size = sha1_of(target)
        except OSError as e:  # noqa: BLE001
            return None, f"读取既有归档件失败 {target}: {type(e).__name__}: {str(e)[:120]}"
        if tgt_sha == src_sha and tgt_size == src_size:
            log(f"    归档件已存在且内容一致（sha1={src_sha[:12]}… {src_size} B），跳过复制: {target}")
            return target, f"复用既有归档件 sha1={src_sha} 体积={src_size}B"

    tmp = target.with_name(target.name + ".part")
    tmp.unlink(missing_ok=True)          # 只清我们自己可能残留的临时件
    try:
        shutil.copy2(srt, tmp)
    except OSError as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None, f"复制到归档目录失败 {tmp}: {type(e).__name__}: {str(e)[:120]}"
    try:
        tmp_sha, tmp_size = sha1_of(tmp)
    except OSError as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None, f"读取归档临时件失败 {tmp}: {type(e).__name__}: {str(e)[:120]}"
    if tmp_sha != src_sha or tmp_size != src_size:
        tmp.unlink(missing_ok=True)
        return None, (f"归档件与原文件不一致（已删除临时件 {tmp.name}，本地原文保留）: "
                      f"{tmp_sha[:12]}…/{tmp_size}B vs 原文 {src_sha[:12]}…/{src_size}B")

    if target.exists():
        # 注意：不能用 with_suffix —— "X.ja.srt" 会被改成 "X.ja.ja.srt.prev"（多插一层）
        prev = target.with_name(target.name + ".prev")
        try:
            prev.unlink(missing_ok=True)
            target.replace(prev)
            log(f"    原文内容已变化，旧版保留为 {prev.name}（旧 sha1={tgt_sha[:12]}…）")
        except OSError as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            return None, f"保留旧版归档件失败 {prev}: {type(e).__name__}: {str(e)[:120]}"
    try:
        tmp.replace(target)
    except OSError as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return None, f"归档件就位失败 {target}: {type(e).__name__}: {str(e)[:120]}"

    try:
        fin_sha, fin_size = sha1_of(target)
    except OSError as e:  # noqa: BLE001
        return None, f"就位后复读归档件失败 {target}: {type(e).__name__}: {str(e)[:120]}"
    if fin_sha != src_sha or fin_size != src_size:
        return None, (f"就位后复核不一致（本地原文保留，请人工核对归档件）: {target} "
                      f"{fin_sha[:12]}…/{fin_size}B vs 原文 {src_sha[:12]}…/{src_size}B")
    log(f"    原文归档完成并校验通过 {target}（sha1={src_sha} 体积={src_size} B）")
    return target, f"sha1={src_sha} 体积={src_size}B"


def do_upload(item, workdir, state, cloud, cid, orig_dir) -> None:
    """验收通过才清理本地：**归档成功 → 本地验收 → 上传 → 云端一致 → 才删本地**（H1/H2）。

    任一步失败都只记状态 + 日志，**绝不删本地文件**、也绝不"静默继续"。
    「上传完成」（接口返回成功）与「验收通过」（结构/时间轴/云端一致）是两件事，状态值分开记。
    """
    stem = item["stem"]
    video = find_on_disk(workdir, stem)
    if not video:
        state.set(stem, "skipped_missing_video")
        return
    srt = video.with_suffix(".srt")
    zh = srt.with_name(srt.stem + ".zh-cn.srt")

    if not (srt.exists() and zh.exists()):
        log(f"[{stem}] ❌ 验收失败：产物缺失（原文 {srt.exists()} / 中文 {zh.exists()}）（保留本地文件）")
        state.set(stem, "skipped_missing_translation",
                  orig_exists=srt.exists(), zh_exists=zh.exists())
        return
    if item.get('_grounded_audit_required'):
        from grounded_translation import require_audit
        require_audit(srt,zh,srt.parent/(srt.stem+'.alignment')/'source_review.json')

    # ① 原文归档（H1）：归档**成功且经 sha1+体积校验**是清理本地原文的前置条件
    archived, note = archive_original(srt, orig_dir)
    if archived is None:
        log(f"[{stem}] ❌ 归档失败：{note}（归档目录={orig_dir}）")
        log(f"[{stem}] ❌ 验收未通过 → 本地文件全部保留、不上传、不清理，状态 skipped_archive_failed"
            f"（原文 {srt}，视频 {video}）")
        state.set(stem, "skipped_archive_failed", reason=note,
                  archive_dir=str(orig_dir) if orig_dir else None, orig=str(srt))
        return
    log(f"[{stem}] 原文已归档并校验通过 {archived}（{note}）")

    # ② 本地验收（H2）：结构 + 时间轴 + 原/译一致 —— 不合格就不上传（别把坏字幕传到云端）
    ok, verdict = validate_subtitles(srt, zh, video)
    if not ok:
        log(f"[{stem}] ❌ 验收失败：{verdict}（保留本地文件）")
        state.set(stem, "skipped_verify_failed", reason=verdict, stage="local")
        return

    # ③ 中文回传。正常路径下这里不可能有同名文件：scan 只把「视频在、同名 srt 不在」
    #    的片入队，状态里已 uploaded 的片重跑时直接跳过。真遇到同名说明是人为强跑，
    #    此时不删不改、告警跳过，避免污染目录或覆盖人工校对过的字幕。
    name = f"{stem}.srt"
    cloud.invalidate(cid)
    existing = cloud.find_by_name(cid, name)
    if existing:
        fids = [it.get("fid") for it in existing]
        log(f"[{stem}] ⚠ 云端已存在同名 {name}（fid={fids}），已跳过不覆盖。"
            f"若确认要重做，先手工处理该文件再重跑。")
        state.set(stem, "skipped_name_exists", name=name, fids=fids)
        return

    if not cloud.upload(zh, cid, name):
        state.set(stem, "skipped_upload_failed")
        log(f"[{stem}] 中文回传失败，跳过（重跑会再试）")
        return
    log(f"[{stem}] 中文已回传（上传接口返回成功）{name}")

    # ④ 云端一致性（H2.3）：重列目录核对体积；能拿到 sha1 就一起核对
    try:
        local_sha, local_size = sha1_of(zh)
        cue_count = len(parse_srt_cues(zh))     # 先取（清理本地后就取不到了）
    except OSError as e:  # noqa: BLE001
        log(f"[{stem}] ❌ 验收失败：读不到本地中文产物 {zh}（{type(e).__name__}: {str(e)[:100]}）"
            f"（保留本地文件）")
        state.set(stem, "skipped_verify_failed", reason="local_read_failed", stage="cloud",
                  name=name, error=str(e)[:150])
        return
    cloud.invalidate(cid)
    verified, hits = cloud.verify_uploaded(cid, name, local_size)
    if len(hits) > 1:
        log(f"[{stem}] ⚠ 回传后目录里有 {len(hits)} 份 {name}，请人工清理")
    if not verified:
        sizes = [it.get("s") for it in hits]
        log(f"[{stem}] ❌ 验收失败：云端体积不一致（本地 {local_size} B，云端 {sizes}）（保留本地文件）")
        state.set(stem, "skipped_verify_failed", reason="cloud_size_mismatch",
                  name=name, local_bytes=local_size, cloud_sizes=sizes, stage="cloud")
        return
    cloud_sha = ""
    for it in hits:
        if int(it.get("s") or -1) == local_size:
            cloud_sha = str(it.get("sha") or it.get("sha1") or "").lower()
            break
    if cloud_sha and cloud_sha != local_sha:
        log(f"[{stem}] ❌ 验收失败：云端 sha1 不一致（本地 {local_sha}，云端 {cloud_sha}）（保留本地文件）")
        state.set(stem, "skipped_verify_failed", reason="cloud_sha1_mismatch",
                  name=name, local_sha1=local_sha, cloud_sha1=cloud_sha, stage="cloud")
        return
    # 云端没返回 sha1 → 明说「未做」，不能算通过；体积一致是当前能拿到的最强证据
    sha_note = (f"sha1 一致（{local_sha[:12]}…）" if cloud_sha
                else "sha1 核对：**未做**（云端未返回 sha 字段，仅比对体积）")
    log(f"[{stem}] ✅ 验收通过：{verdict} / 云端体积一致 {local_size} B / {sha_note}")

    # ⑤ 五大判据全过才清理本地（SOP §4.3「产物与命名」末条）
    failed: list[str] = []
    for f in (video, srt, zh,
              zh.with_name(zh.stem + ".global_context.json"),
              zh.with_name(zh.name + ".checkpoint"),
              srt.with_name(srt.name + ".checkpoint")):
        try:
            Path(f).unlink(missing_ok=True)
        except OSError as e:  # noqa: BLE001
            failed.append(f"{Path(f).name}({type(e).__name__})")
    state.set(stem, "uploaded", name=name, bytes=local_size, sha1=local_sha,
              cues=cue_count, cloud_sha1=cloud_sha or None,
              cloud_sha_checked=bool(cloud_sha), verify="passed")
    if failed:
        log(f"[{stem}] ⚠ 已回传且验收通过，但本地清理失败（不影响云端产物）: {failed}")
    done = sum(1 for r in state.items.values() if r.get("status") == "uploaded")
    log(f"[{stem}] 完成（验收通过、本地已清理）累计完成 {done} 部")


# ------------------------------------------------- 流水线编排
# 收尾规则（2026-09-22 实测踩坑四次，别再改）：
#   ① 漏传必挂：终止符条数不足 → 线程永久阻塞在 q.get()。
#   ② 多投必挂（更隐蔽）：曾让"每个退出的线程投 下游线程数 个 None"，3 个翻译线程
#      各投 2 个 = 6 个，把只有 2 个线程的上传池提前收掉 → 仍留在 upload_q 里的条目
#      **没有任何线程去取**，静默漏上传（实测 IPVR-226-8K 漏传）。
#   ③ **线程自己调用"关闭下游池"同样会踩坑**：3 个翻译线程各调一次
#      `_shutdown(upload_q, UPLOAD_WORKERS)` = 6 个 None，效果同 ②。
# 收尾规则（2026-09-22 实测踩坑四次后的最终方案：**不要再用"数 sentinel 条数"那套**）
#
# 踩过的坑：
#   ① 漏投 → 线程永久阻塞在 q.get()，主线程 join 一起挂死。
#   ② 多投 → 把下游池提前收掉，仍留在队列里的条目**没有任何线程去取**（静默漏上传，
#      实测 IPVR-226-8K 漏传）。数目对不对取决于线程调度顺序，**本质上不可靠**：
#      实测 1 个条目漏 1 个、3 个条目漏 1 个，同一天两次跑结果还不一样。
#   ③ 线程自己"关闭下游池"= 把 ② 放大 N 倍（3 个翻译线程各关一次上传池）。
#
# 最终方案：**计数式屏障**——用「已处理条目数 == 总条目数」来关池，与线程调度无关。
#   * 每级维护 processed / total；一个线程处理完一条就 +1；
#   * 恰好在一次 +1 后 processed == total 时，由**这一不变量**触发：向下游投
#     「下游线程数」个 None（此时下游队列里所有条目都已投递完毕，不会漏）；
#   * 下游线程数 >= 上游线程数 时这一投必然够用（多出来的是多余终止符，无害：
#     它们只会让已经空转的线程退出，且一定排在所有条目之后）。
class StageGate:
    """一个阶段的「处理完毕」计数屏障。"""

    def __init__(self, total: int, name: str):
        self.total = max(1, total)
        self.name = name
        self.done = 0
        self.lock = threading.Lock()
        self.closed = False

    def item_done(self) -> bool:
        """记录一条处理完成；返回 True 表示"恰好在这一条之后全部完成"。"""
        with self.lock:
            self.done += 1
            if self.done >= self.total and not self.closed:
                self.closed = True
                return True
            return False


class ForwardQueue:
    """Track whether a stage forwarded its one item, including failures."""
    def __init__(self, target):
        self.target = target
        self.forwarded = False

    def put(self, item):
        self.target.put(item)
        self.forwarded = True


def _extract_loop(extract_q, workdir, state, translate_q, gate, next_n) -> None:
    while True:
        item = extract_q.get()
        if item is None:
            break
        forwarded = ForwardQueue(translate_q)
        try:
            if item.get('_skip_pipeline'):
                forwarded.put(item)
            else:
                do_extract(item, workdir, state, forwarded)
        except Exception as e:  # noqa: BLE001
            log(f"[{item['stem']}] 提取异常 {type(e).__name__}: {str(e)[:200]}")
            state.set(item["stem"], "skipped_extract_failed", error=str(e)[:200])
        finally:
            if not forwarded.forwarded:
                translate_q.put(dict(item, _skip_pipeline=True))
            if gate.item_done():
                log(f"提取阶段全部完成（{gate.done}/{gate.total}），关闭翻译池")
                _put_nones(translate_q, next_n)


def _translate_loop(translate_q, workdir, state, upload_q, gate, next_n) -> None:
    while True:
        item = translate_q.get()
        if item is None:
            break
        try:
            if not item.get('_skip_pipeline'):
                do_translate(item, workdir, state, upload_q)
        except Exception as e:  # noqa: BLE001
            log(f"[{item['stem']}] 翻译异常 {type(e).__name__}: {str(e)[:200]}")
            state.set(item["stem"], "skipped_translate_failed", error=str(e)[:200])
        finally:
            if gate.item_done():
                log(f"翻译阶段全部完成（{gate.done}/{gate.total}），关闭上传池")
                _put_nones(upload_q, next_n)


def _put_nones(q: queue.Queue, n: int) -> None:
    for _ in range(n):
        q.put(None)


def run_pipeline(queue_items, workdir, cloud, state, cid, orig_dir) -> None:
    workdir.mkdir(parents=True, exist_ok=True)   # 必须先建：下载线程会查磁盘余量
    dl_q: queue.Queue = queue.Queue()
    extract_q: queue.Queue = queue.Queue()
    translate_q: queue.Queue = queue.Queue()
    upload_q: queue.Queue = queue.Queue()
    for it in queue_items:
        dl_q.put(it)

    stop_event = threading.Event()
    n_dl = min(DL_WORKERS, max(1, len(queue_items)))

    def throttle(extract_q) -> None:
        """在途囤货上限：已下载未提取 >= MAX_QUEUED_DL 时暂停取新任务。
        轮询间隔必须短（0.5s）——实测用 10s 会让「最后几个条目」白等 10-20 秒，
        因为睡下去之后不会再检查队列是否已经空了。"""
        while extract_q.qsize() >= MAX_QUEUED_DL and not stop_event.is_set():
            if free_gb(workdir) < 30:
                log(f"磁盘余量 {free_gb(workdir):.1f} GB，暂停取新下载任务")
                stop_event.wait(30)
            else:
                stop_event.wait(0.5)

    def download_loop():
        while not stop_event.is_set():
            throttle(extract_q)              # 取新任务前先看囤货水位
            if extract_q.qsize() >= MAX_QUEUED_DL:
                continue
            try:
                item = dl_q.get_nowait()
            except queue.Empty:
                break
            try:
                forwarded = ForwardQueue(extract_q)
                do_download(item, workdir, cloud, state, forwarded, stop_event)
            except Exception as e:  # noqa: BLE001
                log(f"[{item['stem']}] 下载异常 {type(e).__name__}: {str(e)[:200]}")
                state.set(item["stem"], "skipped_download", error=str(e)[:200])
            finally:
                if not forwarded.forwarded:
                    extract_q.put(dict(item, _skip_pipeline=True))

    def upload_loop():
        while True:
            item = upload_q.get()
            if item is None:
                break
            try:
                do_upload(item, workdir, state, cloud, cid, orig_dir)
            except Exception as e:  # noqa: BLE001
                log(f"[{item['stem']}] 上传异常 {type(e).__name__}: {str(e)[:200]}")
                state.set(item["stem"], "skipped_upload_failed", error=str(e)[:200])

    gates = {"extract": StageGate(len(queue_items), "extract"),
             "translate": StageGate(len(queue_items), "translate")}

    dls = [threading.Thread(target=download_loop, name="download", daemon=True) for _ in range(n_dl)]
    exts = [threading.Thread(target=_extract_loop,
                             args=(extract_q, workdir, state, translate_q,
                                   gates["extract"], TRANSLATE_WORKERS),
                             name="extract", daemon=True) for _ in range(EXTRACT_WORKERS)]
    trs = [threading.Thread(target=_translate_loop,
                            args=(translate_q, workdir, state, upload_q,
                                  gates["translate"], UPLOAD_WORKERS),
                            name="translate", daemon=True) for _ in range(TRANSLATE_WORKERS)]
    ups = [threading.Thread(target=upload_loop, name="upload", daemon=True)
           for _ in range(UPLOAD_WORKERS)]
    threads = dls + exts + trs + ups
    for t in threads:
        t.start()
    log(f"线程已启动：下载 {n_dl} / 提取 {EXTRACT_WORKERS} / 翻译 {TRANSLATE_WORKERS} / "
        f"上传 {UPLOAD_WORKERS}（共 {len(threads)} 个）")
    log(f"收尾机制：计数式屏障（每级处理满 {len(queue_items)} 条后按其下游线程数关池）")

    def scheduler():
        """只负责「下载池排空 → 关闭提取池」这一处；
        其余各级的关闭由 StageGate 在"处理满 N 条"时精确触发（见文件顶部注释）。"""
        try:
            for t in dls:
                t.join()
            if stop_event.is_set():
                log("已停止：等待在途任务收尾")
                _put_nones(extract_q, EXTRACT_WORKERS)
                _put_nones(translate_q, TRANSLATE_WORKERS)
                _put_nones(upload_q, UPLOAD_WORKERS)
                return
            log("下载池已排空，关闭提取池")
            _put_nones(extract_q, EXTRACT_WORKERS)
        except Exception as e:  # noqa: BLE001
            # scheduler 自己出错也必须放行下游，否则整条链静默挂死
            log(f"⚠ scheduler 异常 {type(e).__name__}: {e}；补 sentinel 放行下游")
            _put_nones(extract_q, EXTRACT_WORKERS)
            _put_nones(translate_q, TRANSLATE_WORKERS)
            _put_nones(upload_q, UPLOAD_WORKERS)

    threading.Thread(target=scheduler, name="scheduler", daemon=True).start()

    # 收尾看门狗（轮询式，不用阻塞 join——否则会对已退出的线程误报「未退出」）。
    # 正常情况全部工作线程会自然退出，看门狗直接结束；
    # 只有某个线程真卡住（超过 SLACK 秒仍活着）才补足终止符，仍不退则 dump 栈 + 落盘 + 强制退出。
    # 目的：绝不允许「活干完了、进程挂着不退出」。
    def reaper():
        slack = REAPER_SLACK
        groups = (("download", dls, extract_q, EXTRACT_WORKERS),
                  ("extract", exts, translate_q, TRANSLATE_WORKERS),
                  ("translate", trs, upload_q, UPLOAD_WORKERS),
                  ("upload", ups, None, 0))
        start = {name: time.monotonic() for name, *_ in groups}
        warned: set[str] = set()
        while True:
            if all(not t.is_alive() for t in threads):
                return
            # Active batches and off-peak waits are not shutdown deadlocks.
            # Arm the shutdown watchdog only after both work gates are closed.
            if not all(g.closed for g in gates.values()):
                start = {name: time.monotonic() for name, *_ in groups}
                time.sleep(5)
                continue
            for name, group, next_q, next_n in groups:
                if not any(t.is_alive() for t in group):
                    continue
                if next_q is None or time.monotonic() - start[name] < slack[name]:
                    continue
                if name not in warned:
                    warned.add(name)
                    log(f"⚠ {name} 线程 {slack[name]}s 未退出，补 {next_n} 个 sentinel 放行下游")
                    for _ in range(next_n):
                        next_q.put(None)
            if time.monotonic() - min(start.values()) > 20000:
                break
            time.sleep(5)
        alive = [t.name for t in threads if t.is_alive()]
        log(f"⚠ 工作线程超时未退出: {alive}；dump 线程栈后强制结束")
        frames = sys._current_frames()
        for t in threads:
            if not t.is_alive() or t.ident not in frames:
                continue
            stack = "".join(traceback.format_stack(frames[t.ident])[-5:])
            log(f"--- {t.name} ---\n{stack}")
        state.save()
        log_flush()
        os._exit(0)

    threading.Thread(target=reaper, name="reaper", daemon=True).start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        log("收到 Ctrl-C：停止取新任务，等在途任务收尾（再按一次直接退出）…")
        stop_event.set()
        for _ in range(EXTRACT_WORKERS):
            extract_q.put(None)
        for _ in range(TRANSLATE_WORKERS):
            translate_q.put(None)
        for _ in range(UPLOAD_WORKERS):
            upload_q.put(None)
        for t in threads:
            t.join(timeout=300)


# ------------------------------------------------- 命令
def resolve_cid(cloud: Cloud, args) -> str | None:
    if args.remote_cid:
        return str(args.remote_cid)
    if args.remote_path:
        return cloud.resolve_path(args.remote_path)
    return None


def cmd_scan(args) -> int:
    cookie = cloud_cookie()
    if not cookie:
        print(missing_cookie_message(), file=sys.stderr)
        return 2
    cloud = Cloud(cookie)
    cid = resolve_cid(cloud, args)
    if cid is None:
        print("请给 --remote-cid 或 --remote-path", file=sys.stderr)
        return 2
    if args.list_only:
        for it in cloud.list_dir(cid, ttl=0):
            kind = "DIR " if it.get("fid") is None else "FILE"
            print(f"{kind} {it['n']}\tsize={it.get('s')}\tpc={it.get('pc')}")
        return 0

    s = survey(cloud, cid)
    total = sum(int(v.get("s") or 0) for v in s["missing"])
    biggest = max((int(v.get("s") or 0) for v in s["missing"]), default=0)
    workdir = WORK_ROOT / safe_name(args.name or f"cid{cid}")
    print(f"目录 cid={cid}  条目 {len(s['all'])}"
          f"（视频 {len(s['videos'])} / 字幕 {len(s['subs'])} / 子目录 {len(s['dirs'])}）")
    print(f"已有字幕配对 {len(s['paired'])} 部；缺字幕 {len(s['missing'])} 部，合计 {human(total)}")
    print(f"单部最大 {human(biggest)}；D 盘可用 {free_gb(WORK_ROOT.parent):.1f} GB"
          f"（需 >= 最大文件 x 1.5 = {human(biggest * 1.5)}）")
    print(f"工作目录 {workdir}")
    if s["orphans"]:
        print(f"\n[注意] 孤立字幕 {len(s['orphans'])} 份（没有同名视频，多为改名遗留，需人工核对）：")
        for o in s["orphans"]:
            print(f"   - {o['n']}")
    print("\n缺字幕清单（小到大，流水线按此顺序跑）：")
    for v in sorted(s["missing"], key=lambda x: int(x.get("s") or 0)):
        print(f"   {v['n']}\t{human(int(v.get('s') or 0))}")
    print(f"\n正式跑：mvsub run --remote-cid {cid}")
    return 0


def cmd_run(args) -> int:
    global PAID_ARGS
    from translation_budget import paid_cli_args
    try:PAID_ARGS=paid_cli_args(args.budget_file,args.budget_cny)
    except (OSError,ValueError,RuntimeError) as error:
        print(str(error),file=sys.stderr);return 2
    env = load_env()
    if not env.get("DEEPSEEK_API_KEY"):
        print(f"缺少 DEEPSEEK_API_KEY：写在 {config.repo_root() / '.env'}，或先设成环境变量", file=sys.stderr)
        return 2
    if not PROMPT.exists():
        print(f"提示词不存在: {PROMPT}", file=sys.stderr)
        return 2

    cookie = cloud_cookie(env)
    if not cookie:
        print(missing_cookie_message(), file=sys.stderr)
        return 2
    cloud = Cloud(cookie)
    cid = resolve_cid(cloud, args)
    if cid is None:
        print("请给 --remote-cid 或 --remote-path", file=sys.stderr)
        return 2

    s = survey(cloud, cid)
    workdir = WORK_ROOT / safe_name(args.name or f"cid{cid}")
    workdir.mkdir(parents=True, exist_ok=True)
    for stale in workdir.glob("*.part"):
        log(f"清理上次残留 {stale.name}")
        stale.unlink(missing_ok=True)

    state = State(workdir / "state.json")
    global log_fh
    log_fh = (workdir / "run.log").open("a", encoding="utf-8")

    todos = sorted(s["missing"], key=lambda x: int(x.get("s") or 0))
    if args.only:
        todos = [v for v in todos if any(k.lower() in v["n"].lower() for k in args.only)]
    if args.limit:
        todos = todos[: args.limit]

    items = []
    for v in todos:
        stem = stem_of(v["n"])
        if state.get(stem).get("status") == "uploaded":
            continue
        items.append({"stem": stem, "n": v["n"], "pc": v["pc"], "s": v.get("s")})

    if not items:
        log(f"无可跑条目（云端缺字幕 {len(s['missing'])} 部，均已完成或已被 --only/--limit 过滤）")
        return 0

    orig_dir = Path(args.orig_dir) if args.orig_dir else ORIG_ARCHIVE_DIR
    try:
        orig_dir.mkdir(parents=True, exist_ok=True)
        if not orig_dir.is_dir() or not os.access(str(orig_dir), os.W_OK):
            raise OSError(f"目录不可写或不是目录（{orig_dir}）")
        log(f"原文归档目录（本地，不上云）: {orig_dir}")
    except OSError as e:  # noqa: BLE001
        # H1：归档成功是清理本地原文的前置条件。没有可用归档目录时**拒绝开跑**（rc=2），
        # 而不是降级成"原文只留工作目录"——那会在删本地时丢掉原文。
        # 真正的可写性由 archive_original() 的复制+校验兜底（这里只做提前拦截）。
        log(f"❌ 原文归档目录不可用（{orig_dir}）: {e}")
        log("❌ 拒绝开跑：归档未确认就不能删本地原文（H1）。请用 --orig-dir 指定可写目录，"
            f"或修复默认目录 {config.archive_dir()} 后重跑。")
        print(f"原文归档目录不可用: {orig_dir}: {e}", file=sys.stderr)
        return 2

    log(f"===== 开跑：云端缺字幕 {len(s['missing'])} 部，本次处理 {len(items)} 部 =====")
    log(f"并发：下载 {min(DL_WORKERS, len(items))} / 提取 {EXTRACT_WORKERS} / "
        f"翻译 {TRANSLATE_WORKERS} / 上传 {UPLOAD_WORKERS}")
    log(f"GPU 提取模型 {MODEL_ASR}；翻译 {MODEL_TRANSLATE}；提示词 {PROMPT.name}")
    log(f"工作目录 {workdir}")
    # 运行时指纹：明确「本次跑的是哪一版代码、原文往哪写」。
    # 2026-09-22 教训：改完代码后重启进程走的是新逻辑，而另一处（子代理/人工）读到的可能是旧文件，
    # 于是日志里的行为与文件内容看起来"自相矛盾"，白查一轮。把指纹打出来就不会再误判。
    try:
        _mt = time.strftime("%m-%d %H:%M:%S", time.localtime(Path(__file__).stat().st_mtime))
    except OSError:
        _mt = "未知"
    log(f"本流水线源码版本: nosub_pipeline.py mtime={_mt}；"
        f"原文归档目录={'（未启用）' if orig_dir is None else orig_dir}")
    log(f"原文策略: 只写本地归档，**不上传云端**（云端只新增中文字幕）")

    run_pipeline(items, workdir, cloud, state, cid, orig_dir)

    # 收官：重扫云端核对
    cloud.invalidate(cid)
    after = survey(cloud, cid)
    done = sum(1 for r in state.items.values() if r.get("status") == "uploaded")
    log(f"===== 本轮结束：累计完成 {done} 部；云端仍缺字幕 {len(after['missing'])} 部 =====")
    # 「本轮结果」汇总（2026-09-24 决定 2）：单片跳过是设计行为，但"整批大面积失败"必须一眼看出来，
    # 否则 skipped_* 混在 30 行日志里会看起来像成功。这一行同时进 run.log（log() 双写）。
    counts, extra, stages = batch_status_counts(state, [it["stem"] for it in items])
    log(format_batch_summary(counts, extra, stages))
    if counts["uploaded"] == 0 and items:
        log(f"⚠ 本轮 {len(items)} 部**没有任何一部通过验收**（uploaded 0）——别把这轮当成功；"
            f"先看上面的 ❌ 验收失败 / 跳过统计，再决定重跑")
    log("重跑同一命令可继续未完成项。")
    return 0


def batch_status_counts(state, stems) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """按状态统计**本轮条目**（决定 2 的「本轮结果」汇总用）。

    返回 (counts, extra, stages)：
      counts —— `BATCH_STATUS_ORDER` 里每个状态的出现次数（固定顺序，缺的记 0）；
      extra  —— `unfinished`（还停在 downloaded/extracted/translated，即没走到最后）与 `other`（其余状态）；
      stages —— `skipped_verify_failed` 的 stage 拆解（local / cloud）。
    只读 state，不改任何东西；纯本地。
    """
    counts = {k: 0 for k in BATCH_STATUS_ORDER}
    stages: dict[str, int] = {}
    unfinished = other = 0
    for stem in stems:
        st = (state.get(stem) or {}).get("status", "") or "未开始"
        if st in counts:
            counts[st] += 1
            if st == "skipped_verify_failed":
                sg = str((state.get(stem) or {}).get("stage") or "未知")
                stages[sg] = stages.get(sg, 0) + 1
        elif st in ("downloaded", "extracted", "translated"):
            unfinished += 1
        else:
            other += 1
    return counts, {"unfinished": unfinished, "other": other}, stages


def format_batch_summary(counts: dict[str, int], extra: dict[str, int],
                         stages: dict[str, int]) -> str:
    """把 `batch_status_counts()` 的结果拼成**一行**「本轮结果」（`cmd_run` 收尾打印，同时进 run.log）。"""
    tokens = []
    for k in BATCH_STATUS_ORDER:
        n = counts.get(k, 0)
        if k == "skipped_verify_failed" and n:
            detail = "、".join(f"stage={s} {stages[s]}" for s in ("local", "cloud") if stages.get(s))
            tokens.append(f"{k} {n}（{detail}）" if detail else f"{k} {n}")
        else:
            tokens.append(f"{k} {n}")
    if extra.get("unfinished"):
        tokens.append(f"未完成 {extra['unfinished']}")
    if extra.get("other"):
        tokens.append(f"其他 {extra['other']}")
    return "本轮结果：" + " / ".join(tokens)


def cmd_status(args) -> int:
    cookie = cloud_cookie()
    if not cookie:
        print(missing_cookie_message(), file=sys.stderr)
        return 2
    cloud = Cloud(cookie)
    cid = resolve_cid(cloud, args)
    if cid is None:
        print("请给 --remote-cid 或 --remote-path", file=sys.stderr)
        return 2
    workdir = WORK_ROOT / safe_name(args.name or f"cid{cid}")
    state = State(workdir / "state.json")
    s = survey(cloud, cid)
    counts: dict[str, int] = {}
    for rec in state.items.values():
        counts[rec.get("status", "?")] = counts.get(rec.get("status", "?"), 0) + 1
    print(f"目录 cid={cid}")
    print(f"云端：视频 {len(s['videos'])}，已有字幕 {len(s['paired'])}，缺字幕 {len(s['missing'])}")
    print(f"本地：{workdir / 'state.json'}")
    for k in sorted(counts):
        print(f"   {k:28s} {counts[k]}")
    left = [v["n"] for v in s["missing"]
            if state.get(stem_of(v["n"])).get("status") != "uploaded"]
    print(f"\n还差 {len(left)} 部：")
    for n in sorted(left):
        print(f"   {n}\t{state.get(stem_of(n)).get('status', '未开始')}")
    parts = [f.name for f in workdir.glob("*.part")]
    if parts:
        print(f"\n下载中：{parts}")
    local_videos = [f.name for f in workdir.iterdir()
                    if f.is_file() and f.suffix.lower() in VIDEO_EXT]
    if local_videos:
        print(f"本地已下载待处理：{local_videos}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="无字幕影片加字幕流水线（下载 ∥ 提取 ∥ 翻译 ∥ 回传）")
    sub = p.add_subparsers(dest="cmd", required=True)
    common = [("--remote-cid", dict(default=None, help="115 目标目录 cid")),
              ("--remote-path", dict(default=None, help="115 目标目录路径，如 /twice/破解vr（与 cid 二选一）")),
              ("--name", dict(default=None, help="工作目录名（默认 cid<数字>）"))]
    for name, helptext in (("scan", "只读预检：列出缺字幕的视频"),
                           ("run", "正式跑（断点续跑）"),
                           ("status", "查看进度")):
        sp = sub.add_parser(name, help=helptext)
        for flag, kw in common:
            sp.add_argument(flag, **kw)
        if name == "scan":
            sp.add_argument("--list-only", action="store_true", help="只列目录原始条目")
        if name == "run":
            sp.add_argument('--budget-file',type=Path);sp.add_argument('--budget-cny')
            sp.add_argument("--only", action="append", default=[], help="只跑名字含该关键词的（可重复）")
            sp.add_argument("--limit", type=int, default=0, help="最多跑 N 部（试跑验收用）")
            sp.add_argument("--orig-dir", default=None, help="日文原文本地归档目录（默认 config.archive_dir()，不上云端）")
    vp = sub.add_parser("verify", help="只读复核一对字幕（本地、不联网、不写状态）：SOP §3.1 步骤⑥ 的可自动化判据")
    vp.add_argument("video", help="源视频路径；只跑字幕判据（不做片长范围校验）就写 -")
    vp.add_argument("orig", help="原文 *.ja.srt（时间轴基准）")
    vp.add_argument("zh", help="中文字幕 *.srt（被复核的那份）")
    return p


def cmd_verify(args) -> int:
    """`verify`：纯本地只读复核（不联网、不写状态、不碰云端）。

    退出码：0 = 硬判据全过；1 = 有硬判据失败；2 = 参数/文件缺失（用法错误）。
    """
    video = None if str(args.video).strip() in ("-", "none", "None") else Path(args.video)
    orig, zh = Path(args.orig), Path(args.zh)
    for p, label in ((orig, "原文"), (zh, "中文")):
        if not p.exists():
            print(f"❌ 找不到{label}字幕: {p}", file=sys.stderr)
            print("用法: nosub_pipeline.py verify <video|-> <orig.ja.srt> <zh.srt>", file=sys.stderr)
            return 2
    if video is not None and not video.exists():
        print(f"❌ 找不到视频: {video}（只跑字幕判据请把 video 写成 -）", file=sys.stderr)
        return 2

    result = verify_subtitles(video, orig, zh)
    print(f"[verify] 视频={video if video else '（未提供：片长范围校验记为未做）'}")
    print(f"[verify] 原文={orig}\n[verify] 中文={zh}")
    if result["ok"]:
        print(f"✅ 验收通过：{result['summary']}")
    else:
        print(f"❌ 验收失败：{'; '.join(result['failures'])}（本地文件未改动）")
    st = result["stats"]
    if st:
        print(f"[stat] cue {len(parse_srt_cues(zh))} / 可判 CPS {st.get('cps_judged', 0)} / "
              f"超短 cue {st.get('short_cues', 0)} / 假名残留 {st.get('kana_lines', 0)} / "
              f"单字长重复 {st.get('single_char_runs', 0)} / 连续重复 {st.get('consec_dups', 0)} / "
              f"与原文相同 {st.get('identical_to_orig', 0)}")
    for line in result["details"]:
        print(line)
    for line in result["hints"]:
        print(f"⚠ 提示：{line}")
    print("[verify] 云端一致性不适用（本命令纯粹本地只读；云端体积/sha1 核对在 `run` 里做）")
    return 0 if result["ok"] else 1


def main() -> int:
    args = build_parser().parse_args()
    return {"scan": cmd_scan, "run": cmd_run, "status": cmd_status, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
