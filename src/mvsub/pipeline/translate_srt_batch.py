#!/usr/bin/env python3
"""
批量翻译 SRT 字幕（DeepSeek API / Ollama）。
改进点：
- Translation Memory：自动缓存原文→译文，重复句子直接命中
- 并发请求：按完成顺序补位，输出保留原字幕顺序
- 自适应 batch size：默认较大 batch，失败/残留多时自动拆小
- 滑动上下文：每批携带前 N 行原文，可独立并发
- 全局预处理：先通读全文生成剧情摘要、人物关系、术语表，注入每批翻译
- 一致性审校：整份译文回传模型，自动统一专有名词
- 断点续传：checkpoint 文件，中断后从断点继续
- 多层解析 fallback：索引号 → 顺序补齐 → 原文兜底
"""
import argparse
import concurrent.futures
import hashlib
import json
import http.client
import os
import re
import sys
import tempfile
import time
import ssl
import urllib.error
import urllib.request
from pathlib import Path

import zhconv

import config

from translation_budget import TranslationBudget,BudgetExceeded

API_BUDGET = None
API_MAX_TOKENS = 8192
API_REASONING_EFFORT = None
REFERENCE_TEXT = ''


def read_budgeted_response(response, deadline):
    """DeepSeek sends whitespace keepalives; socket timeout alone is insufficient."""
    chunks = []
    while True:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise RuntimeError('API wall-clock deadline exceeded; reservation retained; no automatic retry')
        if isinstance(response,http.client.HTTPResponse):
            # CPython urllib HTTPS uses HTTPResponse -> BufferedReader -> SSLSocket.
            # Shrink its blocking read timeout, rather than renewing the original
            # timeout for every keepalive near the absolute deadline.
            sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
            if sock is not None:sock.settimeout(max(.001,remaining))
        chunk = response.read1(65536)
        if not chunk:
            return b''.join(chunks)
        chunks.append(chunk)


HIRAGANA_KATAKANA_RE = re.compile(r"[぀-ゟ゠-ヿ]")

# Load the latest system prompt from file so it can be versioned without editing code.
_DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system_prompt_v7.txt"
if _DEFAULT_SYSTEM_PROMPT_PATH.exists():
    DEFAULT_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
else:
    DEFAULT_SYSTEM_PROMPT = """你是一名专业的成人影片日译中字幕翻译。

任务要求：
1. 逐行翻译日文AV字幕为本土化、口语化、符合情境的中文。
2. 必须结合上下文判断：动作主体、因果关系、动词的具体身体含义，避免逻辑谬误、主体混淆、动作失配。
3. 遇到省略主语的台词，根据权力关系、视线焦点和画面动作补全主语。
4. 遇到多义动词（如「出す」「入れる」「いく」等），按当下身体动作选择最准确的中文动词。
5. 特定词汇请参考“全局术语表”，人物称呼请参考“全局人物关系”。

输出纪律（必须严格遵守）：
- 每行只输出一条中文译文，保留原序号。
- 如果一条原文包含多个独立短句（由空格、逗号或语气词分隔），必须逐句完整翻译，禁止遗漏任何子句。
- 禁止输出任何括号注释、译者注、替代译法、解释说明、“此处...”“可依据...”等元信息。
- 禁止在译文末尾或开头添加原条目序号；如果原文本身含序号（如“100. 餃子”），保留该序号并翻译后面的内容。
- 禁止输出日文假名或原文。
- 遇到不确定的地方，按最符合AV情境的自然中文直接翻译，不要加注释或保留原词。
"""

# 模型常添加的注释/元信息关键词，用于解析时兜底清洗
NOTE_KEYWORDS = [
    "此处", "上下文", "保持原样", "可依据", "可能为", "场景描述",
    "调整", "原词", "英文", "日语", "罗马音", "音译", "直译", "意译",
    "译注", "注：", "译者", "翻译", "根据情境", "请参考",
]


def parse_srt(path: Path):
    """解析 SRT，返回 (index, start, end, text) 列表。"""
    content = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    entries = []
    for block in re.split(r"\n\s*\n", content.strip()):
        lines = [line.strip() for line in block.split("\n") if line.strip() != ""]
        if len(lines) < 3:
            continue
        idx = lines[0]
        time_line = lines[1]
        text = " ".join(lines[2:])
        m = re.match(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", time_line)
        if not m:
            continue
        entries.append((idx, m.group(1), m.group(2), text))
    return entries


def write_srt(path: Path, entries):
    """写入 SRT。"""
    lines = []
    for idx, start, end, text in entries:
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def format_reference_context(entries, reference_entries):
    """Reference may include surrounding cues, but must contain this exact source."""
    by_id = {entry[0]: entry for entry in reference_entries}
    if not reference_entries or len(by_id) != len(reference_entries):
        raise ValueError('Reference subtitles are empty or contain duplicate IDs')
    if any(by_id.get(entry[0]) != entry for entry in entries):
        raise ValueError('Reference must contain identical source IDs, text and timestamps')
    return '\n'.join(f'{entry[0]}. {entry[3]}' for entry in reference_entries)


class TranslationMemory:
    """简单的 JSON 文件缓存，键为 (source_text, target_lang, model)。"""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        self.hits = 0
        self.misses = 0
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def _key(self, text: str, target: str, model: str) -> str:
        return f"{model}::{target}::{text}"

    def get(self, text: str, target: str, model: str) -> str | None:
        key = self._key(text, target, model)
        if key in self.data:
            self.hits += 1
            return self.data[key]
        # 回退到人翻字幕缓存（不区分模型）
        human_key = self._key(text, target, "human_sub")
        if human_key in self.data:
            self.hits += 1
            return self.data[human_key]
        self.misses += 1
        return None

    def set(self, text: str, target: str, model: str, translation: str):
        key = self._key(text, target, model)
        self.data[key] = translation

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 限制单文件大小，超过 50k 条目时做 LRU 式裁剪
        if len(self.data) > 50000:
            self.data = dict(list(self.data.items())[-50000:])
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def stats(self) -> str:
        total = self.hits + self.misses
        if total == 0:
            return "0/0"
        return f"{self.hits}/{total} ({self.hits*100//total}%)"


def looks_untranslated(text: str, source: str) -> bool:
    """简单启发式：如果文本里还有日文假名，大概率没翻译。"""
    if source == "ja" or source.startswith("ja"):
        return bool(HIRAGANA_KATAKANA_RE.search(text))
    return False


# ---------------- 固定人名对照表（prompts/name_map.txt） ----------------
# 为什么需要（2026-09-24）：
#   1) 人名原先完全靠模型每次通读原文**自己生成**"全局人物关系"，同一人物在不同片里写法不一，
#      认不出来就**原样留日文**（实测有整句只因人名带假名被判漏译）。
#   2) 语音识别会**听错人名**：`いぶき` 被听成 `ひぶき`（同一语料里 16 处正确 vs 2 处误听）。
# 本表是**固定**的、优先于模型自生成的人物关系；翻译后再按本表做一次"残留日文名归一"，
# 命中即替换成中文，避免"整句因为一个名字带假名"被判漏译。
NAME_MAP_PATH = Path(__file__).resolve().parent / "prompts" / "name_map.txt"
NAME_MAP_CACHE: dict | None = None


def load_name_map(path: Path | None = None) -> dict:
    """读人名对照表 → {"replace": {日文: 中文}, "calls": [(后缀, 中文)], "block": prompt 片段}。

    表格式：`日文<TAB>中文<TAB>类别<TAB>说明`，`#` 开头为注释。表不存在时返回空表（不报错，
    保证旧环境仍能跑）。
    """
    global NAME_MAP_CACHE
    p = path or NAME_MAP_PATH
    if NAME_MAP_CACHE is not None and path is None:
        return NAME_MAP_CACHE
    replace: dict[str, str] = {}
    calls: list[tuple[str, str]] = []
    lines_for_prompt: list[str] = []
    if p.exists():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 2:
                continue
            ja, zh = cols[0].strip(), cols[1].strip()
            kind = (cols[2].strip() if len(cols) > 2 else "")
            note = (cols[3].strip() if len(cols) > 3 else "")
            if not ja or not zh:
                continue
            if kind == "call":
                if ja.startswith("〜"):          # 后缀规则：〜ちゃん → 〜酱（两侧都要去掉占位符）
                    calls.append((ja[1:], zh.lstrip("〜~").split("/")[0].strip()))
                    lines_for_prompt.append(f"- 称呼 {ja} 统一译成“{zh}”（按语境择优）")
                else:
                    replace[ja] = zh
                    lines_for_prompt.append(f"- {ja} -> {zh}")
                continue
            replace[ja] = zh
            tag = {"actress": "演员名", "fix": "语音识别误听纠正", "role": "片中角色"}.get(kind, "")
            lines_for_prompt.append(f"- {ja} -> {zh}" + (f"（{tag}{'；' + note if note else ''}）" if tag else ""))
    block = ""
    if lines_for_prompt:
        block = ("\n=== 固定人名对照表（**优先于**下面模型生成的人物关系；命中必须照此译） ===\n"
                 + "\n".join(lines_for_prompt))
    result = {"replace": replace, "calls": calls, "block": block, "count": len(replace)}
    if path is None:
        NAME_MAP_CACHE = result
    return result


def normalize_names(text: str) -> str:
    """把译文里残留的日文名按对照表换成中文（含"名字+称呼后缀"的组合）。

    只做**表内命中**的替换：表里没有的日文一律不动 —— 这样它既是修复手段，
    也不会擅自改动原文里本就该保留的东西；没被表覆盖的假名仍会被验收闸门拦下。
    """
    nm = load_name_map()
    if not nm["replace"]:
        return text
    out = text
    for ja, zh in sorted(nm["replace"].items(), key=lambda kv: -len(kv[0])):
        if ja and ja in out:
            out = out.replace(ja, zh)
    for suffix, zh_suffix in nm["calls"]:
        if not suffix:
            continue
        # 只处理紧随"表内中文名"的后缀，避免误伤普通词（例如"大家的"里的さん）
        for zh_name in sorted(set(nm["replace"].values()), key=len, reverse=True):
            out = out.replace(f"{zh_name}{suffix}", f"{zh_name}{zh_suffix}")
    return out


def format_global_context(global_ctx: dict | None) -> str:
    """把全局上下文对象格式化为 prompt 文本。"""
    if not global_ctx:
        return ""
    parts = ["\n=== 全局剧情与术语参考（仅供理解上下文，不要直接输出） ==="]
    if global_ctx.get("summary"):
        parts.append(f"剧情概要：{global_ctx['summary']}")
    characters = global_ctx.get("characters", [])
    if characters:
        parts.append("人物关系：")
        for ch in characters:
            name_ja = ch.get("name_ja", "")
            name_zh = ch.get("name_zh", "")
            role = ch.get("role", "")
            traits = ch.get("traits", "")
            notes = ch.get("notes", "")
            detail = traits or notes
            parts.append(f"- {name_ja} -> {name_zh}（{role}）{detail}")
    scenes = global_ctx.get("scenes", [])
    if scenes:
        parts.append("场景结构：")
        for sc in scenes:
            time_range = sc.get("time_range", "")
            setting = sc.get("setting", "")
            key_events = sc.get("key_events", "")
            parts.append(f"- {time_range} | {setting} | {key_events}")
    # 术语表：提示词要求模型输出 "terms"；兼容历史缓存里出现过的 "glossary"/"verbs" 键。
    # 2026-09-22 修复：原先只读 "verbs"/"glossary"，而提示词写的是 "terms"，
    # 导致模型生成的术语表**被静默丢弃**（全局上下文白跑一趟，术语一致性无从保证）。
    terms = global_ctx.get("terms") or global_ctx.get("glossary") or global_ctx.get("verbs") or []
    if terms:
        parts.append("术语表：")
        for item in terms:
            term_ja = item.get("term_ja", "")
            term_zh = item.get("term_zh", "")
            notes = item.get("notes", "")
            parts.append(f"- {term_ja} -> {term_zh} {notes}")
    if global_ctx.get("translation_notes"):
        parts.append(f"翻译注意事项：{global_ctx['translation_notes']}")
    return "\n".join(parts)


def build_prompt(batch: list[tuple], context_lines: list[str], source: str, target: str,
                 system_prompt: str | None = None, global_ctx: dict | None = None) -> str:
    """构造带上下文的 prompt。"""
    target_name = "Chinese" if target == "zh-cn" else target

    lines = []
    for idx, _, _, text in batch:
        lines.append(f"{idx}. {text}")
    input_text = "\n".join(lines)

    if system_prompt:
        parts = [system_prompt.strip()]
    else:
        parts = [DEFAULT_SYSTEM_PROMPT.strip()]

    # 注入固定人名对照表（**必须在模型自生成的"全局上下文"之前**，表里写明它优先）
    name_block = load_name_map()["block"] if source.startswith('ja') else ''
    if name_block:
        parts.append(name_block)

    # 注入全局上下文
    global_text = format_global_context(global_ctx)
    if global_text:
        parts.append(global_text)

    # 始终追加格式约束，防止自定义提示词漏掉输出要求
    parts.append("输出纪律：")
    parts.append("- 每行只输出一条中文译文，保留原序号。")
    parts.append("- 如果一条原文包含多个独立短句（由空格或语气词分隔），必须全部翻译，禁止遗漏后半部分。")
    parts.append("- 严禁添加括号注释、译者注、替代译法、解释说明或任何元信息。")
    parts.append("- 行首编号是条目 ID，必须原样保留，包括不连续编号；不得重新编号、合并或遗漏条目。")
    parts.append("- 禁止输出日文假名或原文。")
    if context_lines:
        parts.append("\n前文原文（只供理解，不要输出）：\n" + "\n".join(context_lines))
    if REFERENCE_TEXT:
        parts.append('\n相邻原文参考（只供理解，不要输出；最终待译范围在后面）：\n'
                     + REFERENCE_TEXT)
    parts.append(f"\n待翻译字幕：\n{input_text}\n\n中文译文：")
    return "\n".join(parts)


def _load_env_file() -> None:
    """把 .env 里的 key 注入环境变量（零依赖，不引入 python-dotenv）。

    为什么需要：批处理流水线（nosub_pipeline）自己会读 .env 并传给子进程，
    但**人工直接跑本脚本时没人注入**，会拿不到 DEEPSEEK_API_KEY 而报
    `HTTP Error 401: Authorization Required`。这里补上，保证两种用法都成立。
    只补环境里还没有的变量，不覆盖已有的 shell 环境变量。
    """
    try:
        for k, v in config.load_dotenv().items():
            if k and v and not os.environ.get(k):
                os.environ[k] = v
    except OSError:
        pass


def check_account_balance(budget,api_key):
    req=urllib.request.Request('https://api.deepseek.com/user/balance',
        headers={'Authorization':'Bearer '+api_key},method='GET')
    try:
        with urllib.request.urlopen(req,timeout=20) as response:data=json.loads(response.read().decode('utf-8'))
        if data.get('is_available') is not True:raise RuntimeError('DeepSeek balance unavailable')
        balance=next(row['total_balance'] for row in data['balance_infos'] if row['currency']=='CNY')
        budget.observe_balance(balance)
    except Exception:
        if not budget.summary().get('halt_reason'):
            budget.halt('Account balance could not be verified or reconciled; no new paid requests')
        raise


def call_ollama(
    prompt: str,
    model: str,
    url: str,
    timeout: int = 300,
    response_format: str | None = None,
    response_path: Path | None = None,
    thinking: bool | None = None,
) -> str:
    """Call LLM: Ollama /api/generate, or OpenAI /chat/completions for DeepSeek etc."""
    if thinking is None:
        thinking = os.environ.get("DEEPSEEK_THINKING", "1") != "0"
    url = url.rstrip("/")
    if "deepseek" in url or "openai" in url:
        if url!='https://api.deepseek.com' or model!='deepseek-flash':
            raise ValueError('Budgeted API supports only official deepseek-flash; no credential sent to another endpoint')
        if API_BUDGET is None:
            raise RuntimeError('Paid translation requires a shared budget; configure --budget-cny and --budget-file')
        if API_BUDGET.summary().get('halt_reason'):raise BudgetExceeded(API_BUDGET.summary()['halt_reason'])
        _load_env_file()
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                f"缺少 DEEPSEEK_API_KEY：写在 {config.repo_root() / '.env'}，或先设成环境变量 "
                "（见 docs/configuration.md）")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "top_p": 0.8,
            "stream": False,
        }
        if 'deepseek' in url:
            # Short subtitle batches do not need hidden reasoning; it dominated 94% of output cost.
            payload['thinking']={'type':'enabled' if thinking else 'disabled'}
            if API_REASONING_EFFORT is not None and thinking:
                payload['reasoning_effort'] = API_REASONING_EFFORT
            if API_BUDGET is not None:
                payload['max_tokens'] = API_MAX_TOKENS
        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        for attempt in range(1):
            if 'deepseek' in url and os.environ.get('DEEPSEEK_OFF_PEAK_ONLY')=='1':
                from deepseek_offpeak import wait_offpeak
                wait_offpeak(timeout)
            check_account_balance(API_BUDGET,api_key)
            reservation = None
            if API_BUDGET is not None:
                if url != 'https://api.deepseek.com':
                    raise ValueError('Budgeted requests require the official DeepSeek endpoint')
                reservation = API_BUDGET.reserve(prompt, API_MAX_TOKENS, model)
            request_started=time.monotonic()
            try:
                with urllib.request.urlopen(req,timeout=timeout) as resp:
                    raw = (read_budgeted_response(resp, request_started + timeout)
                           if reservation is not None else resp.read())
                    result=json.loads(raw.decode('utf-8'))
                if (not isinstance(result,dict) or not isinstance(result.get('choices'),list)
                        or not result['choices'] or not isinstance(result['choices'][0],dict)
                        or not isinstance(result['choices'][0].get('message'),dict)
                        or not isinstance(result.get('usage'),dict)):
                    raise ValueError('Malformed API response; paid reservation retained')
                value=result['choices'][0]['message'].get('content')
                if value is not None and not isinstance(value,str):raise ValueError('Invalid API content type')
                if response_path is not None:
                    choice = (result.get('choices') or [{}])[0]
                    snapshot = dict(prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                        model=model, reasoning_effort=API_REASONING_EFFORT,
                        max_output_tokens=API_MAX_TOKENS, response_format=response_format,
                        content=(choice.get('message') or {}).get('content') or '',
                        finish_reason=choice.get('finish_reason'), usage=result.get('usage') or {})
                    snapshot.update(provider_request_id=result.get('id'),provider_model=result.get('model'),
                                    provider_created_at=result.get('created'))
                    snapshot['content_sha256'] = hashlib.sha256(snapshot['content'].encode('utf-8')).hexdigest()
                    response_path = Path(response_path)
                    response_path.parent.mkdir(parents=True, exist_ok=True)
                    from audio_archive import atomic_json
                    atomic_json(response_path, snapshot)
                if reservation is not None:
                    API_BUDGET.settle(reservation, result.get('usage') or {},receipt=dict(
                        provider_request_id=result.get('id'),provider_model=result.get('model'),provider_created_at=result.get('created')))
                try:check_account_balance(API_BUDGET,api_key)
                except Exception:
                    print('[budget] Account cross-check stopped future requests; received response retained',flush=True)
                break
            except (http.client.HTTPException,urllib.error.URLError,TimeoutError,ConnectionError,ssl.SSLError) as error:
                API_BUDGET.halt(f'{type(error).__name__}: paid result unknown or request rejected; no automatic resend')
                raise
            except (RuntimeError, ValueError, OSError):
                API_BUDGET.halt('Paid response could not be completed/verified; inspect saved response before resuming')
                raise
        choice = result["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        usage = result.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        # Counters only: never print prompts, credentials or reasoning text.
        print('[api] '+json.dumps(dict(seconds=round(time.monotonic()-request_started,2),
            attempts=attempt+1,
            thinking=payload.get('thinking',{}).get('type'),
            finish_reason=choice.get('finish_reason'),prompt_tokens=usage.get('prompt_tokens'),
            completion_tokens=usage.get('completion_tokens'),reasoning_tokens=details.get('reasoning_tokens'),
            content_chars=len(content)),ensure_ascii=True),flush=True)
        if not content:
            API_BUDGET.halt('Empty paid response; inspect before new requests')
            raise ValueError(f'Empty API reply (finish_reason={choice.get("finish_reason")})')
        if choice.get('finish_reason') == 'length':
            API_BUDGET.halt('Output limit reached; inspect cached response and reduce target before resuming')
            raise RuntimeError('API output limit reached; stop and reduce batch size before retrying')
        if choice.get('finish_reason')!='stop':
            API_BUDGET.halt('Paid response did not finish normally; inspect saved receipt')
            raise RuntimeError('Incomplete API response; not accepted as a translation')
        return content

    # Ollama 本地
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.8},
    }
    if response_format == "json":
        payload["format"] = "json"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("response", "").strip()


def generate_global_context(
    entries: list[tuple],
    model: str,
    url: str,
    source: str,
    target: str,
    timeout: int = 300,
    cache_path: Path | None = None,
) -> dict | None:
    """
    先把全部原文通读一遍，让模型生成剧情摘要、人物关系、术语表。
    如果提供了 cache_path 且存在有效缓存，直接复用。
    对长片做智能采样，保证在合理时间内完成全局分析。
    """
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("entries_hash") == entries_mtime(entries) and cached.get("model") == model:
                print(f"[global] 从缓存加载全局上下文: {cache_path}")
                return cached.get("context", {})
        except (json.JSONDecodeError, OSError):
            pass

    target_name = "Chinese" if target == "zh-cn" else target
    sampled = _sample_entries_for_global_context(entries)
    print(f"[global] 原文 {len(entries)} 条，采样后 {len(sampled)} 条用于全局分析")

    # 只给模型看序号和文本，节省时间轴
    full_text = "\n".join(f"{idx}. {text}" for idx, _, _, text in sampled)

    prompt = (
        f"你是一名专业的 {source}-to-{target_name} 影视字幕分析师。"
        "请先通读以下日文字幕原文（已按时间线采样，覆盖全片），"
        "然后输出一份严格 JSON 格式的全局上下文。"
        "这份上下文将直接用于指导后续逐批翻译，所以必须具体、准确、贴合本片实际内容，不能泛泛而谈。\n\n"
        "要求输出严格 JSON，不要 markdown 代码块，不要解释。JSON 结构如下：\n"
        "{\n"
        '  "summary": "2-3句话概括本片剧情主线：人物关系、核心互动、场景转换、情绪走向。要具体，不要套话。",\n'
        '  "scenes": [\n'
        '    {"time_range": "大致时间范围如 00:00-05:00", "setting": "场景地点和氛围", "key_events": "该段落发生的关键互动"}\n'
        "  ],\n"
        '  "characters": [\n'
        '    {"name_ja": "日文称呼或名字", "name_zh": "建议中文称呼", "role": "身份/与主角关系", "traits": "语气、性格、如何称呼对方"}\n'
        "  ],\n"
        '  "terms": [\n'
        '    {"term_ja": "原文明确出现的词或称呼", "term_zh": "建议中文译法", "notes": "出现位置或明确语义"}\n'
        "  ],\n"
        '  "translation_notes": "翻译统一事项：常见省略主语的补全规则、语气风格、重复出现的特殊表达处理方式等"\n'
        "}\n\n"
        "注意：\n"
        "- 只输出 JSON，不要 markdown 代码块，不要解释。\n"
        "- 必须基于本片实际内容分析，不要输出通用描述。\n"
        "- 人物：注意区分是同一人不同称呼，还是不同角色。\n"
        "- 只记录原文能直接支持的事实；不推断身体动作、人物关系或未说出的情节。\n"
        "- 场景按时间线分段，帮助后续理解剧情推进。\n"
        "- 原文不清的内容不要编入术语表。\n\n"
        f"采样后的日文字幕原文（共 {len(sampled)} 条，覆盖全片）：\n{full_text}\n\n"
        "JSON:"
    )

    print("[global] 生成全局上下文（剧情摘要 / 人物关系 / 术语表）...")
    raw = call_ollama(prompt, model, url, timeout=timeout, response_format="json")

    # 尝试解析 JSON
    ctx = None
    try:
        # 先直接解析
        ctx = json.loads(raw)
    except json.JSONDecodeError:
        # 再尝试提取 {} 包裹的内容
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                ctx = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if ctx is None:
        print("[warn] 全局上下文解析失败，将不使用全局上下文继续翻译", file=sys.stderr)
        print(f"[debug] 模型返回:\n{raw[:500]}", file=sys.stderr)
        return None

    print(f"[global] 摘要: {ctx.get('summary', '')[:80]}...")
    _term_count = len(ctx.get("terms") or ctx.get("glossary") or ctx.get("verbs") or [])
    print(f"[global] 场景: {len(ctx.get('scenes', []))} 段，人物: {len(ctx.get('characters', []))} 个，术语: {_term_count} 条")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"entry_count": len(entries), "entries_hash": entries_mtime(entries), "model": model, "context": ctx}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[global] 已缓存: {cache_path}")

    return ctx


def _sample_entries_for_global_context(entries: list[tuple], max_entries: int = 800) -> list[tuple]:
    """
    为全局上下文生成做智能采样。
    保留开头和结尾，中间按比例抽取，确保覆盖全片且不超过模型高效处理长度。
    """
    total = len(entries)
    if total <= max_entries:
        return entries

    # 保留开头 25% 和结尾 25%，中间 50% 按比例采样
    head = max_entries // 4
    tail = max_entries // 4
    middle_budget = max_entries - head - tail
    middle_start = head
    middle_end = total - tail
    middle_count = middle_end - middle_start

    if middle_count <= middle_budget:
        middle_sampled = entries[middle_start:middle_end]
    else:
        positions = [round(i * (middle_count - 1) / (middle_budget - 1)) for i in range(middle_budget)]
        middle_sampled = [entries[middle_start + pos] for pos in positions]
    sampled = entries[:middle_start] + middle_sampled + entries[middle_end:]
    return sampled


def translate_whole_file(
    entries: list[tuple],
    model: str,
    url: str,
    source: str,
    target: str,
    timeout: int = 1800,
    system_prompt: str | None = None,
    global_ctx: dict | None = None,
) -> list[tuple]:
    """
    全文整体翻译：一次性把全部字幕发给模型，要求按序号返回全部译文。
    适合短字幕的整体输出试验；长片优先分批，减少失败后的重复付费。
    """
    total = len(entries)
    print(f"[whole] 开始全文整体翻译，共 {total} 条...")

    target_name = "Chinese" if target == "zh-cn" else target
    full_text = "\n".join(f"{idx}. {text}" for idx, _, _, text in entries)

    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    else:
        parts.append(DEFAULT_SYSTEM_PROMPT.strip())

    global_text = format_global_context(global_ctx)
    if global_text:
        parts.append(global_text)

    parts.extend([
        "输出纪律：",
        "- 逐条翻译以下全部日文字幕为自然、口语化、符合情境的中文。",
        "- 每行只输出一条中文译文，格式必须是：序号. 中文译文",
        "- 必须保留输入中的每个原始序号；序号可不连续，不得重新编号、合并或遗漏。",
        "- 严禁输出日文原文、括号注释、译者注、解释说明。",
        "- 结合全文剧情、人物关系、场景转换，保持语气、人称、术语一致。",
        f"\n待翻译字幕（共 {total} 条）：\n{full_text}\n\n中文译文：",
    ])
    prompt = "\n".join(parts)

    raw = call_ollama(prompt, model, url, timeout=timeout)

    texts = parse_batch_response(raw, entries)
    print(f"[whole] 严格按原始 ID 匹配 {total}/{total} 条")
    return [(idx, start, end, text) for (idx, start, end, _), text in zip(entries, texts)]


def parse_batch_response(raw: str, batch: list[tuple]) -> list[str]:
    """按原始 ID 严格匹配；缺行时拒绝顺序填补，避免污染时间轴。"""
    translations = {}
    expected = [str(entry[0]) for entry in batch]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate source subtitle IDs")
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.strip('"').strip("'").strip()
        if not line:
            continue
        m = re.match(r"^(\d+)[\.。、:：\s]\s*(.*)$", line)
        if m:
            idx, text = m.group(1), m.group(2).strip()
            if idx not in expected or idx in translations or not text:
                raise ValueError(f"Unexpected, duplicate or empty subtitle ID: {idx}")
            translations[idx] = text
        else:
            raise ValueError("Unnumbered translation response")

    missing = set(expected) - translations.keys()
    if missing:
        raise ValueError(f"Missing subtitle IDs: {sorted(missing)}")

    results = []
    for idx, _, _, text in batch:
        key = str(idx)
        cleaned = sanitize_translation(translations[key], text)
        if not cleaned:
            raise ValueError(f"Empty translation after cleaning: {idx}")
        # 人名归一：模型偶尔把认不出的日文专名原样留下，
        # 这里按固定人名对照表换掉；表外的一律不动（留给验收闸门拦）。
        cleaned = normalize_names(cleaned)
        results.append(cleaned)
    return results


def sanitize_translation(text: str, original: str) -> str:
    """清洗模型返回：去掉注释、 stray index 等，同时保留原文自带的序号。"""
    if not text:
        return text

    def remove_note(m: re.Match) -> str:
        inner = m.group(1)
        if any(k in inner for k in NOTE_KEYWORDS):
            return ""
        return m.group(0)

    # 去掉括号注释（含注释关键词的）
    text = re.sub(r"（([^）]+)）", remove_note, text)
    text = re.sub(r"\(([^)]+)\)", remove_note, text)

    # 如果原文没有前导序号，去掉模型自己加的序号
    if not re.match(r"^\d+\.", original or ""):
        text = re.sub(r"^\d+[\.。、]\s*", "", text)

    # 如果原文没有末尾序号，去掉模型自己加的序号
    if not re.search(r"\d+\s*$", original or ""):
        text = re.sub(r"\s+\d+\s*$", "", text)

    return text.strip()


def translate_one(text: str, tm: TranslationMemory | None, model: str, url: str, source: str, target: str) -> str:
    """单条翻译。"""
    source_hint = f" from {source}" if source and source != "auto" else ""
    target_name = "Chinese" if target == "zh-cn" else target
    prompt = (
        f"You are a professional Japanese-to-{target_name} subtitle translator for adult films. "
        "Translate the following Japanese subtitle into natural, fluent Mainland Chinese. "
        "Preserve tone, scene intent, and erotic/adult context. "
        "If the text looks like a moan or gasp, translate it as a Chinese moan/gasp, not as a name or real word. "
        "Do not add explanations or notes. Return only the translation.\n\n"
        f"Japanese: {text}\n\n"
        "Chinese:"
    )
    raw = call_ollama(prompt, model, url, timeout=120)
    if looks_untranslated(raw, source) or raw == text:
        prompt = (
            f"This text is in {source}. Translate it into Mainland {target_name} now. "
            "Only output the translation, no original text, no explanations:\n\n"
            f"{text}\n\n"
            "Translation:"
        )
        raw = call_ollama(prompt, model, url, timeout=120)
    result = sanitize_translation(raw.strip() or text, text)
    if tm and not looks_untranslated(result, source):
        tm.set(text, target, model, result)
    return result


def translate_batch(batch: list[tuple], context_lines: list[str], tm: TranslationMemory,
                    model: str, url: str, source: str, target: str, timeout: int = 300,
                    system_prompt: str | None = None, global_ctx: dict | None = None) -> tuple[list[str], int]:
    """
    翻译一批，返回 (译文列表, cache_hit_count)。
    先查 TM，未命中的再请求模型；请求后写回 TM。
    """
    # 逐句缓存不包含场景和人物，不能用于影片对白或写回新的对应关系。
    results = [None] * len(batch)
    pending = []
    for i, (idx, _, _, text) in enumerate(batch):
        pending.append((i, batch[i]))

    if not pending:
        return results, len(batch)

    # 只有未命中的才构造 prompt
    pending_batch = [item[1] for item in pending]
    prompt = build_prompt(pending_batch, context_lines, source, target, system_prompt=system_prompt, global_ctx=global_ctx)
    raw = call_ollama(prompt, model, url, timeout=timeout)
    texts = parse_batch_response(raw, pending_batch)

    for (i, entry), text in zip(pending, texts):
        results[i] = text

    return results, len(batch) - len(pending)


def translate_batch_with_splits(batch, context_lines, tm, model, url, source, target,
                                timeout=300, system_prompt=None, global_ctx=None, context_window=100):
    """Recover malformed output by at most two split levels; keep strict IDs."""
    def run(chunk, context, depth):
        try:
            return translate_batch(chunk, context, tm, model, url, source, target,
                timeout=timeout, system_prompt=system_prompt, global_ctx=global_ctx)
        except ValueError:
            if depth >= 2 or len(chunk) <= 32:
                raise
            middle = len(chunk)//2
            left, right = chunk[:middle], chunk[middle:]
            print(f'[warn] batch format failed; split {len(chunk)} into {len(left)}+{len(right)} (level {depth+1})',flush=True)
            first, hits_a = run(left, context, depth+1)
            before = context + [f'{entry[0]}. {entry[3]}' for entry in left]
            next_context = before[-context_window:] if context_window > 0 else []
            second, hits_b = run(right, next_context, depth+1)
            return first+second, hits_a+hits_b
    return run(batch, context_lines, 0)


def adaptive_batch_translate(entries: list[tuple], tm: TranslationMemory, model: str, url: str,
                             source: str, target: str, initial_batch_size: int, context_window: int,
                             max_workers: int, checkpoint_path: Path, resume: bool,
                             timeout: int = 300, system_prompt: str | None = None,
                             global_ctx: dict | None = None) -> list[tuple]:
    """
    自适应 batch + 并发翻译。
    维护一个状态列表 translated[idx]，完成一批就写 checkpoint。
    """
    if max_workers<1 or initial_batch_size<1:raise ValueError('Workers and batch size must be positive')
    total = len(entries)
    translated = [None] * total
    start_idx = 0
    thinking_mode='enabled' if 'deepseek' in url else None
    reference_digest = (hashlib.sha256((f'{source}/{target}/{context_window}\n' +
        build_prompt([], [], source, target, system_prompt, global_ctx)).encode('utf-8')).hexdigest()
        if REFERENCE_TEXT else None)

    # 尝试加载 checkpoint
    if resume and checkpoint_path.exists():
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if cp.get("alignment_version") in (2,3) and cp.get("input_mtime") == entries_mtime(entries) and cp.get("model") == model and cp.get("target") == target and cp.get('thinking_mode')==thinking_mode and cp.get('reasoning_effort')==API_REASONING_EFFORT and cp.get('reference_context_digest')==reference_digest:
                loaded = cp.get("entries", [])
                if len(loaded) == total:
                    start_idx = cp.get("next_index", 0)
                    translated = [(e[0], e[1], e[2], t) if (i < start_idx or cp.get('alignment_version')==3) and isinstance(t,str) and t else None
                                  for i, (e, t) in enumerate(zip(entries, loaded))]
                    start_idx=next((i for i,t in enumerate(translated) if t is None),total)
                    print(f"[resume] 从 checkpoint 恢复，已从 {start_idx}/{total} 条继续")
        except (json.JSONDecodeError, OSError):
            pass

    def save_checkpoint(next_index: int):
        cp = {
            "alignment_version": 3,
            "input_mtime": entries_mtime(entries),
            "model": model,
            "target": target,
            "thinking_mode": thinking_mode,
            "reasoning_effort": API_REASONING_EFFORT,
            "reference_context_digest": reference_digest,
            "next_index": next_index,
            "entries": [t[3] if t else "" for t in translated],
        }
        with tempfile.NamedTemporaryFile(dir=checkpoint_path.parent,suffix='.part',delete=False) as stream:
            temporary=Path(stream.name);stream.write(json.dumps(cp,ensure_ascii=False,indent=2).encode('utf-8'))
        try:temporary.replace(checkpoint_path)
        finally:temporary.unlink(missing_ok=True)

    # 工作函数：取下一个 batch 并翻译
    def worker_task(idx: int, batch_size: int):
        batch = entries[idx:idx + batch_size]
        # 仅给前文原文；不传播旧中文，也不把后文拼到待译批次前。
        before = entries[max(0, idx - context_window):idx]
        context = [f"{e[0]}. {e[3]}" for e in before]
        texts, _ = translate_batch_with_splits(batch, context, tm, model, url, source, target,
                                   timeout=timeout, system_prompt=system_prompt, global_ctx=global_ctx,
                                   context_window=context_window)
        if len(texts)!=len(batch):raise ValueError('Translation batch entry count differs')

        # 残留重试
        retry = 0
        while retry < 2 and any(looks_untranslated(t, source) for t in texts):
            retry += 1
            bad_count = sum(1 for t in texts if looks_untranslated(t, source))
            print(f"[warn] batch@{idx} 有 {bad_count} 条疑似未翻译，拆单条重试 #{retry}")
            new_texts = []
            for entry, text in zip(batch, texts):
                if looks_untranslated(text, source):
                    repair_prompt = build_prompt([entry], context, source, target,
                                                 system_prompt=system_prompt, global_ctx=global_ctx)
                    repaired = call_ollama(repair_prompt, model, url, timeout=timeout)
                    new_texts.append(parse_batch_response(repaired, [entry])[0])
                else:
                    new_texts.append(text)
            texts = new_texts

        return idx, [(e[0], e[1], e[2], t) for e, t in zip(batch, texts)]

    # 按顺序调度 batch，但允许并发执行
    pending = {}
    i = start_idx
    batch_size = initial_batch_size
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while i < total or pending:
            # 提交新任务直到并发数满
            while i < total and len(pending) < max_workers:
                if translated[i] is not None:i+=1;continue
                end=min(total,i+batch_size)
                # A resumed out-of-order batch must not be sent again.
                for j in range(i+1,end):
                    if translated[j] is not None:end=j;break
                future = executor.submit(worker_task, i, end-i)
                pending[future]=i
                i=end

            if not pending:
                break

            # HTTP requests have their own timeout; discount waits may take hours.
            # Refill whichever slot finishes first, retaining source order in output.
            gated='deepseek' in url and os.environ.get('DEEPSEEK_OFF_PEAK_ONLY')=='1'
            done,_=concurrent.futures.wait(pending,timeout=None if gated else timeout+60,
                return_when=concurrent.futures.FIRST_COMPLETED)
            if not done:
                for future in pending:future.cancel()
                raise RuntimeError('Translation batches timed out; final output not written')
            failures=[]
            for future in sorted(done,key=lambda f:pending[f]):
                idx=pending.pop(future)
                try:
                    _,completed_entries=future.result()
                    for offset,entry in enumerate(completed_entries):translated[idx+offset]=entry
                except Exception as error:failures.append((idx,error))
            committed=next((j for j,t in enumerate(translated) if t is None),total)
            if any(t is not None for t in translated):save_checkpoint(committed)
            print(f"[progress] {sum(t is not None for t in translated)}/{total}",flush=True)
            if failures:
                idx,error=failures[0]
                for future in pending:future.cancel()
                # Already-sent requests cannot be cancelled. Keep their successful
                # responses while the executor drains, rather than paying twice.
                for future in concurrent.futures.as_completed([f for f in pending if not f.cancelled()]):
                    try:
                        _,entries_done=future.result()
                        start=pending[future]
                        for offset,entry in enumerate(entries_done):translated[start+offset]=entry
                        committed=next((j for j,t in enumerate(translated) if t is None),total)
                        save_checkpoint(committed)
                    except Exception:pass
                raise RuntimeError(f'Translation batch {idx} failed; final output not written') from error

    return translated


def entries_mtime(entries: list[tuple]) -> float:
    """用 entries 内容 hash 作为 input_mtime，简单稳定。"""
    import hashlib
    text = "\n".join(f"{e[0]}|{e[1]}|{e[2]}|{e[3]}" for e in entries)
    return float(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16))


def source_consistency(entries: list[tuple], source_entries: list[tuple]) -> tuple[list[tuple], int]:
    """
    基于原文的一致性修正：同一日文句子出现多次但被译成不同中文时，统一为出现次数最多的译法。
    这是比 LLM 审校更保守、更可靠的兜底机制。
    """
    from collections import Counter
    src_to_translations = {}
    for (_, _, _, src), (_, _, _, dst) in zip(source_entries, entries):
        src_to_translations.setdefault(src, []).append(dst)

    chosen = {}
    fixes = 0
    for src, dsts in src_to_translations.items():
        if len(dsts) > 1 and len(set(dsts)) > 1:
            most_common = Counter(dsts).most_common(1)[0][0]
            chosen[src] = most_common

    new_entries = []
    for (_, _, _, src), (idx, start, end, dst) in zip(source_entries, entries):
        if src in chosen and dst != chosen[src]:
            new_entries.append((idx, start, end, chosen[src]))
            fixes += 1
        else:
            new_entries.append((idx, start, end, dst))
    return new_entries, fixes


def llm_consistency_review(entries: list[tuple], source_entries: list[tuple], model: str, url: str, target: str) -> list[tuple]:
    """
    LLM 一致性审校（可选）：把日中文对照回传模型，让它只修正"同一日语词出现不同中文译法"的情况。
    """
    print("[review] 开始 LLM 一致性审校...")
    lines = []
    for (_, _, _, src), (_, _, _, dst) in zip(source_entries, entries):
        lines.append(f"{src} || {dst}")
    review_input = "\n".join(lines)

    target_name = "Chinese" if target == "zh-cn" else target
    prompt = (
        f"You are reviewing {target_name} translations of Japanese subtitles.\n"
        f"Task: Find cases where the SAME Japanese word (especially proper nouns, names, or shop names) "
        f"appears multiple times but is translated into DIFFERENT {target_name} words. "
        f"Choose the most natural translation and apply it to all occurrences.\n"
        f"IMPORTANT:\n"
        f"- Only fix inconsistencies where the SAME Japanese word has DIFFERENT translations.\n"
        f"- Do NOT change the meaning of any line.\n"
        f"- Do NOT introduce new proper nouns that do not appear in the original Japanese.\n"
        f"- If a Japanese word appears only once, do not change it.\n"
        f"Return ONLY corrections in this exact format (one per line):\n"
        f"JAPANESE_WORD || OLD_CHINESE || CORRECTED_CHINESE\n"
        f"If no corrections are needed, return nothing.\n\n"
        f"Japanese || Chinese:\n{review_input}\n\n"
        f"Corrections:"
    )

    raw = call_ollama(prompt, model, url, timeout=300)
    corrections = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("||")]
        if len(parts) >= 3:
            src_word, old_word, corrected = parts[0], parts[1], parts[2]
            if src_word and old_word and corrected and src_word != corrected:
                corrections.append((src_word, old_word, corrected))

    if not corrections:
        print("[review] 无需修正")
        return entries

    print(f"[review] 发现 {len(corrections)} 处不一致")
    new_entries = []
    applied = 0
    for (_, _, _, src), (idx, start, end, dst) in zip(source_entries, entries):
        new_dst = dst
        for src_word, old_word, corrected in corrections:
            if src_word in src and old_word in new_dst:
                new_dst = new_dst.replace(old_word, corrected)
                applied += 1
        new_entries.append((idx, start, end, new_dst))

    print(f"[review] 应用 {applied} 处替换")
    return new_entries


def main():
    global API_BUDGET, API_MAX_TOKENS, API_REASONING_EFFORT, REFERENCE_TEXT
    parser = argparse.ArgumentParser(description="批量翻译 SRT 字幕")
    parser.add_argument("srt", help="输入 SRT 文件路径")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama 模型名（默认 qwen2.5:14b）")
    parser.add_argument("--url", default="http://localhost:11434", help="Ollama API 地址")
    parser.add_argument("--source", default="ja", help="源语言代码")
    parser.add_argument("--target", default="zh-cn", help="目标语言代码")
    parser.add_argument("--batch-size", type=int, default=128, help="初始每批翻译条数（默认 128，14b 可承受）")
    parser.add_argument("--min-batch-size", type=int, default=4, help="最小每批条数")
    parser.add_argument("--context-lines", type=int, default=100, help="每批携带的前序原文条数，不含后文（默认 100）")
    parser.add_argument("--reference-srt", help="仅供理解的较大原文范围；须包含输入字幕的完全相同条目，按batch-size只输出本批")
    parser.add_argument('--grounded',action='store_true',help='忠实JSON协议、局部参考、持久响应缓存和疑点待核')
    parser.add_argument('--source-review',type=Path,help='绑定当前原文/音轨的词面复核报告')
    parser.add_argument("--max-workers", type=int, default=1, help="每文件并发请求数（DeepSeek批量入口8；本地14b建议1）")
    parser.add_argument("--timeout", type=int, default=600, help="单次 Ollama 请求超时秒数（默认 600）")
    parser.add_argument("--system-prompt", default=None, help="自定义 system prompt 文件路径（覆盖默认提示词）")
    parser.add_argument("--output", default=None, help="输出 SRT 路径（默认覆盖原文件同目录 .zh-cn.srt）")
    parser.add_argument("--tm-path", default=None, help="Translation Memory 路径（默认项目根目录 translation_memory.json）")
    parser.add_argument("--no-resume", action="store_true", help="不加载 checkpoint 从头开始")
    parser.add_argument("--no-review", action="store_true", help="跳过基于原文的一致性修正")
    parser.add_argument("--llm-review", action="store_true", help="额外启用 LLM 一致性审校（较慢，可选）")
    parser.add_argument("--keep-checkpoint", action="store_true", help="完成后保留 checkpoint 文件")
    parser.add_argument("--global-context", action="store_true", help="显式启用基于原文的全局上下文")
    parser.add_argument("--no-global-context", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--global-context-cache", default=None, help="全局上下文缓存路径（默认输出文件同目录 .global_context.json）")
    parser.add_argument("--whole-file", action="store_true", help="短字幕整体输出试验；长片建议分批以保留断点和降低失败重付费用")
    parser.add_argument("--off-peak-only", action="store_true", help="DeepSeek 请求仅在官方优惠时段发送")
    parser.add_argument("--budget-cny", type=str, help="本轮共享预算上限（人民币，含重试）")
    parser.add_argument("--budget-file", help="共享费用台账路径；续跑须使用相同文件和上限")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="预算模式每次输出上限，含思考（默认8192）")
    parser.add_argument("--reasoning-effort", choices=['low', 'high', 'max'], help="DeepSeek 思考强度，保持思考开启；low仅供对照，不作为默认质量档")
    args = parser.parse_args()
    REFERENCE_TEXT = ''
    if args.reference_srt and args.whole_file:
        parser.error('--reference-srt is for batch output, not --whole-file')
    remote = 'deepseek' in args.url or 'openai' in args.url
    if remote and not args.budget_cny:
        parser.error('Paid requests require --budget-cny and --budget-file')
    if args.grounded and (args.whole_file or args.global_context or args.llm_review):
        parser.error('Grounded mode uses local references and review flags, without a whole-file output/summary/review pass')
    API_REASONING_EFFORT = args.reasoning_effort or ('high' if args.grounded else None)
    if bool(args.budget_cny) != bool(args.budget_file):
        parser.error('--budget-cny and --budget-file must be specified together')
    if args.budget_cny:
        if args.model != 'deepseek-flash' or args.url.rstrip('/') != 'https://api.deepseek.com':
            parser.error('Budget mode supports only official deepseek-flash')
        if not 1 <= args.max_output_tokens <= 32768:
            parser.error('--max-output-tokens must be between 1 and 32768')
        API_MAX_TOKENS = args.max_output_tokens
        API_BUDGET = TranslationBudget(args.budget_file, args.budget_cny)
        print('[budget] '+json.dumps(API_BUDGET.summary()), flush=True)
    if args.off_peak_only:
        os.environ["DEEPSEEK_OFF_PEAK_ONLY"] = "1"

    input_path = Path(args.srt).resolve()
    if not input_path.exists():
        print(f"[error] 文件不存在: {input_path}", file=sys.stderr)
        return 1

    system_prompt = None
    if args.system_prompt:
        sp_path = Path(args.system_prompt).resolve()
        if not sp_path.exists():
            print(f"[error] 提示词文件不存在: {sp_path}", file=sys.stderr)
            return 1
        system_prompt = sp_path.read_text(encoding="utf-8").strip()
        print(f"[info] 使用自定义提示词: {sp_path}")

    entries = parse_srt(input_path)
    if args.reference_srt:
        REFERENCE_TEXT = format_reference_context(entries, parse_srt(Path(args.reference_srt).resolve()))
        print('[info] 已载入独立原文参考；只输出待译批次，参考内容不输出', flush=True)
    if args.grounded:
        reference_info='独立参考文件' if args.reference_srt else '参考块48条、两侧各8条'
        print(f'[info] 共 {len(entries)} 条，模型 {args.model}，待译最多{min(args.batch_size,16)}条，{reference_info}，并发最多{min(args.max_workers,2)}')
    else:
        print(f"[info] 共 {len(entries)} 条字幕，模型 {args.model}，batch {args.batch_size}，前文 {args.context_lines}，并发 {args.max_workers}")

    # TM 路径，默认放到仓库根目录（translation_memory.json），跨文件/跨剧集复用
    tm_path = Path(args.tm_path) if args.tm_path else config.work_dir() / "translation_memory.json"
    tm = None if args.grounded else TranslationMemory(tm_path)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        suffix = f".{args.target}" if args.target else ".translated"
        output_path = input_path.with_suffix(f"{suffix}.srt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(output_path.suffix + ".checkpoint")

    if args.grounded:
        if not remote or API_BUDGET is None:parser.error('Grounded mode requires a budgeted DeepSeek client')
        from grounded_translation import translate,write_certificate,SourceContextHold
        from source_review import load_report,reference_source
        review = load_report(args.source_review,input_path) if args.source_review else None
        protected=[input_path]+([reference_source(review)] if review else [])
        def check_context_hold():
            if any((path.parent/'translation_content_hold.json').exists() for path in protected):
                raise SourceContextHold('Source context hold remains active; generation and publication withheld')
        check_context_hold()
        if system_prompt is None:
            system_prompt=(Path(__file__).parent/'prompts/system_prompt_v9_grounded.txt').read_text(encoding='utf-8')
        reference = parse_srt(Path(args.reference_srt)) if args.reference_srt else None
        if reference is not None:
            format_reference_context(entries,reference)
            if review:format_reference_context(reference,parse_srt(reference_source(review)))
        names={key:value for key,value in load_name_map()['replace'].items()
               if any(key in e[3] for e in entries)}
        # The existing API adapter owns budget, discount gating and durable capture.
        def ask(prompt,path):
            check_context_hold()
            return call_ollama(prompt,args.model,args.url,timeout=args.timeout,response_format='json',response_path=path)
        result=translate(entries,output_path,system_prompt,ask,
            dict(model=args.model,reasoning_effort=API_REASONING_EFFORT,max_output_tokens=API_MAX_TOKENS),
            batch_size=min(args.batch_size,16),max_workers=args.max_workers,
            blocked=list(dict.fromkeys(review['hold_ids']+review.get('reference_hold_ids',[]))) if review else (),
            reference_entries=reference,names=names,failure_callback=API_BUDGET.halt)
        result['source_review_available']=review is not None
        result['source_file_sha256']=hashlib.sha256(input_path.read_bytes()).hexdigest()
        result['reference_file']=str(Path(args.reference_srt).resolve()) if args.reference_srt else None
        result['reference_file_sha256']=hashlib.sha256(Path(args.reference_srt).read_bytes()).hexdigest() if args.reference_srt else None
        result['source_review_sha256']=hashlib.sha256(args.source_review.read_bytes()).hexdigest() if review else None
        result['human_audio_verified']=False
        from audio_archive import atomic_json
        atomic_json(output_path.with_suffix(output_path.suffix+'.translation')/'report.json',result)
        write_certificate(result,output_path.with_suffix(output_path.suffix+'.translation'))
        print('[grounded] '+json.dumps(dict(quality_passed=result['quality_passed'],
            missing_ids=result['missing_ids'],review_ids=result['review_ids'],cache_hits=result['cache_hits'])),flush=True)
        print('[budget] '+json.dumps(API_BUDGET.summary()),flush=True)
        return 0 if result['quality_passed'] else (3 if result['source_hold_ids'] or result['review_ids'] else 1)

    # 全局上下文预处理（仅显式 --global-context 时启用，会产生额外调用）
    global_ctx = None
    if args.global_context and not args.no_global_context:
        global_ctx_cache = Path(args.global_context_cache) if args.global_context_cache else output_path.with_suffix(".global_context.json")
        # 全局预处理需要通读全文，给更长超时
        global_timeout = max(args.timeout, 1200)
        global_ctx = generate_global_context(
            entries, args.model, args.url, args.source, args.target,
            timeout=global_timeout, cache_path=global_ctx_cache
        )

    start_time = time.time()

    if args.whole_file:
        translated = translate_whole_file(
            entries, args.model, args.url, args.source, args.target,
            timeout=max(args.timeout, 1800), system_prompt=system_prompt, global_ctx=global_ctx
        )
    else:
        translated = adaptive_batch_translate(
            entries, tm, args.model, args.url, args.source, args.target,
            args.batch_size, args.context_lines, args.max_workers,
            checkpoint_path, resume=not args.no_resume, timeout=args.timeout,
            system_prompt=system_prompt, global_ctx=global_ctx
        )

    # 全局补翻兜底
    untranslated_indices = [
        idx for idx, e in enumerate(translated)
        if looks_untranslated(e[3], args.source)
    ]
    if untranslated_indices:
        raise ValueError(f"Untranslated entries remain: {len(untranslated_indices)}; final output not written")

    # 同句不同场景可有不同译法，不用多数票覆盖。

    # 可选：LLM 一致性审校
    if args.llm_review:
        translated = llm_consistency_review(translated, entries, args.model, args.url, args.target)

    # 保存 TM
    print("[tm] 跨影片逐句缓存已禁用")

    # 最终检查
    still_bad = sum(1 for e in translated if looks_untranslated(e[3], args.source))
    if still_bad:
        raise ValueError(f"Untranslated entries remain after review: {still_bad}")
    else:
        print("[check] 无日文假名残留；这不代表语义审校通过")

    # 目标为简体中文时，统一转换为大陆简体
    if args.target == "zh-cn":
        translated = [(idx, start, end, zhconv.convert(text, "zh-hans")) for idx, start, end, text in translated]

    write_srt(output_path, translated)
    elapsed = time.time() - start_time
    print(f"[ok] 已保存: {output_path} （耗时 {elapsed:.1f}s）")
    if API_BUDGET is not None:
        print('[budget] '+json.dumps(API_BUDGET.summary()), flush=True)

    # 清理 checkpoint
    if not args.keep_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()

    return 0


if __name__ == "__main__":
    # Redirected Windows child logs must use the same encoding as the runner.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
