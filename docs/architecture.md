# 代码结构

## 为什么是这个形状

流水线来自一个长期自用的项目，代码已经用真实影片跑了很久，行为和判据都有实测依据。
抽取成公开包时做了两个决定，都是为了**不改变行为**：

1. **只做配置外置，不做重构。** 原文件逐个复制过来，改动限于把硬编码路径换成 `config.*()` 调用。
2. **保留平铺导入。** 模块之间是 `import cloud_common`、`from asr_config import ...`，
   不是包内相对导入。

第 2 点是刻意的。改成 `from .cloud_common import` 会让同一份代码有两个模块身份
（`cloud_common` 与 `mvsub.pipeline.cloud_common` 各自一份实例），全局状态、缓存、
`isinstance` 判断都会裂开，是真实 bug 来源，收益为零。所以 `mvsub/__init__.py` 只做一件事：
把 `src/mvsub/pipeline` 塞进 `sys.path`，之后所有模块名仍是顶层的。

代价是：**不要在 `pipeline/` 里加需要包语义的代码**。要写新功能，写在 `cli.py` 或新模块，
按现有风格平铺导入。

## 目录

```
src/mvsub/
├── __init__.py        # 只做 sys.path 引导 + 版本号
├── __main__.py        # python -m mvsub
├── cli.py             # 子命令路由：把 mvsub xxx 转给对应模块的 main()
└── pipeline/          # 原样移植的流水线（26 个模块 + aligned_asr/ + prompts/）
    ├── config.py              # ← 唯一允许出现默认路径的地方
    ├── asr_config.py          # 由 config 推导出的模型/对齐器位置常量
    ├── cloud_common.py        # 115 客户端：接口知识、补丁、上传/替换事务
    ├── cloud115.py            # 115 手工单文件 CLI（ls/download/upload）
    ├── extract_srt.py         # faster-whisper 直接识别
    ├── extract_aligned_srt.py # 取音 → 识别 → 对齐 → 原子发布 SRT
    ├── align/（aligned_asr/）  # 对齐分阶段脚本：recognize/vad/align/verify/caption/repair
    ├── source_review.py       # 源文声学复核（送翻译前的证据）
    ├── translate_srt_batch.py # 批量翻译（DeepSeek / Ollama），TM + 断点续跑
    ├── grounded_translation.py# 带源文约束的分批请求 + 响应复用
    ├── translation_budget.py  # 共享预算台账，未知扣费保持预留
    ├── deepseek_offpeak.py    # 优惠时段闸门（逐请求检查）
    ├── native_subtitles.py    # 单遍原生翻译通道：身份、packet、交付
    ├── subtitle_pipeline_core.py      # 纯本地确定性核心
    ├── nosub_pipeline.py      # 主入口：scan / run / status / verify
    ├── nosub_sources.py       # 冻结清单的本地源准备（按语言选识别器）
    ├── nosub_recovery.py      # 可选的恢复证据（不改正常验收上限）
    ├── audio_archive.py       # PCM 音频归档与校验（SHA1 + 时长）
    ├── subtitle_version_reuse.py      # 不同压版的调度（判定仍看音频）
    ├── subtitle_batch.py      # SRT 日期批次编排
    ├── reprocess_subtitles.py # 已有字幕重做时间轴（冻结清单续跑）
    └── ...
```

## 各层职责

### 配置层
`config.py` → 所有路径。`asr_config.py` 只是把配置解析成常量供老代码 `from asr_config import ...`。
新代码直接 `import config`，别再加常量模块。

### 云盘层
`cloud_common.Cloud115` 是**唯一**允许写 115 接口知识的地方。它承担了 p115sync 缺失的两个补丁
（`upload_gettoken` 信封锁、尾片 Content-Length 与分片重试）和上传替换事务
（`publish_missing_subtitle` / `replace_subtitle`）。

`cloud115.py` 是历史的手工单文件入口，**没有**打这些补丁，不要用它跑批量。
详见 [115 接口](115-api.md)。

### 提取层
两条路径，按语言分流：

| 源语言 | 路径 | 产物 |
|---|---|---|
| 日语 | `extract_aligned_srt` → `aligned_asr/` 分阶段（识别 / VAD 证据 / 强制对齐 / 内容校验 / 词恢复 / 事件校验 / 字幕生成 / 修补） | 带 `timing_policy` 与复盘的 SRT + `.alignment.json` |
| 其他 | `extract_srt` 的 word-timestamp 词钟 | 显式记录为**不同**的源方法 |

日语的独立对齐是「识别器」和「对齐器」分开跑，不互相喂结果——这是为了避免用同一个模型的
错误自我确认。

### 翻译层
- `translate_srt_batch.py`：真正的执行器。分批、带相邻上下文、TM 缓存、checkpoint、多层解析兜底。
- `grounded_translation.py`：把「只根据原文翻」落成代码，并复用已付费的响应。
- `translation_budget.py`：预算台账。**未知扣费保持预留**，不因为进程退出就当成没花钱。
- `deepseek_offpeak.py`：优惠时段闸门，每个请求前查一次。

### 编排层
`nosub_pipeline.py` 是主入口，四个并发池 + 看门狗 + 状态落盘。
`nosub_sources.py` 负责「这一部该用哪个识别器、证据够不够」。

### 验收
`nosub_pipeline.cmd_verify` 是纯本地、不联网的判据执行者；常量集中在文件顶部
（`KANA_RE`、`CPS_*`、`CUE_TAIL_TOLERANCE`、`TIMELINE_TOL_MS` 等），改阈值要连着文档一起改。

## CLI 是怎么接的

`cli.py` 不重新定义参数，只把 `mvsub run --foo` 变成 `nosub_pipeline.main()` 的 `run --foo`。
理由：参数只在模块里定义一次，文档和 `--help` 不会漂移。

```python
HANDLERS = {"run": cmd_nosub, "extract": cmd_extract, ...}
```

`extract` / `reprocess` / `review` 没有 `main()`，用 `runpy.run_module(..., run_name="__main__")`
执行它们的 `if __name__ == "__main__"` 段，行为与直接跑脚本一致。

## 测试

```
tests/                 # 从上游原样移植，bootstrap 行改指到 pipeline/
tests/conftest.py      # 6 个上游遗留失败标 xfail 并写明原因
```

运行：

```bash
pytest tests -q
```

测试不需要 GPU、不联网、不读云盘；需要模型的用例会自行跳过或用临时目录构造。

## 改代码的安全边界

- 加路径 → 只改 `config.py`，不要在别处写默认值。
- 加 115 接口调用 → 只加在 `cloud_common.py`。
- 改验收阈值 → 同时改 [pipeline.md](pipeline.md) 的表述，否则文档会变成谎话。
- 不要「顺手」把平铺导入改成相对导入。
