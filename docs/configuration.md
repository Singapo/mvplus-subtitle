# 配置

配置的目标只有一个：**让代码里不出现任何绝对路径**。所有路径都在 `src/mvsub/pipeline/config.py`
里解析一次，其余模块只调用函数，不再各写一份默认值。

## 三层优先级

```
环境变量 MVSUB_<KEY>   >   配置文件   >   默认值
```

配置文件的位置，按顺序取第一个存在的：

1. `$MVSUB_CONFIG`
2. `./mvsub.toml`（当前工作目录）
3. `~/.config/mvsub/config.toml`

`mvsub config` 会打印「实际生效的值」和「配置文件路径」，排查配置问题先跑它。

## 键一览

环境变量名 = `MVSUB_` + 键名大写。

| 键 | 环境变量 | 默认值 | 用途 |
|---|---|---|---|
| `home` | `MVSUB_HOME` | `~/.mvsub` | 所有默认路径的根 |
| `model_home` | `MVSUB_MODEL_HOME` | `<home>` | 模型树根；其下有 `models/` 与 `align-env/` |
| `asr_model` | `MVSUB_ASR_MODEL` | `<model_home>/models/japanese-turbo` | 日文识别模型（需 `model.bin`） |
| `multilingual_model` | `MVSUB_MULTILINGUAL_MODEL` | `<model_home>/models/whisper-large-v3` | 非日语源识别模型 |
| `align_model` | `MVSUB_ALIGN_MODEL` | `<model_home>/models/qwen-aligner` | 强制对齐器（需 `config.json`） |
| `align_python` | `MVSUB_ALIGN_PYTHON` | `<model_home>/align-env/{Scripts/python.exe,bin/python}` | 对齐器解释器 |
| `align_libs` | `MVSUB_ALIGN_LIBS` | 本包的 `pipeline/` 目录 | 对齐器要 import 的 `aligned_asr` 所在目录 |
| `model_libs` | `MVSUB_MODEL_LIBS` | `<model_home>/align-libs` | 模型侧额外 site-libraries（可选） |
| `archive_dir` | `MVSUB_ARCHIVE_DIR` | `<home>/originals` | 日文原文归档（只留本地） |
| `download_dir` | `MVSUB_DOWNLOAD_DIR` | `<home>/downloads` | 下载默认落地目录 |
| `work_dir` | `MVSUB_WORK_DIR` | `<home>/work` | 状态、缓存、断点、临时文件 |

`model_home` 指的是**装着 `models/` 那一层的目录**，不是 `models/` 本身。这样如果你之前已经有一套
`<某目录>/models/japanese-turbo` 的部署，直接把 `MVSUB_MODEL_HOME` 指到 `<某目录>` 就能复用，不用挪文件。

`home` 同样受三层优先级约束，**写进配置文件即可生效**：

```toml
home = "D:/media/mvsub"
```

设了它，没单独配置的 `archive_dir` / `download_dir` / `work_dir` / `model_home` 全部跟着走，
不会留在用户目录。`mvsub config` 的第一行就是解析后的 `home`。

> 0.1.0 曾经忽略配置文件里的 `home`（只在导入时读环境变量），导致「设了 D 盘却还落在 C 盘」。
> 现在 `home()` 走统一的三层解析；`tests/test_release_regressions.py` 有对应用例。

`align_python` 在 `[asr]` 段下（写 `mvsub.toml` 时是 `[asr] align_python = ...`），其余都在 `[paths]` 段。
写在顶层的同名键也认，等价于 `[paths]`。

示例见 [examples/mvsub.toml](../examples/mvsub.toml)。

## 凭证：只走环境或 `.env`

密钥**不进** `mvsub.toml`，避免误提交。

| 变量 | 必需性 | 用途 |
|---|---|---|
| `P115_COOKIES` | 云盘命令必需 | 115 cookie，需含 `UID`/`CID`/`SEID`/`KID` |
| `DEEPSEEK_API_KEY` | 付费翻译必需 | DeepSeek API |

`.env` 的查找顺序：`$MVSUB_ENV_FILE` → 仓库根 `.env` → 当前目录 `.env`。

解析器是内置的十几行（`KEY=VALUE`、`#` 注释、可选引号），不引入 `python-dotenv`。
它只解析、不自动导出：需要把变量传给子进程的模块会显式 `os.environ.setdefault`。

## `mvsub doctor` 会检查什么

```
python>=3.12   ffmpeg   ffprobe
P115_COOKIES（含字段完整性）   DEEPSEEK_API_KEY
asr_model + model.bin   align_model + config.json   align_python
archive_dir / download_dir / work_dir 的可写性
```

只做本地检查，**不联网、不读云盘**。退出码 0 表示全部通过。

缺模型时会报 `FAIL asr_model`，这是预期行为：本项目不替你下载模型。

## 为什么对齐器要单独一个解释器

强制对齐器依赖 torch，而主流水线依赖 faster-whisper + ctranslate2。两者对 CUDA / cuDNN 的版本
要求不同，装进同一个环境里互相顶掉是常见事故。所以默认让它用独立的 `align-env`，
由 `align_python` 指定，主流水线只负责 `subprocess` 调它。

## 从旧部署迁移

如果之前已经有一套能跑的目录（`<根>/models/japanese-turbo`、`<根>/models/qwen-aligner`、
`<根>/align-env`），只需要：

```bash
MVSUB_MODEL_HOME=<根> mvsub doctor
```

模型不用搬、不用重新下载。只把 `archive_dir` / `download_dir` / `work_dir` 指到你原来的位置即可。
