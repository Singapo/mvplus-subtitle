# mvplus-subtitle for DeepSeek Harness

把 `mvsub` CLI 挂成模型可调用的工具，于是可以直接对 DSH 说
「把 115 上这个目录里缺字幕的片补上中文字幕」。

插件是**薄壳**：它不重新实现任何流水线逻辑，只运行 `python -m mvsub <子命令>` 并回传输出。
参数、阈值、验收判据仍然只有 CLI 一份实现，不会出现「插件里一套、命令行里一套」。

## 工具

| 工具 | 性质 | 作用 |
|---|---|---|
| `mvsub_doctor` | 只读 | 环境体检：解释器、ffmpeg、cookie、密钥、模型、目录可写性 |
| `mvsub_config` | 只读 | 打印所有解析后的路径与生效的配置文件 |
| `mvsub_scan` | 只读 | 列出目标目录里缺中文字幕的视频（不下载、不花钱） |
| `mvsub_status` | 只读 | 批次进度：每部片到哪一段、哪些失败 |
| `mvsub_verify` | 只读 | 本地复核一对字幕的硬判据 |
| `mvsub_run` | **写** | 整线：下载 → 识别 → 对齐 → 翻译 → 校验 → 回传 |
| `mvsub_translate` | **付费** | 只翻译本地原文 SRT，默认限制在优惠时段 |

`mvsub_run` 会下载几十 GB、产生翻译费用、并往云端写文件。它的工具描述里明确要求模型
先和用户确认范围、并建议 `limit: 1` 试跑一部再放量。

### `mvsub_translate` 为什么钉死地址和模型

`translate_srt_batch` 的默认值是本地 Ollama（`http://localhost:11434`、`qwen2.5:14b`），
而预算模式只接受 `https://api.deepseek.com` + `deepseek-flash`，否则在**读取字幕之前**就 rc=2。
0.1.0 的插件漏传了 `--url`，于是每次调用都死在参数校验上。

现在 `buildTranslateArgs()`（`args.js`）显式传入两者，`--url` 不开放给调用者——预算模式下它没有
第二种合法取值，开放只会制造一个必然失败的旋钮。想用本地 Ollama 请直接跑 CLI（不带预算参数）。
这个建参函数有独立单测：`node dsh/test/args.test.mjs`。

## 安装

需要先按仓库 [README](../README.md) 把 Python 侧跑通（`mvsub doctor` 全绿），再装插件。

从 GitHub 装：

```bash
dsh plugin add git+https://github.com/Singapo/mvplus-subtitle.git#path:dsh
```

从本地 checkout 装（开发时更方便）：

```bash
dsh plugin add file:D:/path/to/mvplus-subtitle/dsh
```

装完把包名加进 profile 的 `dsh.profile.bundles`：

```json
{
  "dsh": {
    "profile": {
      "bundles": ["@deepseek-ai/dsh-web-app", "dsh-plugin-mvplus-subtitle"]
    }
  }
}
```

**重启 DSH 后生效**（插件在启动时装配）。

> 如果 `dsh plugin add` 报 `'pnpm' is not recognized`，说明这台机器没有 pnpm。
> 也可以把本目录整个复制到 `~/.dsh/profiles/node_modules/dsh-plugin-mvplus-subtitle/`，
> 再在 profile 的 `cordis.patch.yml` 里插一行（`dsh_base` 之外的补丁层）：
>
> ```yaml
> - insert:
>     - id: mvplus-subtitle
>       name: dsh-plugin-mvplus-subtitle
> ```
>
> 这种方式不需要 pnpm，改名/卸载都只是删目录加删这一行。

## 配置：`~/.config/mvsub/plugin.json`

**不要**依赖 DSH 进程的环境变量：宿主进程的环境在启动时就固定了，改一次路径就要重启一次 DSH；
而 PATH 上那个 `python` 往往不是装了 `mvsub` 的那个，报错只有一句 `No module named mvsub`。

所以插件读一个每次调用都会重新解析的配置文件，**改它不需要重启**：

```json
{
  "python": "D:/path/to/MVplus/.venv/Scripts/python.exe",
  "cwd": "D:/path/to/mvplus-subtitle",
  "env": {
    "PYTHONPATH": "D:/path/to/mvplus-subtitle/src",
    "MVSUB_MODEL_HOME": "D:/models/subtitle",
    "MVSUB_ENV_FILE": "D:/path/to/MVplus/.env",
    "MVSUB_HOME": "D:/media/mvsub",
    "MVSUB_CUDA_LIBS": "D:/old-deploy/subtitle/lib/cuda",
    "MVSUB_TRANSLATE_PROMPT": "D:/path/to/mvplus-subtitle/src/mvsub/pipeline/prompts/system_prompt_v9_grounded_name_policy.txt",
    "MVSUB_NAME_MAP": "D:/my/name_map.txt",
    "MVSUB_TRANSLATE_REASONING": "low",
    "MVSUB_ALLOW_ISOLATED": "1"
  }
}
```

| 键 | 必需 | 说明 |
|---|---|---|
| `python` | 是 | 装了 `mvsub` 的解释器绝对路径（3.12+） |
| `cwd` | 是 | 仓库 checkout 目录 |
| `env` | 可选 | 追加到子进程环境。源码方式运行时需要 `PYTHONPATH=<repo>/src`；模型/凭证/工作区用 `MVSUB_*` 指定 |
| `env.PYTHONPATH` | 源码方式必需 | 没 `pip install -e .` 时，靠它让 `python -m mvsub` 找到包 |
| `env.MVSUB_ENV_FILE` | 可选 | 复用已有的 `.env`（里面是 `P115_COOKIES` / `DEEPSEEK_API_KEY`） |
| `env.MVSUB_HOME` | 可选 | 归档/下载/工作目录的根；不给会落到 `~/.mvsub` |
| `env.MVSUB_CUDA_LIBS` | 建议 | 没装 `nvidia-*-cu12` wheel 时，指向现有 CUDA DLL 目录 |
| `env.MVSUB_TRANSLATE_PROMPT` | 建议 | 用 `..._name_policy.txt` 变体：含内容政策与未收录人名音译 |
| `env.MVSUB_NAME_MAP` | 建议 | 自己的 人名对照表；空模板会让表外人名留假名，整片被验收拦下 |
| `env.MVSUB_TRANSLATE_REASONING` | 建议 | `low`：实测能译出 `high` 会拒译的内容，且便宜四倍 |
| `env.MVSUB_ALLOW_ISOLATED` | 可选 | `1` = 无译文行剔除后照常发布；默认缺一行就不发 |

后四项是**跑通一部真片必需**的（见 [配置](../docs/configuration.md#翻译相关决定成败的四项)）。
不给它们，插件会按默认参数运行，遇到露骨对白或表外人名就会停在验收。**改这个文件不需要重启 DSH。**

文件带 BOM 也能读（Windows 编辑器常写 BOM）。优先级：环境变量 > 本文件 > PATH 上的 `python`。

配好后用 `mvsub_config` 与 `mvsub_doctor` 验证：`doctor` 全绿才算打通。

## 环境变量（可选，优先级更高）

| 变量 | 默认 | 用途 |
|---|---|---|
| `MVSUB_PYTHON` | `plugin.json` → `python`/`python.exe` | 装了 `mvsub` 的解释器 |
| `MVSUB_CWD` | `plugin.json` → 当前工作目录 | CLI 的工作目录 |

## 版本兼容

`engines.dsh` 声明 `>=0.2.1-rc.2`。`@deepseek-ai/dsh-tools` 在 npm 上只有
`0.0.1-rc.1`，所以本包**不声明** peer 依赖——`@deepseek-ai/dsh-tools` 与
`@deepseek-ai/cordis` 由 DSH 宿主提供。插件放在 `profiles/node_modules` 下时，
Node 会向上解析到宿主的那一份，不需要额外安装。

## 已知限制

- 长批次会超出单次工具调用的等待上限。真正的整批建议在终端里跑 `mvsub run`，插件用
  `mvsub_status` 看进度；也可以给 `mvsub_run` 传更大的 `timeoutSeconds`，但那只是在等。
- 插件不做进度流式回传，只在结束时返回尾部输出。
