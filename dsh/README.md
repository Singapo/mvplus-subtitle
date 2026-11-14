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

## 环境变量

| 变量 | 默认 | 用途 |
|---|---|---|
| `MVSUB_PYTHON` | `python` / `python.exe` | 装了 `mvsub` 的解释器；虚拟环境请指到绝对路径 |
| `MVSUB_CWD` | 当前工作目录 | CLI 的工作目录，通常设成仓库 checkout |

`MVSUB_PYTHON` 几乎总要显式设置，因为 DSH 的 PATH 里未必有你的 venv：

```powershell
$env:MVSUB_PYTHON = "D:/path/to/mvplus-subtitle/.venv/Scripts/python.exe"
$env:MVSUB_CWD    = "D:/path/to/mvplus-subtitle"
```

## 版本兼容

peer 依赖按 `@deepseek-ai/dsh-tools` 的 `0.2.1-rc.2` 写定，`engines.dsh` 声明 `>=0.2.1-rc.2`。
DSH 升级到不兼容的版本线时，`dsh plugin add` 会要求显式的版本豁免确认，不会静默装坏。

## 已知限制

- 长批次会超出单次工具调用的等待上限。真正的整批建议在终端里跑 `mvsub run`，插件用
  `mvsub_status` 看进度；也可以给 `mvsub_run` 传更大的 `timeoutSeconds`，但那只是在等。
- 插件不做进度流式回传，只在结束时返回尾部输出。
