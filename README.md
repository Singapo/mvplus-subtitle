# mvplus-subtitle

给云盘（115 网盘）里的影片补中文字幕：**下载 → 本地语音识别 → 强制对齐 → LLM 翻译 → 校验 → 回传原目录**。

不依赖任何字幕站的现成字幕，用本机 GPU 从声音里把台词抠出来，再把中文写回云盘、和视频同名，
播放器直接加载。也是 **DeepSeek Harness 插件**（见 [DSH 插件](#dsh-插件)）。

> Subtitle a film that only exists in your cloud drive: download → local ASR → forced alignment →
> LLM translation → verification → upload back to the same folder with the same name. No subtitle
> site involved. Ships as a CLI and as a DeepSeek Harness plugin.

```
┌──────────────┐   ┌──────────────────────────┐   ┌──────────────┐   ┌──────────────┐
│ 下载（3 并发）│ ∥ │ 取音 → 识别 → 强制对齐     │ ∥ │ 翻译（2 并发）│ ∥ │ 校验 → 回传   │
│ 云盘 → 本地   │   │ + 局部声学复核（GPU 串行 1）│   │ 分批 + 上下文 │   │ 同名事务替换  │
└──────────────┘   └──────────────────────────┘   └──────────────┘   └──────────────┘
```

四段是独立的并发池，不是串行管线：识别一部的同时预取下一部、并行翻译已就绪的字幕。
详见 [流水线流程与验收](docs/pipeline.md)。

## 它解决什么

- 老片、冷门片只有视频，没有中文字幕；
- 机器翻译字幕站要注册、要额度、还经常没有对应的片；
- 手工跑 whisper + 翻译 + 上传，每次都要重新解决时钟、时间轴、编码、同名覆盖这些坑。

本仓库把这些坑的解法固化成代码和不变量，并把每一步的验收判据写进文档：**结构校验只能证明
可解析、时间轴一致、无假名残留，不能证明「每条都对」**——这条边界写在 [踩坑与约束](docs/pitfalls.md) 里。

## 环境要求

| 项目 | 要求 | 说明 |
|---|---|---|
| Python | 3.12+ | `p115client` 要求 `>=3.12,<4` |
| FFmpeg | `ffmpeg` / `ffprobe` 在 PATH | 取音、探测时长 |
| GPU | NVIDIA，显存足够跑 760M ASR + 0.6B 对齐 | CPU 能跑但极慢 |
| CUDA DLL | `cublas64_12` / cuDNN（见下） | 缺了 GPU 提取直接报 `cublas64_12.dll is not found` |
| 模型 | 日文 ASR + Qwen 强制对齐器（见下表） | **不自动下载** |
| 对齐环境 | 独立解释器 + torch/transformers/nagisa | 见下「对齐环境」 |
| 账号 | 115 cookie（读目录/下载/上传） | 只需要你自己的账号 |
| 翻译 | DeepSeek API key，或人工/Agent 通道 | 付费翻译默认走 DeepSeek |

CUDA DLL 有两个来源，任选其一：

```bash
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12   # 落到 site-packages，自动被识别
# 或者：把已有的 DLL 目录指过来（`nvidia/<pkg>/bin` 布局），复用旧部署
# MVSUB_CUDA_LIBS=D:/old-deploy/subtitle/lib/cuda
```

只加 PATH **不行**：Windows 上 Python 3.8+ 的 DLL 搜索不再看 PATH，必须走
`os.add_dll_directory`，这也是它被做成配置项的原因。`mvsub doctor` 会报 `cuda_libs`。

### 模型准备（必须手工做一次）

本项目**不会**自动下载模型，也不带模型权重。五个模型的确切来源（Hugging Face 仓库 id）、许可、
下载命令与版本约束写在 **[模型来源](docs/models.md)**，这里只列放置位置和用途：

| 目录 | 用途 | 需要的分支 |
|---|---|---|
| `models/japanese-turbo/` | 日文识别（760M） | **必需**：所有日语片的提取 |
| `models/qwen-aligner/` | Qwen 强制对齐器 | **必需**：日语独立对齐 |
| `models/whisper-large-v3/` | 多语种识别 + 日语时间轴恢复 + 源文复核的第二识别器 | 强烈建议：非日语源、重做时间轴、声学复核都要 |
| `models/qwen-asr-verifier/` | 局部声学复核（判断识别内容是否真的说得出来） | 可选：`source_review` / 重译线要用 |
| `models/audio-event-verifier/` | 声音事件校验（区分对白与非语言发声） | 可选：`verify_events` 阶段要用 |

只跑「日文片 → 字幕」这条最短路径，前两个即可；一旦要用源文复核、声音事件校验、
非日语源或从视频重做时间轴，后三个都会被加载，缺了会在对应分支明确报 missing。

一条命令下最短路径（换掉 `<model_home>` 即可，完整说明见 [docs/models.md](docs/models.md)）：

```bash
pip install -U "huggingface_hub[cli]"
hf download efwkjn/faster-whisper-ja-760M      --local-dir <model_home>/models/japanese-turbo
hf download Qwen/Qwen3-ForcedAligner-0.6B-hf   --local-dir <model_home>/models/qwen-aligner
```

```
~/.mvsub/
├── models/
│   ├── japanese-turbo/          # 需含 model.bin
│   ├── qwen-aligner/            # 需含 config.json
│   ├── whisper-large-v3/        # 非日语源 / 日语时间轴恢复 / 源文复核
│   ├── qwen-asr-verifier/       # 局部声学复核
│   └── audio-event-verifier/    # 声音事件校验
└── align-env/                   # 对齐器专用环境（见下）
```

`MVSUB_MODEL_HOME` 指到**装着 `models/` 那一层的目录**（即 `~/.mvsub`）。已经有这套目录的话直接指过去，
模型不用搬。

### 对齐环境（`align-env`）

对齐器跑在**独立的解释器**里，因为它的依赖集与主流水线不同（torch 与 ctranslate2 对 CUDA/cuDNN 的
要求会互相顶掉）。默认位置 `<model_home>/align-env/`，可用 `MVSUB_ALIGN_PYTHON` 或
`mvsub.toml` 的 `[asr] align_python` 覆盖。

它需要能 import 这五个包（下面是本项目实测通过的组合，仅供参考，不是唯一可行版本）：

```bash
python -m pip install "torch==2.7.1" "transformers==5.22.1" nagisa==0.3.1 "numpy==2.5.4" "scipy==1.18.1"
```

| 包 | 被谁用 | 实测版本 |
|---|---|---|
| `torch` | `align.py`、`verify_content.py`、`verify_events.py` | 2.7.1+cu128 |
| `transformers` | 同上（`AutoModelForTokenClassification` / `AutoModelForMultimodalLM` / `AutoModelForAudioClassification`） | 5.22.1 |
| `nagisa` | 分词。**两个环境都要装**：`align.py` / `verify_content.py` / `verify_events.py` 在对齐环境，`caption.py` 在主环境 | 0.3.1 |
| `numpy` | 全部分阶段脚本 | 2.5.4 |
| `scipy` | WAV 读写与相关运算（含主流水线的 `recognize` / `recover_words`） | 1.18.1 |

`transformers` 的版本不能太低：`AutoModelForMultimodalLM` 与 `AutoModelForAudioClassification`
在旧版本里不存在，会在 `verify_content` / `verify_events` 阶段直接 ImportError。

**`nagisa` 不是只装对齐环境就够。** 分阶段脚本里 `caption`（字幕生成）与 `repair` 跑在**主解释器**上，
`caption.py` 直接 `import nagisa`。本项目已把它列进 `pyproject.toml` 的依赖（`uv sync` 会装），
对齐环境仍需单独装一份。`mvsub doctor` 会分别检查两处：`runtime_deps`（主解释器）与
`align_env`（对齐环境）。

`mvsub doctor` 会检查这个环境能不能 import 上述依赖，并单独报告可选复核模型是否就位。

## 安装

```bash
git clone https://github.com/Singapo/mvplus-subtitle.git
cd mvplus-subtitle
uv sync          # 或 pip install -e .
```

## 配置

三层，后者覆盖前者：**默认值 → 配置文件 → 环境变量**。

```bash
mvsub config     # 打印所有解析后的路径，配置对不对一眼就看出来
mvsub doctor     # 体检：解释器、ffmpeg、模型、密钥、目录可写性
```

最小配置只需要密钥：

```bash
cp .env.example .env
# 编辑 .env，填 P115_COOKIES 与 DEEPSEEK_API_KEY
```

路径相关的写在 `mvsub.toml`（见 [examples/mvsub.toml](examples/mvsub.toml)）：

```toml
home = "D:/media/mvsub"

[paths]
model_home = "D:/models/subtitle"
work_dir   = "D:/media/mvsub/work"
```

完整说明见 [配置](docs/configuration.md)。密钥只放 `.env` 或环境变量，**不要**写进 `mvsub.toml`。

### 翻译相关四项（决定一部片能不能跑通）

```toml
translate_reasoning = "low"          # low / high / max
translate_prompt    = "src/mvsub/pipeline/prompts/system_prompt_v9_grounded_name_policy.txt"
name_map            = "D:/my/name_map.txt"   # 自己的表，别提交
allow_isolated      = "1"            # 无译文的行剔除后照常发布
```

这四项不是调优，是**实测出来的必需项**：

- `low` 推理：同一批露骨对白 `low` 全译出（¥0.0067），`high` 全拒译（¥0.0251）；
- `name_policy` 提示词：内容政策 + 未收录人名按读音译中文，实测能过；
- `name_map`：留空模板时表外人名会留假名，验收直接拦下整片；
- `allow_isolated`：受限/待核行剔除后发布，默认关（缺行就不发）。

详见 [配置：翻译相关四项](docs/configuration.md#翻译相关决定成败的四项)。

## 快速开始

```bash
# 1. 体检
mvsub doctor

# 2. 只读预检：列出某个云盘目录里缺中文字幕的视频（不下载、不上传）
mvsub scan --remote-cid <目录 cid>

# 3. 正式跑（下载 ∥ 提取 ∥ 翻译 ∥ 回传；中断后原样重跑即续跑）
#    付费翻译必须绑定预算台账：两个参数缺一个就 rc=2，这是刻意的闸门
mvsub run --remote-cid <目录 cid> \
    --budget-file ./ledger.json --budget-cny 20

#    第一次先只跑一部，把环境问题一次暴露
mvsub run --remote-cid <目录 cid> --limit 1 \
    --budget-file ./ledger.json --budget-cny 5

# 4. 看进度
mvsub status --remote-cid <目录 cid>

# 5. 本地复核一对字幕（不联网、不写状态）
mvsub verify "<video|->" "<原文.ja.srt>" "<中文.srt>"
```

本地单文件、只翻译、已有字幕重做时间轴：

```bash
mvsub extract "D:/movies/movie.mkv" --output "D:/movies/movie.ja.srt"

# 只翻译：同样必须带预算台账；--url 只接受官方地址（预算模式约束）
mvsub translate "D:/movies/movie.ja.srt" --output "D:/movies/movie.srt" \
    --model deepseek-flash --url https://api.deepseek.com --grounded \
    --batch-size 16 --max-workers 2 --off-peak-only \
    --budget-file ./ledger.json --budget-cny 10

mvsub reprocess "<plan.json>" --budget-file ./ledger.json --budget-cny 20   # 四段并行重做
```

> `translate` 的 `--model` / `--url` 默认值是本地 Ollama（`qwen2.5:14b`、`localhost:11434`）。
> 一旦带上预算参数，入口只接受 `deepseek-flash` + `https://api.deepseek.com`，
> 否则在读取字幕之前就以 rc=2 退出。上面两条命令把这两个值写全，就是因为这个。

`mvsub <子命令> --help` 打印的是各模块自己的原始参数——这里没有二次包装，不会出现
「文档写的参数代码里没有」的偏差。

## 成本

翻译是唯一的付费环节，也是唯一需要算计的地方：

- 默认 `--batch-size 16`、`--max-output-tokens 16384`、`--max-workers 2`、`--off-peak-only`；
- 请求只携带相邻句作为上下文，**不重复发送整片全文**；
- 短对白不默认开思维链（推理常占输出 90% 以上）；
- 断点续跑用 checkpoint 与本地响应缓存，中断重跑不重复付费。

实测数字（思考 token 占比、批次上下文体积的两处翻车与修正）见
[踩坑与约束](docs/pitfalls.md) 第 4 章「翻译：提示词与验收判据」。

## DSH 插件

仓库内 `dsh/` 是一个 DeepSeek Harness 插件包，把 CLI 挂成模型可调用的工具
（`mvsub_scan` / `mvsub_run` / `mvsub_doctor` 等），因此可以直接对 DSH 说
「把 115 上这个目录里缺字幕的片补上中文字幕」。

```bash
dsh plugin add git+https://github.com/Singapo/mvplus-subtitle.git#path:dsh
```

也可以从本地路径装：`dsh plugin add file:D:/path/to/mvplus-subtitle/dsh`。
细节与依赖见 [dsh/README.md](dsh/README.md)。

## 已知边界

诚实列出，不掩饰：

- **跨版本复用未包含**：拿已发布的字幕去套同片的不同压制版本，需要额外的一次性脚本；
  本发行版遇到该分支会**明确失败并阻止重跑 ASR**，而不是静默复制。
- **6 个上游遗留失败用例**标为 `xfail` 并写明原因（p115 版本契约、块大小常量漂移、`kind` 口径分歧），
  见 [tests/conftest.py](tests/conftest.py)。移植没有改动这些行为。
- **Windows 优先**：主路径在 Windows 上验证；层级/编码相关代码有平台判断，但 Linux 未系统验证。
- **不做「每条都对」的承诺**：自动判据只覆盖结构、时间轴、假名残留、CPS 等可判定项；
  语义正确性仍需人工抽查。
- **不做持续值守**：这是单次运行的 CLI，没有守护进程。要长期保障请自己用计划任务包一层。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/models.md](docs/models.md) | 五个模型的确切来源、许可、下载命令、版本约束 |
| [docs/pipeline.md](docs/pipeline.md) | 四段并行、状态机、断点续跑、验收判据、上传事务 |
| [docs/configuration.md](docs/configuration.md) | 配置优先级、每个键的含义、`.env` 键 |
| [docs/pitfalls.md](docs/pitfalls.md) | 踩坑与约束：115 接口、音视频时钟、识别对齐、翻译验收、成本 |
| [docs/architecture.md](docs/architecture.md) | 代码结构与模块职责，改哪里不会踩坏别处 |
| [docs/115-api.md](docs/115-api.md) | 115 接口实测行为与必须打的补丁 |

## 许可

MIT，见 [LICENSE](LICENSE)。

第三方组件（faster-whisper、p115sync、FFmpeg 等）遵循各自许可，模型权重需自行获取并遵守其许可。
本项目由作者自用的 `MVplus` 项目抽取而来；抽取只做了配置外置与路径通用化，流水线行为未变。
