# 模型：来源、许可、放置位置

本项目**不会**自动下载模型，也不会替你去某个站点点下载。原因很直接：这些权重加起来约 8.3 GB，
来源、版本和许可需要你自己确认一眼；静默下载一个 3 GB 的模型到用户目录，不是 CLI 该悄悄做的事。

下面是本项目实测使用的**确切来源**（`huggingface.co` 上的仓库 id），以及各自放在哪里。

## 一张表看完

| 放置目录（`<model_home>/models/`） | Hugging Face 仓库 | 许可 | 大小 | 关键文件 | 谁需要它 |
|---|---|---|---|---|---|
| `japanese-turbo/` | [`efwkjn/faster-whisper-ja-760M`](https://huggingface.co/efwkjn/faster-whisper-ja-760M) | 上游未声明 | ~1.4 GB | `model.bin` | **必需**：所有日语片识别 |
| `qwen-aligner/` | [`Qwen/Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) | Apache-2.0 | ~1.8 GB | `config.json` | **必需**：日语独立对齐 |
| `whisper-large-v3/` | [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3) | MIT | ~2.9 GB | `model.bin` | 非日语源、日语时间轴恢复、源文复核的第二识别器 |
| `qwen-asr-verifier/` | [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) | Apache-2.0 | ~1.5 GB | `config.json` | 局部声学复核（`source_review`、重译线） |
| `audio-event-verifier/` | [`MIT/ast-finetuned-audioset-10-10-0.4593`](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | BSD-3-Clause | ~0.66 GB | `config.json` | 声音事件校验（`verify_events` 阶段） |

「必需」只针对最短路径：**日语片 → 中文 SRT**。一旦要用源文复核、声音事件校验、非日语源，或从视频
重做时间轴，后三个都会被加载，缺了会在对应阶段明确报 missing。

许可两处需要你自己判断：

- `efwkjn/faster-whisper-ja-760M` 与其上游 `efwkjn/whisper-ja-760M` **都没有声明许可证**。
  本项目只是调用它；能否用于你的用途，需要你自己确认。
- 其余四个都是宽松许可（MIT / Apache-2.0 / BSD-3-Clause），但仍建议看一下模型卡里的使用条款。

## 下载

用 Hugging Face 官方 CLI（新版叫 `hf`，旧版叫 `huggingface-cli`），逐个下到对应目录：

```bash
pip install -U "huggingface_hub[cli]"

# 把 <model_home> 换成你的模型根，例如 ~/.mvsub
hf download efwkjn/faster-whisper-ja-760M --local-dir <model_home>/models/japanese-turbo
hf download Qwen/Qwen3-ForcedAligner-0.6B-hf --local-dir <model_home>/models/qwen-aligner

# 下面三个按需
hf download Systran/faster-whisper-large-v3 --local-dir <model_home>/models/whisper-large-v3
hf download Qwen/Qwen3-ASR-0.6B-hf --local-dir <model_home>/models/qwen-asr-verifier
hf download MIT/ast-finetuned-audioset-10-10-0.4593 --local-dir <model_home>/models/audio-event-verifier
```

`--local-dir` 是必须的：模型要落在固定目录里，由 `model_home` 指过去，而不是躺在 HF 缓存里。

国内网络可以走镜像，不需要改任何配置：

```bash
HF_ENDPOINT=https://hf-mirror.com hf download ...
```

## 版本约束（实测踩过的）

| 约束 | 原因 | 本项目实测 |
|---|---|---|
| `ctranslate2 >= 4.7.1` | 日文 760M 模型卡明确要求：词表变了，旧版 ctranslate2 加载不了 | 4.8.0 |
| `transformers` 不能太旧 | 对齐与复核用到 `AutoModelForMultimodalLM`、`AutoModelForAudioClassification`，旧版没有这些类，会在 `verify_content` / `verify_events` 阶段 ImportError | 5.22.1 |
| 对齐器用独立解释器 | torch 与 ctranslate2 对 CUDA / cuDNN 的版本要求会互相顶掉 | torch 2.7.1+cu128、nagisa 0.3.1、numpy 2.5.4、scipy 1.18.1 |

对齐环境的完整依赖清单见 [README 的「对齐环境」一节](../README.md#对齐环境align-env)。

## 下完怎么确认

```bash
mvsub doctor
```

它会检查：

- `asr_model/model.bin`、`align_model/config.json` 是否存在；
- 对齐器解释器能否 import `torch` / `transformers` / `nagisa` / `numpy` / `scipy`；
- 三个可选复核模型是否存在（缺了只报 `warn`，因为最短路径不需要）。

全绿才算能开跑。只想确认路径解析对不对，用 `mvsub config`。

## 目录长什么样

```
<model_home>/
├── models/
│   ├── japanese-turbo/          # model.bin、config.json、tokenizer.json、vocabulary.txt ...
│   ├── qwen-aligner/            # config.json、processor_config.json、model.safetensors ...
│   ├── whisper-large-v3/
│   ├── qwen-asr-verifier/
│   └── audio-event-verifier/
└── align-env/                   # 独立 Python 环境（对齐器用）
    └── Scripts/python.exe       # Linux/macOS 下是 bin/python
```

模型目录里通常还会带模型自带的 `README.md`（HF 模型卡），不用删——上面的来源信息就是从它和
HF 元数据里核对的。
