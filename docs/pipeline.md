# 流水线流程与验收

给云盘里的影片补中文字幕，只有一条主路径：**下载 ∥ 提取（含声学复核）∥ 翻译 ∥ 校验回传**。四段各自成池、并行推进、可断点续跑。

写给**第一次跑通的人**：每段为什么这样切、状态怎么流转、什么条件算验收通过、上传为什么必须走事务、原文怎么归档。每个阈值都能在代码常量里找到出处（`src/mvsub/pipeline/nosub_pipeline.py` 顶部），本文只解释，不另立数字。

> 本文描述 `mvsub run` 这条「缺字幕整线」。已有字幕重做（`mvsub reprocess`）是另一套状态机（`downloading_audio → … → uploaded`，失败为 `failed_*`），恢复步骤不可混用；只翻译已有原文用 `mvsub translate`；纯本地复核用 `mvsub verify`。

---

## 1. 全景

```
mvsub doctor   体检：解释器 / ffmpeg / 模型 / 密钥 / 目录
mvsub scan     只读：缺哪些片、多大、盘够不够（不下载）
mvsub run      下载(3) ∥ 提取+复核(1) ∥ 翻译(2) ∥ 校验回传(2)
mvsub status   进度与差哪些
mvsub verify   单部只读复核（不联网、不写状态）
```

**入队条件只有一条**：视频在、同名 `.srt` 不在（`survey()`）。这是整条线唯一的幂等依据——也是「跑完再跑一遍什么也不做」的原因，以及云端已有同名文件时不会自愈的原因（§3.5）。

第一次跑的顺序就是下面这几条：`mvsub doctor` 全绿 → `mvsub scan` 确认清单与磁盘 → `mvsub run --limit 1` 先跑一部 → 看 `run.log` 的里程碑（`下载完成 → 提取完成 → 翻译完成 → 中文已回传 → ✅ 验收通过`）→ `mvsub verify` 抽查 → `mvsub status` 看汇总。收尾时工作目录只剩 `state.json` 与 `run.log`。

---

## 2. 四段并行

### 2.1 为什么这样切

按**卡在什么资源上**切，而不是按步骤顺序：下载卡带宽与磁盘、提取卡显存、翻译卡外部 API 并发与费用、回传卡上传带宽。串行时最慢的 GPU 段会把网络与 API 时间完全暴露；四段成池后，识别这一部时已在预取下一部。它们是四个独立队列 + 四个独立线程池，某段变慢只让上游排队。

### 2.2 并发数与水位

| 参数 | 值 | 常量 |
|---|---|---|
| 下载并发 | 3 | `DL_WORKERS` |
| 提取并发 | **1** | `EXTRACT_WORKERS`（§2.3） |
| 翻译文件并发 | 2 | `TRANSLATE_WORKERS`（每文件内再分 2 请求：`TRANSLATE_EXTRA` 的 `--max-workers 2`） |
| 上传并发 | 2 | `UPLOAD_WORKERS` |
| 在途囤货上限 | 3 | `MAX_QUEUED_DL`（「已下载未提取」，不爆盘也不让 GPU 饿着） |
| 下载磁盘下限 | 15 GB | `DISK_FLOOR_GB`（下载途中低于此值暂停，等提取腾空间） |
| 看门狗超时 | 1800/1800/900/600 s | `REAPER_SLACK`（下载/提取/翻译/上传） |

**实际下载线程数是 `min(DL_WORKERS, 条目数)`**，只有 1 部片时不会起 3 个线程。这些值是单机、单卡、单盘、单账号的实测结论，换机器/模型/限速档位都要重测。

### 2.3 为什么 GPU 段必须是 1

1. **显存是一次性买断的**：识别模型（760M，FP16）+ 强制对齐器（0.6B）常驻。第二个并发提取不是「慢一半」，而是 OOM，或静默溢出到内存——后者更糟：看起来在跑，实际慢到不可用，日志里还没有明确错误。
2. **提取是一串独占型工作**：`do_extract()` 起子进程跑 `extract_aligned_srt.py`，内部按固定顺序跑 9 个阶段（`recognize → vad_evidence → align → verify_content → recover_words → verify_events → align → caption → repair`），对齐阶段还用**另一个解释器**（`ALIGN_PYTHON`，能 import torch 的独立环境）。它们共享同一份音频与同一块显存，并行只会互相抢。非对白声音（拟声/呼吸）在这一层排除并记入 `alignment.json` 的 `omissions`，不进待译 SRT；翻译前还要过最多 8 条风险 cue 的局部声学复核（`ensure_review(max_checks=8)`）。
3. **GPU 段是整批的节拍器**：决定总时长的是它。所以代码不是「让 GPU 并发」，而是「让别的段别拖它」——`MAX_QUEUED_DL = 3` 保证它有下一部等着，又不超过磁盘能承受的在途量。
4. **同一块卡上不要再叠别的显存任务**（去马赛克、本地大模型）。这是运行前提，不是可调参数。

### 2.4 队列、背压与收尾

- **按体积小→大取任务**：小的先出成果，便于早期验收。
- **囤货背压**：取新任务前看 `extract_q.qsize()`，`>= MAX_QUEUED_DL` 就等；磁盘余量低于 30 GB 等 30 s，否则 0.5 s 轮询。
- **收尾用计数式屏障 `StageGate`**：每级记住总条目数，处理满额那一条由「不变量」触发，向下游投`下游线程数`个终止符——既不会**漏投**（线程永久阻塞），也不会**多投**（下游池被提前关掉、队列条目没人取 → 静默漏传）。
- **上游失败也转发**：`ForwardQueue` 在 `finally` 里没转发就投 `_skip_pipeline=True`，保证下游计数不变。
- **看门狗只判断收尾**：两个屏障都关闭后才计时，超 `REAPER_SLACK` 才补终止符，整轮超 20000 s 才 dump 线程栈并 `os._exit`；优惠等待与长片识别不算死锁。

### 2.5 提取段内部

取音（16 kHz 单声道 PCM，`aresample=16000:async=16000:first_pts=0` 对齐源 PTS，先写 `.part.wav` 再原子替换）→ 九阶段 → **质量门**（cue 时间合法/不重叠/不超长 >8 s/不超短 <0.2 s、issues 为空、未解决事件 0，音轨 sha1 与流水线版本未变）→ 全过才**原子替换**输出，否则抛错并保留旧输出。

---

## 3. 状态机与断点续跑

### 3.1 状态落点

目录 `WORK_ROOT/nosub/<工作名>/`（默认 `cid<数字>`，`--name` 可指定），文件 `state.json` + `run.log`。`state.json` 先写 `.tmp` 再 `replace`（原子），进程被杀不会留下半个 JSON。key 是视频 stem——「字幕与视频同名」这条规则同时也是状态主键。

### 3.2 正常流转

```
downloaded ──► extracted ──► translated ──► uploaded
```

| 状态 | 何时写入 | 附带记录 |
|---|---|---|
| `downloaded` | 视频落盘（或本地已有且非空） | `file`、`size` |
| `extracted` | 原文 SRT 产出并过质量门 | `srt_bytes`、`audio_archive(_sha1)` |
| `translated` | 中文 SRT 产出且审计通过 | `zh` |
| `uploaded` | **上传 + 验收全过 + 本地已清理** | `name`、`bytes`、`sha1`、`cues`、`cloud_sha1`、`cloud_sha_checked`、`verify` |

**`uploaded` 的语义是「完成」，不是「传上去了」。** 上传后验收没过就不是 `uploaded`，而是 `skipped_verify_failed`，本地文件全部保留。

### 3.3 失败与等待状态

| 状态 | 含义 | 重跑会怎样 |
|---|---|---|
| `skipped_download` | 下载 3 次失败 / 异常 | 会重试 |
| `skipped_disk` | `size × 1.5` 预检，等 20×30 s 仍不足 | 会重试 |
| `skipped_missing_video` | 任一段找不到视频 | 会重试，**先查为什么没了** |
| `skipped_extract_failed` | 提取子进程 rc≠0 或抛异常 | 会重试 |
| `skipped_empty_srt` | `parse_srt_cues()` 解析出 0 条 | 会重试；**删视频不算完成** |
| `skipped_translate_failed` | rc≠0、无 `.zh-cn.srt`、异常、付费参数缺失 | 会重试 |
| `skipped_missing_translation` | 上传段原文或中文任一不在 | 会重试，**先查为什么丢** |
| `skipped_archive_failed` | 原文归档失败 **或** 音轨归档失败 | 修目录后重试 |
| `skipped_verify_failed` | 本地或云端验收失败，带 `stage=local\|cloud` | `local` 会重试；**`cloud` 不会**（§3.5） |
| `skipped_name_exists` | 上传前查到云端已有同名 | 需人工处理云端那份 |
| `skipped_upload_failed` | `cloud.upload()` 3 次都失败 | 会重试（本地可复用） |
| `waiting_source_review` | 缺原始音轨、或翻译返回 rc=3 | 补齐证据后重跑 |

两个易误读点：`skipped_archive_failed` **不只属于原文归档**（音轨归档失败也用这个值，看 `reason` 与日志区分）；`waiting_source_review` 是「等待」不是「失败」。

### 3.4 断点续跑：每段复用什么、不重算什么

中断后**原样重跑同一条命令**即可，不用加开关。判定依据「状态文件 + 磁盘现状」，各段规则不同：

| 段 | 判定「已完成」的依据 | 不重算什么 |
|---|---|---|
| 下载 | `find_on_disk()` 有同名视频且体积 >0 | 不重下（同进程重试按 `Range` 续传 `.part`；重启后先清 `*.part`，从头下） |
| 提取 | `require_current_timing()` 通过：`timing_policy=='utterance_clock_v2'`、`quality_passed`、`pipeline_revision`/人工复核摘要/音轨与原文 sha1 均一致 | 不重识别；不通过时旧原文先存 `<stem>.alignment/prior_sources/` 再重提 |
| 翻译 | `.zh-cn.srt` 在 **且** `require_audit()` 通过（`protocol=grounded-json-v1`、`quality_passed`、无 missing/review/hold、源与产物 sha 匹配） | 不重翻（子进程内还有 `requests/` 响应缓存与 checkpoint，不重复付费） |
| 归档 | sha1 与体积都相同 | 不复制、不产生 `.prev` |
| 整体 | 已是 `uploaded` 的条目入队前被过滤 | 不重跑、不重传 |

**续跑的边界**：入队清单由 `survey()` 现算。只要云端存在同名 SRT，该片就不在缺字幕队列里——这正是 `stage=cloud` 失败不会自愈的原因。

### 3.5 `stage=cloud` 的恢复

云端已有一份验收没过的 SRT 时，重跑**不会**修好它（它会被判成「已有字幕」）。三步：① `mvsub verify "<video|->" "<原文.ja.srt>" "<中文.srt>"` 本地复核；② 处理云端那份同名——**改名优先**（改成非 `.srt` 扩展名就不再被当成字幕，随时可回滚），确要删再删；③ 重新 `mvsub run`，该片才会重新进入缺字幕队列。

### 3.6 批次收尾

`run` 结束会重扫云端，打印「累计完成 N 部；云端仍缺字幕 M 部」，再打一行**本轮结果**：按 `BATCH_STATUS_ORDER` 固定 12 个状态计数（缺的记 0，`skipped_verify_failed` 拆 `stage`），同时进 `run.log`。单片跳过是设计行为，但整批大面积失败必须一眼看出来；有条目却 `uploaded 0` 时会额外警告。

---

## 4. 验收判据

### 4.1 三层口径

| 说法 | 含义 | 谁判定 |
|---|---|---|
| **上传完成** | `cloud.upload()` 返回 `state=true`，只说明云盘收下了这次请求 | 代码 |
| **验收通过** | 归档校验 + 本地硬判据 + 云端一致性全过 | 代码（`do_upload()` 五步） |
| **观看合格** | 漏译、复读、阅读速度、首/中/尾对得上画面 | **人**（只能抽查） |

顺序固定：**归档成功 → 本地验收 → 上传 → 云端一致 → 才删本地**。任一步失败只记状态 + 日志，**不删本地文件、绝不静默继续**。

### 4.2 硬判据（失败即 `❌`、保留本地）

| # | 判据 | 实现 / 出处 | 阈值常量 |
|---|---|---|---|
| 1 | SRT 可解析，cue 数 > 0 | `parse_srt_cues()` / `_check_srt()`：解析不出的块直接丢弃，丢光即失败 | — |
| 2 | 没有空文本 cue | `_check_srt()`；解析层特意**保留**空文本 cue，否则判据会被解析吃掉 | — |
| 3 | 每条 cue `end > start` | `_check_srt()` | — |
| 4 | 开始时间非递减 | `_check_srt()` 逐条与上一条比 | — |
| 5 | 起止都在片长内 | `probe_duration()` + `_check_srt()`；**生产路径取不到片长按失败**，只有 `verify -` 记「未做」 | `CUE_TAIL_TOLERANCE = 5.0` |
| 6 | 中文与原文条数相等 | `validate_subtitles()` | — |
| 7 | 中文与原文时间轴逐条一致 | `validate_subtitles()`：原样拷贝时间戳 | `TIMELINE_TOL_MS = 0` |
| 8 | 中文里**假名/日文残留 = 0** | `validate_subtitles()` 与 `verify_subtitles()` 都查；**删本地之前就拦** | `KANA_RE = [\u3040-\u30ff]` |
| 9 | 云端体积与本地一致 | `verify_uploaded(cid, name, expect_size)` | — |
| 10 | 云端 sha1（拿到了就比） | 返回 `sha`/`sha1` 才比；没返回就**明写「未做」**，记 `cloud_sha_checked=false` | — |
| 11 | 原文归档经 sha1 + 体积双校验 | `archive_original()` | — |

第 8 条单独强调：它原先只在事后复核里查、删本地的闸门不查，模型偶发把日文专名原样留下就成了成品。现在它与结构校验一起，是**上传与清理的前置条件**。

### 4.3 提示项与人工项

| 项 | 判据 | 常量 |
|---|---|---|
| 单字长重复 | 同一字符连续 ≥8 次 | `SINGLE_CHAR_RUN_RE = (.)\1{7,}` |
| 连续重复 | 相邻两条文本相同且长度 ≥4 字 | `CONSEC_DUP_MIN_CHARS = 4` |
| 阅读速度过高 | 只判时长 ≥1.0 s 的 cue，>17 CPS 提示、>25 CPS 必看 | `CPS_MIN_CUE_SEC = 1.0`、`CPS_WARN = 17.0`、`CPS_MUST_LOOK = 25.0` |
| 与原文完全相同 | 粗信号，专有词/数字本就该相同 | — |

这些只打印 `⚠`、**不影响退出码**：规则能把可疑处摆到眼前，但语气词天然重复，自动判死会造大量假阳性。CPS 是字符/秒、阈值本仓库自定（公开口径多为英文字符/秒，中文不能照搬），超短 cue 单列计数、不参与判定。**首 3/中 3/尾 3 条与画面对照属人工项**：结构校验只能证明「可解析、时间轴一致、无假名残留」，不能证明「每条都对」。

### 4.4 `mvsub verify`

```bash
mvsub verify "<视频路径>" "<原文.ja.srt>" "<中文.srt>"   # 有视频：顺带校片长范围
mvsub verify - "<原文.ja.srt>" "<中文.srt>"              # 只有字幕：片长范围记「未做」
```

退出码 `0` = 硬判据全过、`1` = 有硬判据失败、`2` = 参数/文件缺失。纯本地只读（不联网、不写状态、不改文件），因此**不做云端一致性**——云端核对在 `run` 里做。硬判据 = §4.2 的 #1–#8，阈值只存在于 `nosub_pipeline.py` 顶部常量：**改阈值必须改常量并在提交信息写原因**，改完用 `mvsub verify` 复核几部成品。

---

## 5. 上传事务

### 5.1 为什么同名不能裸传

- **云盘的同名上传不覆盖，会并存两份同名文件**（`Cloud115.upload()` 的注释写得很直白）。裸传的结果不是「替换」，而是目录里多一份同名 SRT；播放器加载哪一份不可控，还会顶掉人工校对过的版本。
- 接口返回 `state=true` 只代表「收下了这次请求」，不代表字节对、名字对、只有一份，**不能凭它删本地数据**。
- 所以缺字幕线**先查重**：`find_by_name()` 命中同名就 `skipped_name_exists`，告警跳过；真要替换旧字幕，走 `replace_subtitle()` 事务。

### 5.2 缺字幕的发布：`publish_missing_subtitle()`

1. **冻结身份**：视频 fid、目标名、本地 sha1、体积写进 journal；任一项与记录不符直接拒（防止发到别的视频名下）；随后重列目录确认源视频名字与 sha 未变。
2. **传到暂存名** `<stem>.mvplus-stage-<视频fid>.srt`——必须仍以 `.srt` 结尾（云盘批量改名保留原扩展名），并校验体积与 sha1 与本地一致。
3. **改名就位**后重列目录核对：正式名只能一份，且 **FID 必须还是暂存那个**（变了说明撞上别人的同名文件）；已有正式名且 FID 就是自己记录的暂存 FID → 视为已完成，不重复上传。
4. **`upload_attempted` 已置位但云端查不到暂存件** → 拒绝自动重传：上次结果未知时再传一次只会得到两份同名文件。

### 5.3 替换旧字幕：`replace_subtitle()` 事务

1. **备份旧字节**到回滚副本；元数据有 sha 就比 sha，不符即停（**回滚副本不可信时不许继续**）。
2. **暂存**为 `.mvplus-stage-<原fid>.srt` 后校验：有 sha 比 sha；**云端不给 sha 时回下载一份到 `<local>.cloudcheck` 比本地字节**，比完删掉——这是「没有 sha 也能证明字节一致」的唯一办法。
3. **旧字幕改名加「旧」**（`<stem>旧.srt`，重名时 `<stem>(fid)旧.srt`），每个改名写进 journal 的 `retained_old`，任一步都能逆序回滚；新字幕改名就位失败时也按逆序改回。
4. **收尾复核**：新名的 FID/名字/体积/sha1 全部复读，`retained_old` 仍在各自新名字上；**只有名字里带自己暂存标记的旧暂存件才允许删**（否则拒绝清理，防止误删别人或人工的文件）。
5. **journal 逐步落盘**：`staged → old_renamed → published → complete`，中断按 journal 续。

### 5.4 发布后核对清单

1. **名字**：与视频同 stem、`.srt` 结尾、目录里没有第二份同名。
2. **FID**：还是自己那次上传产生的那个（不是别人的同名文件）。
3. **体积**：云端字节数 == 本地字节数。
4. **SHA1**：云端给了就比；没给就回下载比字节，或**明确记为「未做」**（`cloud_sha_checked=false`，别读成「sha1 已核对」）。
5. **状态**：`uploaded` 里 `verify=passed`，`cues` 与 `run.log` 的验收行一致；事后复核只认 `uploaded` 里的 `name/bytes/sha1/cues/cloud_sha1/cloud_sha_checked`。

---

## 6. 原文归档

### 6.1 只留本地，不上云端

日文原文**只归档到本地**（`config.archive_dir()`，默认 `~/.mvsub/originals`；`--orig-dir` 或 `MVSUB_ARCHIVE_DIR` 可覆盖），云端只新增中文字幕。原因很直接：原文是**唯一副本**，本地留着才能随时重译、重对齐、重新验收。

### 6.2 归档判据：`archive_original()`

1. 目录不存在就建；建不出来或不可写 → **失败**（不降级成「原文只留工作目录」）；本地原文 sha1 与体积可读且非 0 字节，否则失败。
2. 目标 `<stem>.ja.srt` 已存在且 **sha1 与体积都相同** → 复用，不复制、不产生 `.prev`（日志写「内容一致」）。
3. 否则复制到 `<名>.part`（只清自己可能残留的临时件），校验 sha1 + 体积；不符就删 part，**既有归档件与本地原文都不动**。
4. 目标已存在但内容不同 → 旧版改名 `<名>.prev`（上一版 `.prev` 被覆盖）；用字符串拼接而非 `with_suffix`，否则 `X.ja.srt` 会变成 `X.ja.ja.srt.prev`。
5. part → 目标，**再复读校验一次**；通过才算归档成功。

只有「归档成功」才允许删本地原文：**本地是原文的唯一副本，归档没落定就删 = 原文永久丢失。**

### 6.3 归档目录不可用就修目录，不绕过

`run` 开跑前就检查归档目录可写，不可用**直接拒绝开跑（rc=2）**并提示用 `--orig-dir` 换目录。没有 `--no-archive` 这类开关：归档成功是「删本地原文」的唯一安全前置。

### 6.4 音轨归档与清理顺序

整片 PCM 音轨在删视频前归档（`audio_archive.archive_audio()`）：校验 16 kHz/单声道/PCM16 与帧数完整 → sha1 + 体积 → 同名且一致用硬链接（跨卷失败则校验复制）；不一致另存 `<stem>.audio.<sha1>.wav`，**绝不覆盖未知内容**，旁边写 `<stem>.audio.json` manifest。失败会让条目停在 `skipped_archive_failed`。

清理是**最后一步**，只在验收全过后：视频、原文 SRT、中文 SRT、`*.global_context.json`、`*.checkpoint`。清理失败不影响云端产物，仍记 `uploaded`，只留垃圾并打警告。反过来，**上传成功但验收没过时本地一律保留**；残留 `.ja.srt.part` 与 `.prev` 都**不要手工删**。

---

## 7. 命令速查

```bash
mvsub doctor / mvsub config        # 体检 / 打印解析后的路径

mvsub scan   --remote-cid <目录 cid>                 # 只读预检（也可 --remote-path）
mvsub run    --remote-cid <目录 cid> --budget-file <台账> --budget-cny <上限>
mvsub status --remote-cid <目录 cid>
mvsub run --remote-cid <cid> --limit 2 --budget-file <台账> --budget-cny <上限>   # 试跑；--only/--name 见 --help

mvsub extract   "<视频>" --output "<原文.ja.srt>"     # 取音 → 识别 → 对齐
mvsub translate "<原文.ja.srt>" --output "<中文.srt>" --model deepseek-flash --grounded --batch-size 16 --off-peak-only
mvsub reprocess "<plan.json>"                         # 已有字幕从视频重做（四段并行）

mvsub verify "<video|->" "<原文.ja.srt>" "<中文.srt>"  # 只读复核
mvsub cloud ls / download / upload                    # 直连调试
```

参数口径：`run` 的 `--budget-file` 与 `--budget-cny` **必填**（缺一个直接 rc=2），`--remote-cid` 与 `--remote-path` 二选一。`mvsub cloud download` 只做状态码检查 + 分块写盘，不做体积/sha1 校验、不写 `.part`——批量下载走 `mvsub run`（内部有 `.part`、`Range` 续传、体积 0.99 校验与 `ffprobe` 探时长）。

---

## 8. 收尾检查

工作目录应只剩 `state.json` 与 `run.log`；残留视频/SRT 说明该条目没走到验收通过，**保留即可**，重跑会重试。`run.log` 的 `✅/❌` 与 `state.json` 的 `status`/`reason`/`stage` 是判断「过没过」的唯一凭据，不要凭「上传返回成功」推定完成。

**维护纪律**：阈值只有两处——代码常量 + 本文表格，改一处就同步另一处并在提交信息写原因；并发只改常量，不要靠数终止符调收尾；写进文档的每个数字都要能指到常量，指不到就写「见代码常量 X」。

