#!/usr/bin/env python3
"""
MVplus 字幕提取（faster-whisper 直连版）
直接调用 faster-whisper（不走 pyVideoTrans 的进程/超时封装），稳定性更高。

用法:
    python subtitle/tools/extract_srt.py "D:/movies/movie.mkv"
    python subtitle/tools/extract_srt.py "D:/movies/movie.mkv" --model large-v3-turbo --language ja --cuda

说话人分离（可选，需要 uv sync --extra diarize）:
    python subtitle/tools/extract_srt.py "D:/movies/movie.mkv" --cuda --diarize --hf-token <token>
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import sysconfig
import time
import warnings
from pathlib import Path
from asr_config import ASR_MODEL
import config

TOOLS_DIR = Path(__file__).resolve().parent           # src/mvsub/pipeline
SUBTITLE_ROOT = config.repo_root()                    # 仓库根
PROJECT_ROOT = SUBTITLE_ROOT                          # .env 所在
MODEL_ROOT = config.model_home() / "faster-whisper"
CUDA_LIB_ROOT = SUBTITLE_ROOT / "lib" / "cuda"

# Display cues must not bridge long pauses, even when Whisper merges the words.
WORD_PAUSE_SECONDS = 1.0
MAX_CUE_SECONDS = 8.0

# faster-whisper 官方模型名 -> 本地缓存目录名
FASTER_MODELS = {
    "tiny": "models--Systran--faster-whisper-tiny",
    "base": "models--Systran--faster-whisper-base",
    "small": "models--Systran--faster-whisper-small",
    "medium": "models--Systran--faster-whisper-medium",
    "large-v1": "models--Systran--faster-whisper-large-v1",
    "large-v2": "models--Systran--faster-whisper-large-v2",
    "large-v3": "models--Systran--faster-whisper-large-v3",
    "large-v3-turbo": "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo",
}


def format_time(ms: int) -> str:
    """毫秒 -> SRT 时间格式 HH:MM:SS,mmm。"""
    hours, rem = divmod(ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def load_faster_whisper():
    """Load faster-whisper from the project environment."""
    from faster_whisper import WhisperModel
    return WhisperModel


def find_model_dir(model_name: str) -> Path:
    """定位本地 faster-whisper 模型目录。"""
    if model_name in FASTER_MODELS:
        local = MODEL_ROOT / FASTER_MODELS[model_name]
        if local.exists():
            return local
    # 兜底：尝试直接作为路径
    candidate = Path(model_name)
    if candidate.exists():
        return candidate
    # 尝试 models--xxx 形式
    local = MODEL_ROOT / model_name.replace("/", "--")
    if local.exists():
        return local
    raise FileNotFoundError(
        f"找不到模型 {model_name}：本地缓存 {MODEL_ROOT} 里没有，"
        "请放入对应模型目录，或用 --model 指定本地模型路径"
    )


def _find_ffmpeg() -> str:
    """Locate ffmpeg executable or raise a clear error."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH; required for speaker diarization")
    return ffmpeg


def _extract_diarization_wav(video_path: Path, wav_path: Path) -> None:
    """Extract a 16 kHz mono WAV for pyannote."""
    ffmpeg = _find_ffmpeg()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def _load_diarization_pipeline(
    model_name: str,
    hf_token: str | None,
    device: str,
    cache_dir: Path | None,
):
    """Lazy-load pyannote.audio diarization pipeline."""
    for pat in (
        r"\s*torchcodec is not installed correctly.*",
        r"In 2\.9, this function.*torchaudio\.load_with_torchcodec.*",
        r"std\(\): degrees of freedom is <= 0.*",
    ):
        warnings.filterwarnings("ignore", message=pat, category=UserWarning)
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Diarization requires pyannote.audio. Run: uv sync --extra diarize"
        ) from exc

    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    pipeline_kwargs = {}
    if cache_dir is not None:
        pipeline_kwargs["cache_dir"] = str(cache_dir)

    # pyannote.audio 4.x uses `token`, 3.x used `use_auth_token`.
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token, **pipeline_kwargs)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token, **pipeline_kwargs)
    pipeline.to(torch.device(device))
    return pipeline


def _assign_speaker(
    start: float,
    end: float,
    turns: list[tuple[float, float, str]],
    unknown: str = "?",
    min_overlap: float = 0.2,
) -> str:
    """Pick the speaker with the largest temporal overlap."""
    best_label = unknown
    best_overlap = 0.0
    for t0, t1, label in turns:
        overlap = max(0.0, min(end, t1) - max(start, t0))
        if overlap > best_overlap and overlap >= min_overlap:
            best_overlap = overlap
            best_label = label
    return best_label


def _canonicalize_speaker(label: str, mapping: dict[str, str]) -> str:
    """Map pyannote labels (SPEAKER_00, ...) to A, B, C, ..."""
    if label not in mapping:
        mapping[label] = chr(ord("A") + len(mapping))
    return mapping[label]


def _extract_turns_from_output(output, min_overlap: float = 0.2) -> list[tuple[float, float, str]]:
    """Handle pyannote.audio 3.x Annotation and 4.x PipelineOutput."""
    turns = []
    # pyannote.audio 4.x returns a dataclass with .speaker_diarization
    if hasattr(output, "speaker_diarization"):
        for turn, speaker in output.speaker_diarization:
            if (turn.end - turn.start) >= min_overlap:
                turns.append((turn.start, turn.end, str(speaker)))
        return turns
    # pyannote.audio 3.x returns an Annotation
    for turn, _, label in output.itertracks(yield_label=True):
        if (turn.end - turn.start) >= min_overlap:
            turns.append((turn.start, turn.end, label))
    return turns


def _write_srt(output_srt: Path, segments: list[tuple[int, int, str]]) -> None:
    """Write (start_ms, end_ms, text) segments to an SRT file.

    **原子写（2026-09-24）**：先写完整内容到 `<名字>.part`，再 `os.replace()` 就位。
    这样"提取中途失败/被杀"只会留一个 `.part`，**既有的 `<视频名>.srt` 不会被破坏**
    （旧实现直接在目标文件上截断重写，一个失败就把上一次的成果清空）。
    """
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (start_ms, end_ms, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_time(start_ms)} --> {format_time(end_ms)}")
        lines.append(text)
        lines.append("")
    tmp = output_srt.with_name(output_srt.name + ".part")
    tmp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    os.replace(tmp, output_srt)


def _word_cues(segment) -> list[tuple[int, int, str]]:
    """Use aligned words, preserving text while splitting pauses and long cues.

    Missing or impossible alignment is an extraction error, never a reason to
    fall back to the segment timestamps that caused early subtitles.
    Zero-duration tokens can occur next to a valid word; keep their text in
    that group, but reject a group with no positive duration.
    """
    words = [word for word in (segment.words or []) if word.word.strip()]
    if not words:
        raise ValueError(f"Missing word alignment at {segment.start:.3f}s")
    compact = lambda text: "".join(text.split())
    if compact("".join(word.word for word in words)) != compact(segment.text):
        raise ValueError(f"Word alignment does not cover text at {segment.start:.3f}s")
    groups = []
    group = []
    start = end = None
    previous_start = -1.0
    for word in words:
        a, b = word.start, word.end
        if (not math.isfinite(a) or not math.isfinite(b) or a < 0 or b < a
                or a < previous_start):
            raise ValueError(f"Invalid word alignment at {segment.start:.3f}s")
        if b - a > MAX_CUE_SECONDS:
            raise ValueError(f"Word spans {b - a:.2f}s at {a:.3f}s; alignment needs review")
        if group and (a - end >= WORD_PAUSE_SECONDS or b - start > MAX_CUE_SECONDS):
            groups.append((start, end, "".join(group).strip()))
            group = []
        if not group:
            start, end = a, b
        else:
            end = max(end, b)
        group.append(word.word)
        previous_start = a
    groups.append((start, end, "".join(group).strip()))
    cues = []
    for a, b, text in groups:
        start_ms, end_ms = round(a * 1000), round(b * 1000)
        if end_ms <= start_ms:
            raise ValueError(f"Zero-duration aligned cue at {a:.3f}s; alignment needs review")
        cues.append((start_ms, end_ms, text))
    return cues


def _resolve_diarize_device(diarize_device: str | None, use_cuda: bool) -> str:
    """Pick a default device for pyannote."""
    if diarize_device is not None:
        return diarize_device
    if use_cuda:
        # Whisper is already on CUDA; keep pyannote on CPU to save VRAM by default.
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _run_diarization(
    video_path: Path,
    output_srt: Path,
    annotated_srt: Path,
    kept_segments: list[tuple[int, int, str]],
    *,
    diarize_model: str,
    hf_token: str | None,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
    diarize_device: str | None,
    diarize_cache_dir: Path | None,
    keep_wav: bool,
    speaker_unknown_label: str,
    min_speaker_overlap: float,
    use_cuda: bool,
) -> None:
    """Run pyannote diarization and write an annotated SRT."""
    device = _resolve_diarize_device(diarize_device, use_cuda)

    wav_path = output_srt.with_suffix(".diarize.wav")
    print(f"[diarize] extracting WAV: {wav_path}")
    _extract_diarization_wav(video_path, wav_path)

    print(f"[diarize] loading pipeline: {diarize_model} on {device}")
    pipeline = _load_diarization_pipeline(
        model_name=diarize_model,
        hf_token=hf_token,
        device=device,
        cache_dir=diarize_cache_dir,
    )

    diarize_kwargs = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

    print("[diarize] loading audio into memory...")
    try:
        import torch
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    except Exception as exc:
        raise RuntimeError(f"Failed to load audio with torchaudio: {exc}") from exc

    print("[diarize] running diarization...")
    output = pipeline(audio_input, **diarize_kwargs)
    turns = _extract_turns_from_output(output, min_overlap=min_speaker_overlap)

    mapping: dict[str, str] = {}
    annotated_segments = []
    for start_ms, end_ms, text in kept_segments:
        start_s = start_ms / 1000.0
        end_s = end_ms / 1000.0
        raw_label = _assign_speaker(
            start_s, end_s, turns,
            unknown=speaker_unknown_label,
            min_overlap=min_speaker_overlap,
        )
        label = _canonicalize_speaker(raw_label, mapping)
        annotated_segments.append((start_ms, end_ms, f"[{label}] {text}"))

    _write_srt(annotated_srt, annotated_segments)
    print(f"[diarize] annotated SRT -> {annotated_srt}")

    if not keep_wav:
        wav_path.unlink(missing_ok=True)


def extract(
    video_path: Path,
    model_name: str,
    language: str,
    use_cuda: bool,
    output_srt: Path,
    compute_type: str = "float16",
    beam_size: int = 5,
    max_duration: float = 29.9,
    no_speech_threshold: float = 0.5,
    min_logprob: float = -1.0,
    min_logprob_short: float = -2.5,
    skip_duplicates: bool = True,
    *,
    diarize: bool = False,
    diarize_model: str = "pyannote/speaker-diarization-community-1",
    hf_token: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    diarize_device: str | None = None,
    diarize_cache_dir: Path | None = None,
    keep_wav: bool = False,
    speaker_unknown_label: str = "?",
    min_speaker_overlap: float = 0.2,
    repair_alignment=None,
):
    WhisperModel = load_faster_whisper()
    model_dir = find_model_dir(model_name)
    print(f"[model] {model_dir}")

    device = "cuda" if use_cuda else "cpu"
    compute_type = compute_type or ("float16" if use_cuda else "int8")

    print("[step 1/2] 加载模型...")
    model = WhisperModel(
        str(model_dir),
        device=device,
        device_index=0,
        compute_type=compute_type,
    )

    print("[step 2/2] 识别字幕并按词级时间对齐...")
    output_srt.parent.mkdir(parents=True, exist_ok=True)
    # 不再在这里 `write_text("")` 清空目标文件：写入改走 _write_srt() 的 `.part` + os.replace 原子替换，
    # 中途失败只留 `.part`，既有的 `<视频名>.srt` 保持不动（2026-09-24）。

    segments, info = model.transcribe(
        str(video_path),
        language=language if language and language != "auto" else None,
        task="transcribe",
        beam_size=beam_size,
        # 关闭内置 VAD：faster-whisper 的 VAD 对成人影片的轻声/气声丢失严重
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=no_speech_threshold,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    )

    detected = info.language if info else language
    print(f"[info] 检测到语言: {detected}")

    kept_segments = []
    last_log = 0
    prev_text = None
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        # 后处理过滤：
        # 1) 连续重复文本（无声/背景音乐幻觉）
        # 2) 模型判定为非语音
        # 3) 拆句后置信度过低；短于 2.5s 的轻声/气声允许更低置信度
        if skip_duplicates and text == prev_text:
            continue
        if segment.no_speech_prob > no_speech_threshold:
            continue
        # A segment below both confidence thresholds cannot produce any cue,
        # regardless of word durations. Filter it before validating alignment so
        # already-rejected noise cannot abort the whole subtitle extraction.
        if segment.avg_logprob < min(min_logprob, min_logprob_short):
            prev_text = text
            continue
        # Split before checking duration: a long segment may contain several
        # valid short utterances separated by silence.
        try:cues = _word_cues(segment)
        except ValueError:
            if repair_alignment is None:raise
            cues = repair_alignment(model,segment)
        for start_ms, end_ms, cue_text in cues:
            dur = (end_ms - start_ms) / 1000
            effective_min_logprob = min_logprob_short if dur < 2.5 else min_logprob
            if segment.avg_logprob < effective_min_logprob:
                continue
            if (end_ms - start_ms) / 1000 >= max_duration:
                raise ValueError(f"Aligned cue exceeds max_duration at {start_ms / 1000:.3f}s")
            if kept_segments and start_ms < kept_segments[-1][0]:
                raise ValueError(f"Aligned cues move backwards at {start_ms / 1000:.3f}s")
            kept_segments.append((start_ms, end_ms, cue_text))
        prev_text = text

        # 每 60 秒报告进度；全部完成前不发布字幕。
        now = time.time()
        if kept_segments and (now - last_log) > 60:
            print(f"[progress] {len(kept_segments)} 条字幕, 已处理至 {format_time(kept_segments[-1][1])}")
            # Publish only after the entire extraction passes alignment checks.
            # A later failure must not replace an existing SRT with a prefix.
            last_log = now

    if not kept_segments:
        raise ValueError("No usable aligned speech; existing subtitle preserved")
    _write_srt(output_srt, kept_segments)
    print(f"[ok] 共 {len(kept_segments)} 条字幕 -> {output_srt}")

    if diarize:
        annotated_srt = output_srt.with_suffix(".speakers.srt")
        # When output_srt already ends with .speakers.srt (unlikely), avoid collision.
        if annotated_srt == output_srt:
            annotated_srt = output_srt.parent / f"{output_srt.stem}.speakers.srt"
        _run_diarization(
            video_path=video_path,
            output_srt=output_srt,
            annotated_srt=annotated_srt,
            kept_segments=kept_segments,
            diarize_model=diarize_model,
            hf_token=hf_token,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            diarize_device=diarize_device,
            diarize_cache_dir=diarize_cache_dir,
            keep_wav=keep_wav,
            speaker_unknown_label=speaker_unknown_label,
            min_speaker_overlap=min_speaker_overlap,
            use_cuda=use_cuda,
        )

    return output_srt


def main():
    parser = argparse.ArgumentParser(description="MVplus 字幕提取（faster-whisper 直连）")
    parser.add_argument("video", help="输入视频文件的绝对路径")
    parser.add_argument(
        "--model",
        default=str(ASR_MODEL),
        help="Whisper 模型名或路径（默认本地日语760M；完整对齐使用 extract_aligned_srt.py）",
    )
    parser.add_argument(
        "--language",
        default="ja",
        help="源语言代码（默认 ja，设为 auto 则自动检测）",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="输出目录（默认视频所在目录）",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="使用 GPU (CUDA) 加速",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="计算类型（默认：cuda 用 float16，cpu 用 int8）",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="束搜索大小（默认 5）",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=29.9,
        help="拆句后字幕持续时间上限（默认 29.9；超限报错，不静默丢弃台词）",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.7,
        help="跳过 no_speech_prob 超过该值的片段（默认 0.7；片头轻声对话可再调低）",
    )
    parser.add_argument(
        "--min-logprob",
        type=float,
        default=-1.0,
        help="长片段（>=2.5s）最低平均 logprob（默认 -1.0）",
    )
    parser.add_argument(
        "--min-logprob-short",
        type=float,
        default=-2.5,
        help="短片段（<2.5s）最低平均 logprob（默认 -2.5，保留轻声/气声）",
    )
    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="保留连续重复文本（默认去重，成人内容可开启）",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="启用说话人分离（默认关闭）",
    )
    parser.add_argument(
        "--diarize-model",
        default="pyannote/speaker-diarization-community-1",
        help="pyannote 说话人分离模型（默认 pyannote/speaker-diarization-community-1）",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace access token；未提供时尝试 HF_TOKEN 环境变量",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="指定说话人数量（可选）",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="最小说话人数量（可选）",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="最大说话人数量（可选）",
    )
    parser.add_argument(
        "--diarize-device",
        default=None,
        help="pyannote 运行设备（默认：Whisper 用 CUDA 时为 cpu，否则自动检测）",
    )
    parser.add_argument(
        "--diarize-cache-dir",
        default=None,
        help="pyannote 模型缓存目录（默认 subtitle/models/pyannote，可被 HF_HOME/TORCH_HOME 覆盖）",
    )
    parser.add_argument(
        "--keep-wav",
        action="store_true",
        help="保留中间 16kHz WAV 文件（调试用）",
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"[error] 视频不存在: {video_path}", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir).resolve() if args.outdir else video_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    output_srt = outdir / f"{video_path.stem}.srt"
    if output_srt.exists():
        output_srt = outdir / f"{video_path.stem}.auto.srt"
        print(f"[warn] {video_path.stem}.srt already exists, writing to {output_srt.name}")

    diarize_cache_dir = None
    if args.diarize:
        if args.diarize_cache_dir:
            diarize_cache_dir = Path(args.diarize_cache_dir).resolve()
        elif not (os.environ.get("HF_HOME") or os.environ.get("TORCH_HOME")):
            diarize_cache_dir = SUBTITLE_ROOT / "models" / "pyannote"
            diarize_cache_dir.mkdir(parents=True, exist_ok=True)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Ensure PATH includes cuDNN/cuBLAS DLLs on Windows.
    # 两个来源：当前 venv 里装的 nvidia-* 包，以及项目自带的 subtitle/lib/cuda/nvidia
    # （从旧 pyvideotrans 环境迁出，`uv sync` 不会删除它）。
    # 注意：Python 3.8+ 起 Windows 的 DLL 搜索不再看 PATH，必须用 os.add_dll_directory，
    # 否则会报 `Library cublas64_12.dll is not found or cannot be loaded`。
    if args.cuda:
        site_packages = Path(sysconfig.get_path("purelib"))
        cuda_bins: list[str] = []
        for root in (site_packages / "nvidia", CUDA_LIB_ROOT / "nvidia"):
            if not root.is_dir():
                continue
            for pkg in root.iterdir():
                bindir = pkg / "bin"
                if bindir.is_dir():
                    cuda_bins.append(str(bindir))
        os.environ["PATH"] = os.pathsep.join(cuda_bins + [os.environ.get("PATH", "")])
        if hasattr(os, "add_dll_directory"):
            for d in cuda_bins:
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
        print(f"[cuda] DLL 目录已注入 {len(cuda_bins)} 个: "
              f"{[Path(d).parent.parent.name for d in cuda_bins]}")

    try:
        extract(
            video_path=video_path,
            model_name=args.model,
            language=args.language,
            use_cuda=args.cuda,
            output_srt=output_srt,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            max_duration=args.max_duration,
            no_speech_threshold=args.no_speech_threshold,
            min_logprob=args.min_logprob,
            min_logprob_short=args.min_logprob_short,
            skip_duplicates=not args.keep_duplicates,
            diarize=args.diarize,
            diarize_model=args.diarize_model,
            hf_token=hf_token,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            diarize_device=args.diarize_device,
            diarize_cache_dir=diarize_cache_dir,
            keep_wav=args.keep_wav,
        )
    except Exception as e:
        print(f"[error] 提取失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
