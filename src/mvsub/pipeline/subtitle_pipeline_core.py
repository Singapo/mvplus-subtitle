"""Deterministic, local-only core for the DSflash subtitle pipeline.

This module deliberately has no model, network, or command-line dependency.
The runner owns model calls and writes ``*.result.json`` checkpoints; this
module owns the immutable source snapshot, request contracts, and safe SRT
rendering.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


PIPELINE_VERSION = "subtitle-pipeline-core-v1"
_TIMELINE_RE = re.compile(
    r"^\s*(\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*"
    r"(\d{1,3}:\d{2}:\d{2}[,.]\d{3})\s*$"
)
_STAMP_RE = re.compile(r"^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})$")
_KANA_RE = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9f]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_HTML_RE = re.compile(r"<[^>]*>")
_BRACKET_TAG_RE = re.compile(r"[【\[][^】\]]*[】\]]")


def _absolute(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_srt(path: Path) -> str:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-16", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"cannot decode SRT {path}: {'; '.join(errors)}")


def _stamp_ms(value: str) -> int:
    match = _STAMP_RE.match(value)
    if not match:
        raise ValueError(f"invalid timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    if minutes > 59 or seconds > 59:
        raise ValueError(f"invalid timestamp: {value!r}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_srt(path: Path) -> list[dict[str, Any]]:
    """Strictly parse an SRT while retaining source IDs, times, and text.

    Empty subtitle text is retained: it is a source-review concern, not a
    license to shift IDs or timings.  Structural defects raise ``ValueError``.
    """
    path = _absolute(path)
    raw = _read_srt(path).replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n", raw.strip()) if raw.strip() else []
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            raise ValueError(f"{path}: block {position} lacks ID or timeline")
        subtitle_id = lines[0].strip()
        timeline = _TIMELINE_RE.match(lines[1])
        if not subtitle_id or not timeline:
            raise ValueError(f"{path}: block {position} has invalid ID or timeline")
        if subtitle_id in seen:
            raise ValueError(f"{path}: duplicate subtitle ID {subtitle_id!r}")
        seen.add(subtitle_id)
        start, end = timeline.groups()
        start_ms, end_ms = _stamp_ms(start), _stamp_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"{path}: invalid timeline for subtitle ID {subtitle_id!r}")
        entries.append({
            "id": subtitle_id,
            "start": start.replace(".", ","),
            "end": end.replace(".", ","),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": "\n".join(lines[2:]).strip(),
        })
    if not entries:
        raise ValueError(f"{path}: no subtitle entries")
    return entries


def save_json(path: Path, payload: Any) -> None:
    """Atomically persist a JSON checkpoint for the local runner."""
    path = _absolute(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


# Private name retained for concise use inside this module.
_write_json = save_json


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _clean_name(value: Any) -> str:
    return _HTML_RE.sub("", str(value or ""))


def _normal_stem(value: Any) -> str:
    name = _clean_name(value).replace("\\", "/").rsplit("/", 1)[-1]
    if name.casefold().endswith(".srt"):
        name = name[:-4]
    # A checked-in mirror is named ``<fid>__<remote name>``.
    name = re.sub(r"^\d+__", "", name)
    # Preserve cut/part/resolution tags: [4K] and [60fps] can identify a different cut.
    name = re.sub(r"[【\[]?dsflash[】\]]?", "", name, flags=re.I)
    name = re.sub(r"\.(?:ja|jp|jpn)$", "", name, flags=re.I)
    return re.sub(r"\s+", "", name).casefold()


def _is_selected(record: dict[str, Any], keywords: list[str]) -> bool:
    if not keywords:
        keywords = ["DSflash"]
    name = _clean_name(record.get("n") or record.get("ns"))
    if not name.casefold().endswith(".srt") or any(tag in name.casefold() for tag in ("mvplus-stage", "mvplus-backup", "gpt直译旧版")):
        return False
    compact = re.sub(r"\s+", "", name).casefold()
    normalized_keywords = [
        re.sub(r"[【】\[\]]", "", str(keyword)).strip().casefold()
        for keyword in keywords
    ]
    return any(keyword and keyword in compact for keyword in normalized_keywords)


def _dedupe_candidates(candidates: list[Path], label: str) -> Path:
    unique = sorted({_absolute(item) for item in candidates})
    if not unique:
        raise FileNotFoundError(label)
    if len(unique) == 1:
        return unique[0]
    hashes = {_sha256(item) for item in unique}
    if len(hashes) == 1:
        return unique[0]
    names = ", ".join(str(item) for item in unique)
    raise ValueError(f"ambiguous local original match ({label}): {names}")


def _existing_local(record: dict[str, Any], manifest_dir: Path, role: str) -> list[Path]:
    raw = record.get("local")
    if not raw:
        return []
    requested = Path(str(raw))
    candidates = [requested] if requested.is_file() else []
    # Old download roots are intentionally remapped only into the manifest's
    # checked-in baseline, never guessed from a remote catalogue.
    fallback_dirs = [manifest_dir]
    if role == "original":
        fallback_dirs.append(manifest_dir / "originals")
    for directory in fallback_dirs:
        candidate = directory / requested.name
        if candidate.is_file():
            candidates.append(candidate)
    return candidates


def _resolve_original(record: dict[str, Any], manifest_dir: Path, roots: list[Path], index=None) -> Path:
    fid = str(record.get("fid") or "")
    direct = _existing_local(record, manifest_dir, "original")
    if direct:
        return _dedupe_candidates(direct, f"manifest original {fid or record.get('n')}")

    fid_matches: list[Path] = []
    stem_matches: list[Path] = []
    desired_stem = _normal_stem(record.get("n"))
    candidates = index if index is not None else [p for root in roots if root.is_dir() for p in root.rglob("*.srt")]
    for candidate in candidates:
        if not re.search(r"\.(ja|jp|jpn)\.srt$", candidate.name, re.I):
            continue
        if fid and candidate.name.startswith(fid + "__"):
            fid_matches.append(candidate)
        elif _normal_stem(candidate.name) == desired_stem:
            stem_matches.append(candidate)
    if fid_matches:
        return _dedupe_candidates(fid_matches, f"fid {fid}")
    return _dedupe_candidates(stem_matches, f"exact normalized stem {desired_stem!r}")


def _resolve_target(record: dict[str, Any], manifest_dir: Path) -> Path | None:
    matches = _existing_local(record, manifest_dir, "translation")
    return _dedupe_candidates(matches, "manifest translation") if matches else None


def _check_baseline(source: list[dict[str, Any]], target: list[dict[str, Any]], target_path: Path) -> None:
    if len(source) != len(target):
        raise ValueError(f"local translation/source entry count mismatch: {target_path}")
    for source_entry, target_entry in zip(source, target):
        if source_entry["id"] != target_entry["id"]:
            raise ValueError(f"local translation/source ID mismatch: {target_path}")
        if (source_entry["start"], source_entry["end"]) != (target_entry["start"], target_entry["end"]):
            raise ValueError(f"local translation/source timing mismatch: {target_path}")


def _source_issues(entries: list[dict[str, Any]], source_language: str = 'ja') -> list[dict[str, Any]]:
    text = "".join(entry["text"] for entry in entries)
    hangul = len(_HANGUL_RE.findall(text))
    printable = max(1, len(re.sub(r"\s", "", text)))
    controls = sum(ord(char) < 32 and char not in "\n\t" for char in text)
    # Require a meaningful amount or a large share.  A stray label fragment is
    # deliberately not enough to hold a whole film for review.
    issues: list[dict[str, Any]] = []
    if source_language != 'ko' and hangul >= 8 and (hangul >= 20 or hangul / printable >= 0.12):
        issues.append({"type": "mixed_hangul_source", "count": hangul})
    if controls >= 3:
        issues.append({"type": "control_character_garbage", "count": controls})
    replacement = text.count("\ufffd")
    if replacement >= 3 or (replacement and replacement / printable >= 0.08):
        issues.append({"type": "garbled_source", "count": replacement})
    return issues


def _route(entries):
    texts = [e["text"] for e in entries]
    long_lines = sum(len(t) > 45 for t in texts)
    foreign_fragments = sum(bool(_HANGUL_RE.search(t) or re.search(r"[A-Za-z]{4,}", t)) for t in texts)
    if long_lines >= max(2, len(texts) // 10) or foreign_fragments >= 2:
        return "terra", "long_dialogue_or_asr_fragments"
    return "luna", "ordinary_dialogue"


def _identity(source_hash: str, config: dict[str, Any]) -> str:
    payload = json.dumps({"source_hash": source_hash, "config": config}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_jobs(
    manifest_path: Path,
    work_dir: Path,
    source_roots: list[Path],
    keywords: list[str],
    chunk_size: int = 200,
    context_lines: int = 16,
    source_language: str = 'ja',
) -> list[dict[str, Any]]:
    """Create or resume selected DSflash jobs without scanning remotely."""
    if chunk_size < 1 or context_lines < 0:
        raise ValueError("chunk_size must be >= 1 and context_lines must be >= 0")
    if source_language not in ('ja','ko'):
        raise ValueError('Unsupported source language')
    manifest_path, work_dir = _absolute(manifest_path), _absolute(work_dir)
    data = _load_json(manifest_path)
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("manifest must contain a 'pairs' list")
    pairs = list(pairs) + [{"translation": item, "original": None} for item in data.get("missing_original", [])]
    roots = list(dict.fromkeys([manifest_path.parent / "originals", *map(_absolute, source_roots)]))
    source_index = [p for root in roots if root.is_dir() for p in root.rglob("*.srt")]
    config = {
        "pipeline_version": PIPELINE_VERSION,
        "chunk_size": chunk_size,
        "context_lines": context_lines,
        "keywords": sorted(str(item).casefold() for item in (keywords or ["DSflash"])),
    }
    if source_language != 'ja':
        config['source_language'] = source_language
    jobs: list[dict[str, Any]] = []
    seen_fids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        translation = pair.get("translation") if isinstance(pair.get("translation"), dict) else {}
        original = pair.get("original") if isinstance(pair.get("original"), dict) else {}
        if not translation or not _is_selected(translation, keywords):
            continue
        fid = str(translation.get("fid") or "")
        if not fid:
            raise ValueError("selected manifest pair has no translation fid")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", fid):
            raise ValueError("unsafe translation fid")
        if fid in seen_fids:
            raise ValueError(f"duplicate selected translation fid: {fid}")
        seen_fids.add(fid)
        job_dir = work_dir / "jobs" / fid
        job_path = job_dir / "job.json"
        remote = {"fid": fid, "cid": str(translation.get("cid") or ""),
                  "remote_name": str(translation.get("n") or ""),
                  "target_sha1": str(translation.get("sha") or "")}
        try:
            if not original:
                original = {"n": re.sub(r"[【\[]?dsflash[】\]]?", "", remote["remote_name"], flags=re.I).removesuffix(".srt") + ".ja.srt"}
            source_path = _resolve_original(original, manifest_path.parent, roots, source_index)
            source_entries = parse_srt(source_path)
            target_path = _resolve_target(translation, manifest_path.parent)
            if target_path is not None:
                _check_baseline(source_entries, parse_srt(target_path), target_path)
            source_hash = _sha256(source_path)
        except (FileNotFoundError, ValueError) as exc:
            reason = str(exc)
            state = "missing_source" if isinstance(exc, FileNotFoundError) else "ambiguous_source" if "ambiguous" in reason else "alignment_review" if "mismatch" in reason else "source_review"
            failed = {**remote, "job_path": str(job_path), "status": state,
                      "source_issues": [{"type": state, "reason": reason}], "chunks": []}
            if job_path.exists() and _load_json(job_path).get("chunks"):
                raise ValueError(f"existing source changed: {job_path}; checkpoints preserved") from exc
            _write_json(job_path, failed)
            jobs.append(failed)
            continue
        identity = _identity(source_hash, {**config, "remote": remote})
        if job_path.is_file():
            existing = _load_json(job_path)
            if existing.get("identity") == identity:
                jobs.append(existing)
                continue
            if existing.get("chunks"):
                raise ValueError(
                f"stale job {job_path}: source or configuration changed; "
                "previous checkpoints were preserved (choose a new work directory)"
                )
        issues = _source_issues(source_entries, source_language)
        snapshot = job_dir / "source.srt"
        job_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, snapshot)
        chunks: list[dict[str, Any]] = []
        for offset in range(0, len(source_entries), chunk_size):
            chunk_entries = source_entries[offset:offset + chunk_size]
            chunk_id = f"{len(chunks) + 1:04d}"
            request_path = job_dir / "chunks" / f"{chunk_id}.request.json"
            request = {
                "film_key": fid,
                "chunk_id": chunk_id,
                "source_hash": source_hash,
                "target_entries": [{key: entry[key] for key in ("id", "start", "end", "text")} for entry in chunk_entries],
                "context_before": [entry["text"] for entry in source_entries[max(0, offset - context_lines):offset]],
                "context_after": [entry["text"] for entry in source_entries[offset + len(chunk_entries):offset + len(chunk_entries) + context_lines]],
            }
            _write_json(request_path, request)
            route, route_reason = _route(chunk_entries)
            chunks.append({
                "id": chunk_id,
                "request_path": str(request_path),
                "route": route,
                "route_reason": route_reason,
                "status": "pending",
                "source_hash": source_hash,
            })
        job = {
            "fid": fid,
            "cid": str(translation.get("cid") or ""),
            "remote_name": str(translation.get("n") or ""),
            "priority": 0 if "dsflash" in str(translation.get("n") or "").casefold() else 1,
            "original_sha256": source_hash,
            "target_sha1": str(translation.get("sha") or ""),
            "source_path": str(source_path),
            "source_snapshot": str(snapshot),
            "job_path": str(job_path),
            "identity": identity,
            "config": config,
            "status": "source_review" if issues else "pending",
            "source_issues": issues,
            "chunks": chunks,
        }
        _write_json(job_path, job)
        jobs.append(job)
    return jobs


def validate_result(request: dict[str, Any] | Path, payload: dict[str, Any]) -> list[str]:
    """Return all contract errors; an empty list means a safe checkpoint."""
    if isinstance(request, Path):
        request = _load_json(_absolute(request))
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be an object"]
    if payload.get("source_hash") != request.get("source_hash"):
        errors.append("source_hash mismatch")
    # ``target`` was used in an early local prototype.  Keeping it readable
    # costs nothing, while all new requests use the unambiguous public key.
    target = request.get("target_entries", request.get("target"))
    translations = payload.get("translations")
    if not isinstance(target, list) or not isinstance(translations, list):
        return errors + ["target and translations must be lists"]
    expected_ids = [str(item.get("id")) for item in target if isinstance(item, dict)]
    actual_ids: list[str] = []
    for item in translations:
        if not isinstance(item, dict):
            errors.append("translation item must be an object")
            continue
        identifier, text = item.get("id"), item.get("text")
        if not isinstance(identifier, (str, int)):
            errors.append("translation ID must be a string")
            continue
        actual_ids.append(str(identifier))
        if not isinstance(text, str) or not text.strip():
            errors.append(f"empty translation for ID {identifier}")
            continue
        if "\n" in text or "\r" in text or any(ord(char) < 32 for char in text):
            errors.append(f"control or multiline injection for ID {identifier}")
        if _KANA_RE.search(text):
            errors.append(f"Japanese kana remains for ID {identifier}")
        if _HANGUL_RE.search(text):
            errors.append(f"Korean remains for ID {identifier}")
    if len(set(actual_ids)) != len(actual_ids):
        errors.append("duplicate translation IDs")
    if actual_ids != expected_ids:
        errors.append("translation IDs must exactly match target IDs in order")
    uncertain = payload.get("uncertain_ids")
    if not isinstance(uncertain, list) or any(not isinstance(item, (str, int)) for item in uncertain):
        errors.append("uncertain_ids must be a list of IDs")
    elif len({str(item) for item in uncertain}) != len(uncertain) or not set(map(str, uncertain)).issubset(set(expected_ids)):
        errors.append("uncertain_ids must be unique target IDs")
    if not isinstance(payload.get("review_required"), bool):
        errors.append("review_required must be boolean")
    markers = [str(item.get("id")) for item in translations if isinstance(item, dict) and "[原文待核]" in str(item.get("text", ""))]
    if isinstance(uncertain, list) and not set(markers).issubset(set(map(str, uncertain))):
        errors.append("unreported uncertainty marker")
    if "reviewed" in payload and not isinstance(payload["reviewed"], bool):
        errors.append("reviewed must be boolean")
    if "context_notes" in payload and not (isinstance(payload["context_notes"], list) and all(isinstance(item, str) for item in payload["context_notes"])):
        errors.append("context_notes must be a list of strings")
    return errors


def _result_path(request_path: Path) -> Path:
    return request_path.with_name(request_path.name.replace(".request.json", ".result.json"))


def render_job(job_path: Path) -> dict[str, Any]:
    """Render a reviewed job to ``output.srt``; never bless draft output."""
    job_path = _absolute(job_path)
    job = _load_json(job_path)
    if job.get("source_issues"):
        job["status"] = "source_review"
        _write_json(job_path, job)
        return job
    if _sha256(Path(job["source_snapshot"])) != job["original_sha256"]:
        raise ValueError("source snapshot hash mismatch")
    source = parse_srt(Path(job["source_snapshot"]))
    source_lookup = {e["id"]: e for e in source}
    all_translations: dict[str, str] = {}
    all_results_present = True
    has_draft = False
    for chunk in job.get("chunks", []):
        request_path = _absolute(chunk["request_path"])
        result_path = _result_path(request_path)
        if not result_path.is_file():
            all_results_present = False
            continue
        request, result = _load_json(request_path), _load_json(result_path)
        if request.get("source_hash") != job["original_sha256"]:
            raise ValueError("request source hash mismatch")
        for entry in request.get("target_entries", []):
            expected = source_lookup.get(entry.get("id"), {})
            if any(entry.get(key) != expected.get(key) for key in ("id", "start", "end", "text")):
                raise ValueError("request disagrees with source snapshot")
        errors = validate_result(request, result)
        if errors:
            raise ValueError(f"invalid checkpoint {result_path}: {'; '.join(errors)}")
        if result.get("uncertain_ids") or result.get("review_required") or any("[原文待核]" in str(x.get("text")) for x in result["translations"]):
            job["status"] = "source_review"
            job["review_issues"] = [{"type": "uncertain_translation", "chunk": chunk.get("id")}]
            _write_json(job_path, job)
            return job
        for item in result["translations"]:
            if str(item["id"]) in all_translations:
                raise ValueError("duplicate ID across chunks")
            all_translations[str(item["id"])] = item["text"]
        if result.get("reviewed") is not True or chunk.get("status") != "reviewed":
            has_draft = True
    if (not all_results_present) or has_draft or any(chunk.get("status") != "reviewed" for chunk in job.get("chunks", [])):
        job["status"] = "ready_for_review" if all_results_present and all_translations else "translated" if all_translations else "pending"
        _write_json(job_path, job)
        return job
    source = parse_srt(Path(job["source_snapshot"]))
    if [entry["id"] for entry in source] != list(all_translations):
        raise ValueError("reviewed translations do not cover the immutable source ID sequence")
    output_path = job_path.parent / "output.srt"
    rendered = "\n\n".join(
        f"{entry['id']}\n{entry['start']} --> {entry['end']}\n{all_translations[entry['id']]}"
        for entry in source
    ) + "\n"
    output_path.write_text(rendered, encoding="utf-8")
    job["output_path"] = str(output_path)
    job["status"] = "ready"
    _write_json(job_path, job)
    return job
