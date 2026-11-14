from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# 脚本目录现为 subtitle/tools，直接把它塞进 sys.path（与同目录其它测试一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))

import subtitle_pipeline_core as core  # noqa: E402


SRT = """7
00:00:00,000 --> 00:00:01,000
こんにちは

11
00:00:01,000 --> 00:00:02,000
世界
"""

def test_declared_korean_source_preserves_japanese_mixed_language_guard():
    entries=[{'text':'안녕하세요 반갑습니다 오늘 만나서 기쁩니다'}]
    assert core._source_issues(entries)[0]['type']=='mixed_hangul_source'
    assert core._source_issues(entries,source_language='ko')==[]
    assert core._source_issues([{'text':'\ufffd\ufffd\ufffd'}],source_language='ko')


def write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def manifest(path: Path, original: Path | str, translation: Path | str, *, name: str = "film【DSflash].srt") -> Path:
    payload = {
        "pairs": [{
            "translation": {"fid": "target-1", "cid": "folder-1", "n": name, "sha": "ABCDEF", "local": str(translation)},
            "original": {"fid": "source-1", "n": "film.ja.srt", "local": str(original)},
        }]
    }
    return write(path, json.dumps(payload, ensure_ascii=False))


def test_parse_and_render_preserve_sparse_ids_and_timing(tmp_path: Path) -> None:
    source = write(tmp_path / "film.ja.srt", SRT)
    plan = manifest(tmp_path / "pair_manifest.json", source, tmp_path / "not-present.srt")
    job = core.prepare_jobs(plan, tmp_path / "work", [], ["DSflash"], chunk_size=1)[0]
    assert [chunk["id"] for chunk in job["chunks"]] == ["0001", "0002"]
    for chunk in job["chunks"]:
        request_path = Path(chunk["request_path"])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        assert "target_entries" in request
        response = {
            "source_hash": request["source_hash"],
            "translations": [{"id": item["id"], "text": "中文" + item["id"]} for item in request["target_entries"]],
            "uncertain_ids": [],
            "review_required": False,
            "reviewed": True,
        }
        core.save_json(request_path.with_name(request_path.name.replace(".request.json", ".result.json")), response)
        chunk["status"] = "reviewed"
    core.save_json(Path(job["job_path"]), job)
    rendered = core.render_job(Path(job["job_path"]))
    assert rendered["status"] == "ready"
    output = Path(rendered["output_path"]).read_text(encoding="utf-8")
    assert "7\n00:00:00,000 --> 00:00:01,000\n中文7" in output
    assert "11\n00:00:01,000 --> 00:00:02,000\n中文11" in output


def test_existing_local_target_must_exactly_align(tmp_path: Path) -> None:
    source = write(tmp_path / "film.ja.srt", SRT)
    wrong_target = write(tmp_path / "film.zh.srt", SRT.replace("00:00:01,000 --> 00:00:02,000", "00:00:01,010 --> 00:00:02,000"))
    plan = manifest(tmp_path / "pair_manifest.json", source, wrong_target)
    jobs = core.prepare_jobs(plan, tmp_path / "work", [], ["DSflash"])
    assert jobs[0]["status"] == "alignment_review"


def test_original_fid_remap_and_ambiguous_stem_matching(tmp_path: Path) -> None:
    root = tmp_path / "roots"
    remapped = write(root / "source-1__film.ja.srt", SRT)
    old_original = tmp_path / "gone" / "source-1__film.ja.srt"
    plan = manifest(tmp_path / "pair_manifest.json", old_original, tmp_path / "none.srt", name="film【DSflash.srt")
    job = core.prepare_jobs(plan, tmp_path / "work", [root], [], chunk_size=200)[0]
    assert Path(job["source_path"]) == remapped.resolve()

    # No FID-prefixed candidate: different bytes with the same normalized stem
    # must not be silently chosen.
    write(root / "a" / "film.ja.srt", SRT)
    write(root / "b" / "film.ja.srt", SRT.replace("世界", "別の内容"))
    payload = json.loads(plan.read_text(encoding="utf-8"))
    payload["pairs"][0]["translation"]["fid"] = "target-2"
    payload["pairs"][0]["original"] = {"fid": "not-found", "n": "film.ja.srt", "local": str(tmp_path / "missing.srt")}
    plan.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    jobs = core.prepare_jobs(plan, tmp_path / "other-work", [root], [])
    assert jobs[0]["status"] == "ambiguous_source"


def test_changed_source_or_config_never_reuses_checkpoint(tmp_path: Path) -> None:
    source = write(tmp_path / "film.ja.srt", SRT)
    plan = manifest(tmp_path / "pair_manifest.json", source, tmp_path / "none.srt")
    core.prepare_jobs(plan, tmp_path / "work", [], ["DSflash"], chunk_size=200)
    write(source, SRT.replace("世界", "地球"))
    with pytest.raises(ValueError, match="stale job"):
        core.prepare_jobs(plan, tmp_path / "work", [], ["DSflash"], chunk_size=200)


def test_validate_result_rejects_injection_and_missing_source(tmp_path: Path) -> None:
    request = {
        "source_hash": "hash",
        "target_entries": [{"id": "7", "start": "00:00:00,000", "end": "00:00:01,000", "text": "こんにちは"}],
    }
    injected = {
        "source_hash": "hash",
        "translations": [{"id": "7", "text": "好\n999\n00:00:00,000 --> 00:00:01,000"}],
        "uncertain_ids": [],
        "review_required": False,
    }
    assert any("injection" in error for error in core.validate_result(request, injected))
    plan = manifest(tmp_path / "pair_manifest.json", tmp_path / "gone.ja.srt", tmp_path / "none.srt")
    jobs = core.prepare_jobs(plan, tmp_path / "work", [], ["DSflash"])
    assert jobs[0]["status"] == "missing_source"


def test_matching_preserves_cut_and_part_labels():
    assert core._normal_stem("12__film [4K]【DSflash].srt") == core._normal_stem("film [4K].ja.srt")
    assert core._normal_stem("film [4K].ja.srt") != core._normal_stem("film [60fps].ja.srt")
    assert core._normal_stem("film.part1.ja.srt") != core._normal_stem("film.part2.ja.srt")


def test_missing_original_inventory_not_silently_skipped(tmp_path):
    data = {"pairs": [], "missing_original": [{"fid": "77", "cid": "9", "n": "unknown【DSflash].srt"}]}
    path = write(tmp_path / "manifest.json", json.dumps(data))
    jobs = core.prepare_jobs(path, tmp_path / "work", [], ["DSflash"])
    assert len(jobs) == 1 and jobs[0]["status"] == "missing_source"


def test_target_hash_change_invalidates_resume(tmp_path):
    source = write(tmp_path / "film.ja.srt", SRT)
    path = manifest(tmp_path / "manifest.json", source, tmp_path / "none.srt")
    core.prepare_jobs(path, tmp_path / "work", [], ["DSflash"])
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pairs"][0]["translation"]["sha"] = "NEW_HASH"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="stale job"):
        core.prepare_jobs(path, tmp_path / "work", [], ["DSflash"])
