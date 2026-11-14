"""补字幕线验收（H1 原文归档校验 / H2 删本地前的字幕与云端一致性校验）最小单测。

跑法：
    uv run python -m pytest subtitle/tests/test_nosub_pipeline_verify.py -q
    # 或不装 pytest：.venv\\Scripts\\python.exe subtitle/tests/test_nosub_pipeline_verify.py

全部离线：不联网、不跑 ffprobe（用 patch 固定片长）、不碰 115（用 FakeCloud 替身）。
"""
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import nosub_pipeline as ns  # noqa: E402


def make_srt(*cues, first_index=1) -> str:
    blocks = [f"{first_index + i}\n{start} --> {end}\n{text}\n"
              for i, (start, end, text) in enumerate(cues)]
    return "\n".join(blocks)


JA = make_srt(("00:00:01,000", "00:00:02,500", "こんにちは"),
              ("00:00:03,000", "00:00:04,000", "おやすみなさい"))
ZH = make_srt(("00:00:01,000", "00:00:02,500", "你好"),
              ("00:00:03,000", "00:00:04,000", "晚安"))


class FakeCloud:
    """最小 115 替身：记录上传字节，`verify_uploaded()` 按字节复算体积/sha1。"""

    def __init__(self, corrupt_size: bool = False, corrupt_sha: bool = False):
        self.entries: list[dict] = []
        self.uploads: list[str] = []
        self.corrupt_size = corrupt_size
        self.corrupt_sha = corrupt_sha

    def invalidate(self, cid) -> None:
        pass

    def find_by_name(self, cid, name: str) -> list[dict]:
        return [e for e in self.entries if e["n"] == name]

    def upload(self, local, cid, name: str) -> bool:
        data = Path(local).read_bytes()
        self.uploads.append(name)
        self.entries.append({
            "fid": str(len(self.entries) + 1),
            "n": name,
            "s": len(data) + (7 if self.corrupt_size else 0),
            "sha": "0" * 40 if self.corrupt_sha else hashlib.sha1(data).hexdigest(),
        })
        return True

    def verify_uploaded(self, cid, name: str, expect_size: int):
        hits = self.find_by_name(cid, name)
        return any(int(it.get("s") or -1) == int(expect_size) for it in hits), hits


class WorkspaceMixin:
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.video = self.work / "IPVR-008.mp4"
        self.video.write_bytes(b"\x00" * 4096)
        self.ja = self.work / "IPVR-008.srt"
        self.ja.write_text(JA, encoding="utf-8")
        self.zh = self.work / "IPVR-008.zh-cn.srt"
        self.zh.write_text(ZH, encoding="utf-8")
        self.archive = self.root / "archive"
        self.state = ns.State(self.root / "state.json")

    def tearDown(self):
        self._tmp.cleanup()

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in self.archive.iterdir()) if self.archive.exists() else []


class ValidateSubtitlesTests(WorkspaceMixin, unittest.TestCase):
    def test_accepts_matching_pair(self):
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, note = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertTrue(ok, note)
        self.assertIn("cue 2", note)
        self.assertIn("时间轴一致", note)

    def test_duration_unknown_with_video_is_a_failure(self):
        """生产路径（有视频）取不到片长 = 失败，不是「未做」（2026-09-24 收口径）。"""
        with patch.object(ns, "probe_duration", return_value=None):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("取不到片长", why)

    def test_duration_skipped_only_when_no_video(self):
        """只有显式不提供视频（`verify -`）时才允许把该项记为「未做」。"""
        with patch.object(ns, "probe_duration", return_value=None):
            ok, note = ns.validate_subtitles(self.ja, self.zh, None)
        self.assertTrue(ok, note)
        self.assertIn("未做", note)

    def test_rejects_empty_text_cue(self):
        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,500", "你好"),
                                    ("00:00:03,000", "00:00:04,000", "   ")), encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("文本为空", why)

    def test_rejects_non_monotonic_timeline(self):
        self.zh.write_text(make_srt(("00:00:03,000", "00:00:04,000", "你好"),
                                    ("00:00:01,000", "00:00:02,500", "晚安")), encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("非单调递增", why)

    def test_rejects_reversed_cue(self):
        self.zh.write_text(make_srt(("00:00:02,500", "00:00:01,000", "你好"),
                                    ("00:00:03,000", "00:00:04,000", "晚安")), encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("结束时间 <= 开始时间", why)

    def test_rejects_cue_beyond_video_duration(self):
        # 时间戳放到 10 分钟处、片长只声明 60s：超出 60+5=65s 容差 → 必须失败
        late = make_srt(("00:10:00,000", "00:10:02,500", "こんにちは"))
        self.ja.write_text(late, encoding="utf-8")
        self.zh.write_text(make_srt(("00:10:00,000", "00:10:02,500", "你好")), encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=60.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("超出视频时长", why)

    def test_rejects_timeline_drift_between_ja_and_zh(self):
        self.zh.write_text(make_srt(("00:00:01,500", "00:00:02,500", "你好"),
                                    ("00:00:03,000", "00:00:04,000", "晚安")), encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("时间轴与原/译不一致", why)

    def test_rejects_cue_count_mismatch(self):
        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,500", "你好")), encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("条数不一致", why)

    def test_rejects_unparsable_srt(self):
        self.zh.write_text("这不是 SRT\n", encoding="utf-8")
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ok, why = ns.validate_subtitles(self.ja, self.zh, self.video)
        self.assertFalse(ok)
        self.assertIn("解析不出任何 cue", why)


class ArchiveOriginalTests(WorkspaceMixin, unittest.TestCase):
    def test_archives_with_sha1_and_size_check(self):
        target, note = ns.archive_original(self.ja, self.archive)
        self.assertIsNotNone(target, note)
        self.assertEqual(target.read_bytes(), self.ja.read_bytes())
        self.assertIn(hashlib.sha1(self.ja.read_bytes()).hexdigest(), note)
        self.assertIn("体积=", note)
        self.assertEqual([p.name for p in self.archive.iterdir()], ["IPVR-008.ja.srt"])

    def test_identical_archive_is_reused_without_prev(self):
        ns.archive_original(self.ja, self.archive)
        target, note = ns.archive_original(self.ja, self.archive)
        self.assertIsNotNone(target, note)
        self.assertIn("复用既有归档件", note)
        self.assertEqual([p.name for p in self.archive.iterdir()], ["IPVR-008.ja.srt"])

    def test_changed_content_keeps_old_version_as_prev(self):
        ns.archive_original(self.ja, self.archive)
        old = (self.archive / "IPVR-008.ja.srt").read_bytes()
        self.ja.write_text(JA + make_srt(("00:00:05,000", "00:00:06,000", "ありがとう")),
                           encoding="utf-8")
        target, note = ns.archive_original(self.ja, self.archive)
        self.assertIsNotNone(target, note)
        self.assertEqual(target.read_bytes(), self.ja.read_bytes())
        self.assertEqual((self.archive / "IPVR-008.ja.srt.prev").read_bytes(), old)
        self.assertNotIn("IPVR-008.ja.srt.part", self.leftovers())

    def test_bad_copy_fails_and_source_is_untouched(self):
        before = self.ja.read_bytes()

        def truncated_copy(src, dst, *a, **kw):
            Path(dst).write_bytes(b"partial")

        with patch.object(ns.shutil, "copy2", side_effect=truncated_copy):
            target, note = ns.archive_original(self.ja, self.archive)
        self.assertIsNone(target)
        self.assertIn("不一致", note)
        self.assertEqual(self.ja.read_bytes(), before)          # 本地原文未被改动
        self.assertFalse((self.archive / "IPVR-008.ja.srt").exists())
        self.assertEqual(self.leftovers(), [])                  # 临时件已清掉

    def test_existing_archive_is_kept_when_new_copy_fails(self):
        ns.archive_original(self.ja, self.archive)
        good = (self.archive / "IPVR-008.ja.srt").read_bytes()
        self.ja.write_text(JA + make_srt(("00:00:07,000", "00:00:08,000", "はい")),
                           encoding="utf-8")

        def truncated_copy(src, dst, *a, **kw):
            Path(dst).write_bytes(b"partial")

        with patch.object(ns.shutil, "copy2", side_effect=truncated_copy):
            target, note = ns.archive_original(self.ja, self.archive)
        self.assertIsNone(target)
        self.assertEqual((self.archive / "IPVR-008.ja.srt").read_bytes(), good)
        self.assertFalse((self.archive / "IPVR-008.ja.srt.prev").exists())

    def test_missing_source_and_bad_target_dir_are_explicit_failures(self):
        target, note = ns.archive_original(self.work / "nope.srt", self.archive)
        self.assertIsNone(target)
        self.assertIn("本地原文不存在", note)

        as_file = self.root / "not_a_dir"
        as_file.write_text("x", encoding="utf-8")
        target, note = ns.archive_original(self.ja, as_file)
        self.assertIsNone(target)
        self.assertTrue("不是目录" in note or "归档目录不可用" in note, note)

        target, note = ns.archive_original(self.ja, None)
        self.assertIsNone(target)
        self.assertIn("未启用本地归档目录", note)


class DoUploadTests(WorkspaceMixin, unittest.TestCase):
    item = {"stem": "IPVR-008", "n": "IPVR-008.mp4", "pc": "pc1", "s": 4096}

    def run_upload(self, cloud, archive=None):
        with patch.object(ns, "probe_duration", return_value=3600.0):
            ns.do_upload(self.item, self.work, self.state, cloud, "cid1",
                         archive if archive is not None else self.archive)

    def test_success_verifies_then_cleans_local(self):
        cloud = FakeCloud()
        self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "uploaded")
        self.assertEqual(rec["verify"], "passed")
        self.assertEqual(rec["cues"], 2)
        self.assertEqual(cloud.uploads, ["IPVR-008.srt"])
        self.assertFalse(self.video.exists())
        self.assertFalse(self.ja.exists())
        self.assertFalse(self.zh.exists())
        self.assertTrue((self.archive / "IPVR-008.ja.srt").exists())

    def test_archive_failure_keeps_everything_and_skips_upload(self):
        cloud = FakeCloud()
        as_file = self.root / "blocked"
        as_file.write_text("x", encoding="utf-8")
        self.run_upload(cloud, archive=as_file)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_archive_failed")
        self.assertEqual(cloud.uploads, [])
        self.assertTrue(self.video.exists())
        self.assertTrue(self.ja.exists())
        self.assertTrue(self.zh.exists())

    def test_local_verify_failure_blocks_upload_and_keeps_local(self):
        cloud = FakeCloud()
        self.zh.write_text(make_srt(("00:00:09,000", "00:00:10,000", "你好"),
                                    ("00:00:11,000", "00:00:12,000", "晚安")), encoding="utf-8")
        self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_verify_failed")
        self.assertEqual(rec["stage"], "local")
        self.assertEqual(cloud.uploads, [])
        self.assertTrue(self.video.exists())
        self.assertTrue(self.zh.exists())

    def test_cloud_size_mismatch_keeps_local(self):
        cloud = FakeCloud(corrupt_size=True)
        self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_verify_failed")
        self.assertEqual(rec["reason"], "cloud_size_mismatch")
        self.assertEqual(rec["stage"], "cloud")
        self.assertTrue(self.video.exists())
        self.assertTrue(self.ja.exists())
        self.assertTrue(self.zh.exists())

    def test_cloud_sha1_mismatch_keeps_local(self):
        cloud = FakeCloud(corrupt_sha=True)
        self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_verify_failed")
        self.assertEqual(rec["reason"], "cloud_sha1_mismatch")
        self.assertTrue(self.zh.exists())

    def test_unreadable_local_artifact_keeps_local(self):
        cloud = FakeCloud()
        real = ns.sha1_of

        def flaky(path, chunk=ns.HASH_CHUNK):
            if Path(path).name.endswith(".zh-cn.srt"):     # 只让"读中文产物"这一步炸
                raise OSError("boom")
            return real(path, chunk)

        with patch.object(ns, "sha1_of", side_effect=flaky):
            self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_verify_failed")
        self.assertEqual(rec["reason"], "local_read_failed")
        self.assertTrue(self.video.exists())
        self.assertTrue(self.ja.exists())
        self.assertTrue(self.zh.exists())

    def test_missing_translation_is_explicit(self):
        cloud = FakeCloud()
        self.zh.unlink()
        self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_missing_translation")
        self.assertEqual(cloud.uploads, [])
        self.assertTrue(self.ja.exists())

    def test_existing_remote_name_is_not_overwritten(self):
        cloud = FakeCloud()
        cloud.entries.append({"fid": "9", "n": "IPVR-008.srt", "s": 10, "sha": "a" * 40})
        self.run_upload(cloud)
        rec = self.state.get("IPVR-008")
        self.assertEqual(rec["status"], "skipped_name_exists")
        self.assertEqual(cloud.uploads, [])
        self.assertTrue(self.zh.exists())


class VerifySubtitlesTests(WorkspaceMixin, unittest.TestCase):
    """`verify` 子命令的判据（SOP §3.1 步骤⑥）：硬判据失败 / 提示项只是提示。"""

    def verify(self, video=None, orig=None, zh=None):
        with patch.object(ns, "probe_duration", return_value=3600.0 if video else None):
            return ns.verify_subtitles(video, orig or self.ja, zh or self.zh)

    def test_clean_pair_passes_with_only_manual_hint(self):
        r = self.verify(video=self.video)
        self.assertTrue(r["ok"], r["failures"])
        self.assertEqual(r["failures"], [])
        self.assertTrue(any("人工抽查项" in h for h in r["hints"]))

    def test_kana_residue_is_hard_failure_with_detail(self):
        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,500", "こんにちは"),
                                    ("00:00:03,000", "00:00:04,000", "晚安")), encoding="utf-8")
        r = self.verify(video=self.video)
        self.assertFalse(r["ok"])
        self.assertTrue(any("假名" in f for f in r["failures"]), r["failures"])
        self.assertTrue(any("假名残留" in d for d in r["details"]))

    def test_single_char_run_is_hint_not_failure(self):
        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,500", "你好"),
                                    ("00:00:03,000", "00:00:04,000", "澪" * 30)), encoding="utf-8")
        r = self.verify(video=self.video)
        self.assertTrue(r["ok"], r["failures"])          # 提示级：不影响退出码
        self.assertEqual(r["stats"]["single_char_runs"], 1)
        self.assertTrue(any("单字长重复" in h for h in r["hints"]))

    def test_high_cps_is_hint_and_short_cues_are_counted(self):
        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,500", "你好"),
                                    ("00:00:03,000", "00:00:04,000", "晚安"),
                                    ("00:00:04,000", "00:00:04,200", "超短")), encoding="utf-8")
        self.ja.write_text(self.zh.read_text(encoding="utf-8"), encoding="utf-8")
        r = self.verify(video=self.video)
        self.assertTrue(r["ok"], r["failures"])
        self.assertEqual(r["stats"]["short_cues"], 1)
        self.assertEqual(r["stats"]["cps_judged"], 2)

        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,000", "字" * 40),
                                    ("00:00:03,000", "00:00:04,000", "晚安")), encoding="utf-8")
        self.ja.write_text(self.zh.read_text(encoding="utf-8"), encoding="utf-8")
        r = self.verify(video=self.video)
        self.assertTrue(r["ok"], r["failures"])
        self.assertEqual(r["stats"]["cps_over_warn"], 1)
        self.assertEqual(r["stats"]["cps_over_must_look"], 1)

    def test_identical_to_original_cue_is_reported(self):
        self.ja.write_text(make_srt(("00:00:01,000", "00:00:02,500", "おやすみなさい"),
                                    ("00:00:03,000", "00:00:04,000", "おやすみなさい")), encoding="utf-8")
        self.zh.write_text(make_srt(("00:00:01,000", "00:00:02,500", "晚安"),
                                    ("00:00:03,000", "00:00:04,000", "おやすみなさい")), encoding="utf-8")
        r = self.verify(video=self.video)
        self.assertEqual(r["stats"]["identical_to_orig"], 1)
        self.assertTrue(any("完全相同的 cue" in h for h in r["hints"]))

    def test_without_video_duration_check_is_marked_not_done(self):
        r = self.verify(video=None)
        self.assertTrue(r["ok"])
        self.assertIn("未做", r["summary"])


class CmdVerifyTests(WorkspaceMixin, unittest.TestCase):
    def run_cli(self, *argv):
        args = ns.build_parser().parse_args(["verify", *argv])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ns.cmd_verify(args)
        return code, out.getvalue(), err.getvalue()

    def test_exit_0_on_pass_and_1_on_failure(self):
        code, out, _ = self.run_cli("-", str(self.ja), str(self.zh))
        self.assertEqual(code, 0, out)
        self.assertIn("✅ 验收通过", out)
        self.assertIn("云端一致性不适用", out)

        self.zh.write_text(make_srt(("00:00:09,000", "00:00:10,000", "你好"),
                                    ("00:00:11,000", "00:00:12,000", "晚安")), encoding="utf-8")
        code, out, _ = self.run_cli("-", str(self.ja), str(self.zh))
        self.assertEqual(code, 1, out)
        self.assertIn("❌ 验收失败", out)

    def test_exit_2_on_missing_file(self):
        code, _, err = self.run_cli("-", str(self.work / "nope.ja.srt"), str(self.zh))
        self.assertEqual(code, 2)
        self.assertIn("找不到原文字幕", err)

    def test_exit_2_on_missing_video_path(self):
        code, _, err = self.run_cli(str(self.work / "nope.mp4"), str(self.ja), str(self.zh))
        self.assertEqual(code, 2)
        self.assertIn("找不到视频", err)

    def test_read_only_no_side_effects(self):
        before = {p.name: p.read_bytes() for p in sorted(self.work.iterdir())}
        self.run_cli("-", str(self.ja), str(self.zh))
        after = {p.name: p.read_bytes() for p in sorted(self.work.iterdir())}
        self.assertEqual(before, after)


class BatchSummaryTests(unittest.TestCase):
    def test_counts_extra_and_stages(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            state = ns.State(root / "state.json")
            state.set("A", "uploaded")
            state.set("B", "skipped_verify_failed", stage="local")
            state.set("C", "skipped_verify_failed", stage="cloud")
            state.set("D", "skipped_archive_failed")
            state.set("E", "translated")
            state.set("F", "weird_status")
            counts, extra, stages = ns.batch_status_counts(state, ["A", "B", "C", "D", "E", "F", "G"])
            self.assertEqual(counts["uploaded"], 1)
            self.assertEqual(counts["skipped_verify_failed"], 2)
            self.assertEqual(counts["skipped_archive_failed"], 1)
            self.assertEqual(counts["skipped_download"], 0)
            self.assertEqual(stages, {"local": 1, "cloud": 1})
            self.assertEqual(extra, {"unfinished": 1, "other": 2})   # F=未知状态，G=未开始
            line = ns.format_batch_summary(counts, extra, stages)
            self.assertTrue(line.startswith("本轮结果：uploaded 1 / skipped_verify_failed 2（stage=local 1、stage=cloud 1）"))
            self.assertIn("skipped_archive_failed 1", line)
            self.assertIn("skipped_missing_translation 0", line)      # 固定顺序里缺的记 0，不省略
            self.assertIn("未完成 1", line)
            self.assertIn("其他 2", line)




class TestSubtitleDecoding(unittest.TestCase):
    def test_legacy_and_bom_sources_preserve_chinese_and_clock(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'legacy.srt'
            expected = None
            for encoding in ('utf-8-sig', 'utf-16', 'gb18030'):
                path.write_text(ZH, encoding=encoding)
                actual = ns.parse_srt_cues(path)
                self.assertEqual([r[3] for r in actual], ['你好', '晚安'])
                self.assertEqual(actual[0][0], '1')
                if expected is None: expected = actual
                self.assertEqual(actual, expected)

    def test_invalid_bytes_fail_without_replacement_characters(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'invalid.srt'
            path.write_bytes(b'1\n00:00:01,000 --> 00:00:02,000\n\xff')
            with self.assertRaises(ValueError): ns.parse_srt_cues(path)


if __name__ == '__main__':
    unittest.main()
