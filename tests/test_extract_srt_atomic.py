"""`extract_srt._write_srt()` 原子写最小单测（2026-09-24）。

覆盖的核心承诺：**写入失败/被打断时既有的 `<视频名>.srt` 不被破坏**
（旧实现直接在目标文件上 `write_text("")` 截断重写，一个失败就把上一次的成果清空）。

跑法：
    uv run python -m pytest subtitle/tests/test_extract_srt_atomic.py -q
    # 或不装 pytest：.venv\\Scripts\\python.exe subtitle/tests/test_extract_srt_atomic.py

全部离线：不加载模型、不做转写，只测写文件这一步。
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import extract_srt as sut  # noqa: E402

OLD = "1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n\n"
NEW_SEGMENTS = [(0, 1000, "你好"), (2000, 3000, "晚安")]


class WriteSrtAtomicTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.target = self.dir / "movie.srt"

    def tearDown(self):
        self._tmp.cleanup()

    def test_success_writes_target_and_leaves_no_part(self):
        self.target.write_text(OLD, encoding="utf-8")
        sut._write_srt(self.target, NEW_SEGMENTS)
        text = self.target.read_text(encoding="utf-8")
        self.assertIn("你好", text)
        self.assertNotIn("こんにちは", text)                 # 目标已被新内容替换
        self.assertFalse((self.dir / "movie.srt.part").exists())   # 临时件已就位、无残留

    def test_partial_write_failure_keeps_existing_file(self):
        self.target.write_text(OLD, encoding="utf-8")
        real_write_text = Path.write_text

        def half_then_fail(path, data, *a, **kw):
            if path.name.endswith(".part"):
                real_write_text(path, str(data)[:5], **kw)   # 写一半（模拟磁盘满/被杀）
                raise OSError("No space left on device")
            return real_write_text(path, data, *a, **kw)

        with patch.object(Path, "write_text", half_then_fail):
            with self.assertRaises(OSError):
                sut._write_srt(self.target, NEW_SEGMENTS)
        self.assertEqual(self.target.read_text(encoding="utf-8"), OLD)   # 既有文件原样
        self.assertTrue((self.dir / "movie.srt.part").exists())          # 只留下 .part

    def test_replace_failure_keeps_existing_file(self):
        self.target.write_text(OLD, encoding="utf-8")
        with patch.object(sut.os, "replace", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                sut._write_srt(self.target, NEW_SEGMENTS)
        self.assertEqual(self.target.read_text(encoding="utf-8"), OLD)
        self.assertTrue((self.dir / "movie.srt.part").exists())

    def test_first_write_without_existing_file(self):
        sut._write_srt(self.target, NEW_SEGMENTS)
        self.assertTrue(self.target.exists())
        self.assertFalse((self.dir / "movie.srt.part").exists())


if __name__ == "__main__":
    unittest.main()
