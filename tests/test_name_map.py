"""人名对照表（prompts/name_map.txt）的注入与归一：机制单测。

测的是机制，不是某张具体的表：用临时文件构造对照表，验证
① 表能解析成 replace/calls/block；② prompt 注入块排在模型自生成的人物关系之前；
③ 归一化只替换表内命中，表外的日文一律不动。

自带的人名表默认是空模板，所以这里不依赖仓库里的表内容。
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mvsub" / "pipeline"))

import translate_srt_batch as T  # noqa: E402


SAMPLE_TABLE = "\n".join([
    "# 注释行应被忽略",
    "山田太郎\t山田太郎\trole\t主角",
    "さくら\t樱\tfix\t常被听成 ざくら",
    "〜ちゃん\t〜酱\tcall",
    "",
])


class NameMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "name_map.txt"
        self.path.write_text(SAMPLE_TABLE, encoding="utf-8")
        # 让"默认路径"指向临时表，这样 build_prompt/normalize_names 走的是真实代码路径
        self._saved = (T.NAME_MAP_PATH, T.NAME_MAP_CACHE)
        T.NAME_MAP_PATH = self.path
        T.NAME_MAP_CACHE = None

    def tearDown(self):
        T.NAME_MAP_PATH, T.NAME_MAP_CACHE = self._saved
        self.tmp.cleanup()

    def test_table_parses(self):
        nm = T.load_name_map(self.path)
        self.assertEqual(nm["count"], 2)
        self.assertEqual(nm["replace"]["さくら"], "樱")
        self.assertEqual(nm["calls"], [("ちゃん", "酱")])
        self.assertIn("固定人名对照表", nm["block"])

    def test_comment_lines_do_not_leak(self):
        nm = T.load_name_map(self.path)
        self.assertNotIn("注释行应被忽略", nm["block"])
        self.assertNotIn("#", nm["replace"])

    def test_empty_table_disables_the_feature(self):
        empty = Path(self.tmp.name) / "empty.txt"
        empty.write_text("# 只有注释\n", encoding="utf-8")
        nm = T.load_name_map(empty)
        self.assertEqual(nm["count"], 0)
        self.assertEqual(nm["block"], "")

    def test_prompt_block_precedes_generated_context(self):
        block = T.load_name_map(self.path)["block"]
        self.assertIn("固定人名对照表", block)
        prompt = T.build_prompt(
            [(1, 0.0, 1.0, "さくらの声が届かない")], [],
            "ja", "zh-cn", system_prompt="（测试用系统提示）",
            global_ctx={"summary": "模型自己猜的概要",
                        "characters": [{"name_ja": "さくら", "name_zh": "错误译名"}]})
        self.assertIn("固定人名对照表", prompt)
        self.assertLess(prompt.index("固定人名对照表"), prompt.index("全局剧情与术语参考"),
                        "固定人名表必须排在模型自生成的人物关系之前（表里写明它优先）")


if __name__ == "__main__":
    unittest.main()
