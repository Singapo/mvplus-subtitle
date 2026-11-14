import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import translate_srt_batch as sut


class SubtitleAlignmentTests(unittest.TestCase):
    batch = [('7', '00:00:01,000', '00:00:02,000', 'こんにちは'),
             ('11', '00:00:02,000', '00:00:03,000', 'おやすみなさい')]

    def test_sparse_ids_and_reordered_response(self):
        self.assertEqual(sut.parse_batch_response('11. 晚安\n7. 你好', self.batch), ['你好', '晚安'])
        self.assertEqual(sut.parse_batch_response('7：你好\n11: 晚安', self.batch), ['你好', '晚安'])

    def test_rejects_shift_causes(self):
        for raw in ['1. 你好\n2. 晚安', '7. 你好', '7. 你好\n11. ',
                    '7. 你好\n7. 晚安', '你好\n晚安', '7. 你好\n11. 晚安\n12. 多余']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                sut.parse_batch_response(raw, self.batch)

    def test_whole_file_uses_same_strict_id_rules(self):
        with patch.object(sut, 'call_ollama', return_value='11. 晚安\n7. 你好'):
            result = sut.translate_whole_file(self.batch, 'test', 'local', 'ja', 'zh-cn')
        self.assertEqual([x[3] for x in result], ['你好', '晚安'])
        with patch.object(sut, 'call_ollama', return_value='1. 你好\n2. 晚安'):
            with self.assertRaises(ValueError):
                sut.translate_whole_file(self.batch, 'test', 'local', 'ja', 'zh-cn')

    def test_global_sample_preserves_final_entry(self):
        entries = [(str(i), '00:00:00,000', '00:00:01,000', str(i)) for i in range(1000)]
        sampled = sut._sample_entries_for_global_context(entries, max_entries=800)
        self.assertEqual(len(sampled), 800)
        self.assertEqual(sampled[0][0], '0')
        self.assertEqual(sampled[-1][0], '999')

    def test_poisoned_cache_not_used_or_updated(self):
        with tempfile.TemporaryDirectory() as d:
            tm = sut.TranslationMemory(Path(d) / 'tm.json')
            tm.set('おやすみなさい', 'zh-cn', 'test', '继续别停')
            before = dict(tm.data)
            with patch.object(sut, 'call_ollama', return_value='7. 你好\n11. 晚安') as call:
                translated, hits = sut.translate_batch(self.batch, [], tm, 'test', 'local', 'ja', 'zh-cn')
            self.assertEqual(translated, ['你好', '晚安'])
            self.assertEqual(hits, 0)
            self.assertEqual(before, tm.data)
            self.assertIn('11. おやすみなさい', call.call_args.args[0])

    def test_parallel_failure_does_not_save_source_as_translation(self):
        with tempfile.TemporaryDirectory() as d:
            cp = Path(d) / 'checkpoint.json'
            tm = sut.TranslationMemory(Path(d) / 'tm.json')
            with patch.object(sut, 'call_ollama', side_effect=ValueError('bad response')):
                with self.assertRaises(RuntimeError):
                    sut.adaptive_batch_translate(self.batch, tm, 'test', 'local', 'ja', 'zh-cn',
                                                 1, 2, 2, cp, resume=False, timeout=1)
            self.assertFalse(cp.exists())


if __name__ == '__main__':
    unittest.main()
