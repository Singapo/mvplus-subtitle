"""Sound omission is upstream of translation and must not remove dialogue."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from aligned_asr.utterance import make_units, non_dialogue_sound, sound_omission_indices


class SoundExtractionTests(unittest.TestCase):
    def unit(self, text, kind='sound'):
        return dict(text=text, kind=kind, start=1., end=2.)

    def test_clear_sounds_are_omitted(self):
        for text in ('はぁ、はぁ', 'ハァ、ハァ', 'ちゅっ、ちゅぱっ', 'すぅー',
                     'んぐっんぐっ', 'あぁっあぁっ', 'んんん'):
            with self.subTest(text=text):
                self.assertTrue(non_dialogue_sound(self.unit(text)))

    def test_short_responses_and_questions_are_preserved(self):
        for text in ('うん', 'うんうん', 'ううん', 'あ', 'ああ', 'ええ', 'いい',
                     'いう', 'すう', 'くう', 'くん', 'ふうん', 'はぁ？', 'ん？'):
            with self.subTest(text=text):
                self.assertFalse(non_dialogue_sound(self.unit(text)))

    def test_lexical_and_uncertain_classifications_are_not_deleted(self):
        for text in ('はぁ', 'だめだめだめ', 'お兄さん', '吸って', 'すごい'):
            self.assertFalse(non_dialogue_sound(self.unit(text, 'dialogue')))
        self.assertFalse(non_dialogue_sound(self.unit('お兄さん')))

    def test_mixed_segment_keeps_words_clocks_and_original_unit_indexes(self):
        words = [dict(text='はぁ', start=0., end=.7, probability=.9),
                 dict(text='お', start=.8, end=1., probability=.9),
                 dict(text='兄さん', start=1., end=2., probability=.9)]
        block = dict(start=10., segments=[dict(text='はぁお兄さん', words=words)])
        tag = lambda _: SimpleNamespace(words=['はぁ', 'お', '兄さん'],
                                        postags=['感動詞', '接頭辞', '名詞'])
        units, issues = make_units(block, tag)
        self.assertFalse(issues)
        original = [dict(u) for u in units]
        omitted = sound_omission_indices(units)
        self.assertEqual(omitted, {0})
        kept = [(i, u) for i, u in enumerate(units) if i not in omitted]
        self.assertEqual(kept[0][0], 1)
        self.assertEqual(kept[0][1]['text'], 'お兄さん')
        self.assertEqual((kept[0][1]['start'], kept[0][1]['end']), (10.8, 12.))
        self.assertEqual(units, original)

    def test_human_corrected_dialogue_is_not_filtered_using_wrong_asr_text(self):
        units = [self.unit('はぁ')]
        review = dict(utterance_confirmations=[dict(human_verified=True,
            source_text='はぁ', source_start=1., source_end=2., text='はい', start=1., end=2.)])
        self.assertEqual(sound_omission_indices(units, review), set())
        review['utterance_confirmations'][0]['source_start'] = 99.
        self.assertEqual(sound_omission_indices(units, review), {0})


if __name__ == '__main__':
    unittest.main()
