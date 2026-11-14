from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from aligned_asr.clock_order import reconcile


class NeighborClockTests(unittest.TestCase):
    def test_weak_zero_vowel_is_fragment_but_clear_prefix_or_positive_sound_is_not(self):
        from aligned_asr.clock_order import zero_phonetic_fragment
        unit=dict(text='お',start=10.,end=10.,kind='sound',confidence=.2168)
        self.assertTrue(zero_phonetic_fragment(unit))
        self.assertFalse(zero_phonetic_fragment(dict(unit,kind='dialogue')))
        self.assertFalse(zero_phonetic_fragment(dict(unit,confidence=.9)))
        self.assertFalse(zero_phonetic_fragment(dict(unit,end=10.2)))
        self.assertFalse(zero_phonetic_fragment(dict(unit,text='おはよう')))

    def row(self,key,text,start,end,kind='dialogue',clock='local_qwen'):
        return dict(utterance=key,text=text,start=start,end=end,kind=kind,clock=clock)

    def source(self,text,start,end,kind='dialogue',segment=0):
        return dict(text=text,start=start,end=end,kind=kind,segment=segment,words=[])

    def test_adjacent_lines_use_joint_clock_without_sorting_or_rewriting(self):
        rows=[self.row('a','こっち?',1.56,1.64),self.row('b','どっち?',1.52,2.64)]
        sources=dict(a=self.source('こっち?',1.16,1.52),b=self.source('どっち?',1.64,2.4))
        fixed,omitted,repairs=reconcile(rows,sources,lambda *args:([self.row('a','こっち?',1.2,1.5)],[self.row('b','どっち?',1.6,2.4)]))
        self.assertEqual([w['text'] for w in fixed],['こっち?','どっち?'])
        self.assertTrue(all(w['clock']=='joint_local_qwen' for w in fixed))
        self.assertFalse(omitted);self.assertEqual(len(repairs),1)

    def test_unlocated_sound_prefix_does_not_erase_following_dialogue(self):
        rows=[self.row('a','あぁ、',1.82,2.,'sound','independently_classified_native_sound_event'),self.row('b','悪魔ちゃん',1.7,3.38)]
        sources=dict(a=self.source('あぁ、',1.7,1.7,'sound'),b=self.source('悪魔ちゃん',2.,3.18))
        fixed,omitted,repairs=reconcile(rows,sources,lambda *args:None)
        self.assertEqual(fixed,[rows[1]]);self.assertEqual(omitted[0]['text'],'あぁ、')
        self.assertEqual(omitted[0]['reason'],'ambiguous_zero_sound_before_dialogue')

    def test_located_sound_and_failed_joint_alignment_still_block(self):
        for clock,kind,end in [('native_sound','sound',2.),('local_qwen','dialogue',1.7),('human_confirmed','dialogue',1.7)]:
            rows=[self.row('a','はい',1.8,2.,kind,clock),self.row('b','そう',1.7,2.4)]
            sources=dict(a=self.source('はい',1.,end,kind),b=self.source('そう',1.7,2.4))
            fixed,omitted,repairs=reconcile(rows,sources,lambda *args:None)
            self.assertEqual(fixed,rows);self.assertFalse(omitted);self.assertFalse(repairs)

    def test_joint_alignment_cannot_change_words_or_override_human_confirmation(self):
        rows=[self.row('a','はい',1.8,2.),self.row('b','そう',1.7,2.4)]
        sources=dict(a=self.source('はい',1.,1.5),b=self.source('そう',1.7,2.4))
        candidate=lambda *args:([self.row('a','違う',1.,1.5)],[rows[1]])
        self.assertEqual(reconcile(rows,sources,candidate)[0],rows)
        rows[0]['clock']='human_confirmed'
        def forbidden(*args):raise AssertionError('Human timing must not be overridden')
        self.assertEqual(reconcile(rows,sources,forbidden)[0],rows)

