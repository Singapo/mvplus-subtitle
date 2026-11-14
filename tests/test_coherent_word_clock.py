from pathlib import Path
import hashlib
import sys
import tempfile
import json
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from aligned_asr.timing import coherent_words, timing_problems
from reprocess_subtitles import require_current_timing, TIMING_POLICY, pipeline_revision, review_digest


class CoherentClockTests(unittest.TestCase):
    def test_zero_suffix_uses_native_position_without_spanning_block_origin(self):
        block={'start':145.,'end':175.,'segments':[{'words':[
            {'text':'おはよう','start':16.24,'end':19.58},
            {'text':'ございます','start':19.58,'end':19.58}]}]}
        aligned=[{'text':'おはよう','start_time':0.,'end_time':0.},
                 {'text':'ございます','start_time':0.,'end_time':0.}]
        result,note=coherent_words(aligned,block)
        self.assertEqual(note['clock'],'native_block')
        self.assertAlmostEqual(result[0]['start'],161.24)
        self.assertAlmostEqual(result[1]['start'],164.58)
        self.assertEqual(result[1]['start'],result[1]['end'])

    def test_verb_stem_and_suffix_use_one_clock(self):
        block={'start':3625.,'end':3655.,'segments':[{'words':[
            {'text':'舐め','start':22.74,'end':23.1},
            {'text':'て','start':23.1,'end':23.24},
            {'text':'ほしい','start':23.24,'end':23.6}]}]}
        aligned=[{'text':'舐め','start_time':24.,'end_time':24.},
                 {'text':'て','start_time':25.28,'end_time':25.44},
                 {'text':'ほしい','start_time':25.44,'end_time':25.44}]
        result,note=coherent_words(aligned,block)
        self.assertEqual(note['clock'],'native_block')
        self.assertFalse([p for p in timing_problems(result,3625.,3655.) if p['type']=='backward_word'])
        self.assertAlmostEqual(result[-1]['end'],3648.6)

    def test_good_independent_alignment_is_preserved(self):
        block={'start':10.,'end':40.,'segments':[]}
        result,note=coherent_words([{'text':'こんにちは','start_time':1.,'end_time':2.}],block)
        self.assertEqual(note['clock'],'qwen')
        self.assertEqual(result,[{'text':'こんにちは','start':11.,'end':12.}])

    def test_text_mismatch_cannot_silently_patch_times(self):
        block={'start':0.,'end':30.,'segments':[{'words':[{'text':'別の文','start':1.,'end':2.}]}]}
        _,note=coherent_words([{'text':'こんにちは','start_time':0.,'end_time':0.}],block)
        self.assertEqual(note['clock'],'qwen_unresolved')

    def test_invalid_native_clock_cannot_replace_independent_clock(self):
        block={'start':0.,'end':30.,'segments':[{'words':[{'text':'はい','start':20.,'end':19.}]}]}
        _,note=coherent_words([{'text':'はい','start_time':0.,'end_time':0.}],block)
        self.assertEqual(note['reason'],'invalid_native_clock')

    def test_aligner_token_crossing_utterances_is_explicit(self):
        block={'start':0.,'end':30.,'segments':[
            {'words':[{'text':'はい','start':1.,'end':2.}]},
            {'words':[{'text':'はい','start':3.,'end':4.}]}]}
        result,_=coherent_words([{'text':'はいはい','start_time':1.,'end_time':4.}],block)
        self.assertTrue(result[0]['crosses_utterance'])
        self.assertTrue(result[0]['utterance_end'])

    def test_cached_old_subtitle_cannot_bypass_policy_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            ja=Path(tmp)/'movie.srt';ja.write_text('existing subtitle')
            with self.assertRaises(RuntimeError):require_current_timing(ja)
            ja.with_suffix('.alignment.json').write_text('{}')
            with self.assertRaises(RuntimeError):require_current_timing(ja)
            ja.with_suffix('.alignment.json').write_text(json.dumps({'timing_policy':TIMING_POLICY,
                'pipeline_revision':pipeline_revision(),
                'manual_review_digest':review_digest(ja.parent),
                'quality_passed':True,'source_sha1':hashlib.sha1(ja.read_bytes()).hexdigest().upper()}))
            require_current_timing(ja)


if __name__=='__main__':unittest.main()
