"""Timing regressions from FNS-122, plus failure preservation at the ASR boundary."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import extract_srt as sut


def segment(start, end, words, text=None, logprob=-0.5):
    return NS(start=start, end=end, words=[NS(start=a, end=b, word=t) for a,b,t in words],
              text=text if text is not None else ''.join(t for _,_,t in words),
              avg_logprob=logprob, no_speech_prob=0.0)


class TimingTests(unittest.TestCase):
    def test_short_line_does_not_start_at_old_segment_boundary(self):
        s = segment(1244.52, 1257.94, [(1256.06,1256.8,'な'),
                    (1256.8,1257.54,'に'),(1257.54,1257.94,'これ')])
        self.assertEqual(sut._word_cues(s), [(1256060,1257940,'なにこれ')])

    def test_pause_is_not_displayed_as_continuous_speech(self):
        s = segment(5399.52,5415.74,[(5399.52,5400.16,'あ、'),(5415.52,5415.74,'ちょっと')])
        self.assertEqual(sut._word_cues(s),[(5399520,5400160,'あ、'),(5415520,5415740,'ちょっと')])

    def test_long_continuous_speech_is_split_without_losing_text(self):
        s=segment(0,12,[(i,i+1,str(i)+' ') for i in range(12)])
        cues=sut._word_cues(s)
        self.assertEqual(''.join(t for _,_,t in cues).replace(' ',''),s.text.replace(' ',''))
        self.assertTrue(all(b-a<=8000 for a,b,_ in cues))

    def test_zero_duration_prefix_keeps_text_with_valid_neighbor(self):
        s=segment(3087.8,3114.8,[(3113.78,3113.78,'気'),(3113.78,3114.82,'持ちいいな')])
        self.assertEqual(sut._word_cues(s),[(3113780,3114820,'気持ちいいな')])

    def test_unusable_alignment_cannot_fall_back_to_old_segment_times(self):
        cases=[segment(0,20,[],'hello'), segment(0,20,[(0,15,'hello')]),
               segment(0,20,[(0,0,'hello')]),segment(0,20,[(1,2,'part')],'complete'),
               segment(0,20,[(2,3,'a'),(1,2,'b')]),segment(0,20,[(float('nan'),2,'a')])]
        for s in cases:
            with self.subTest(s=s),self.assertRaises(ValueError):
                sut._word_cues(s)

    def test_late_alignment_failure_preserves_existing_complete_subtitle(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'movie.srt'
            out.write_text('previous complete subtitle',encoding='utf-8')
            backend=NS(transcribe=lambda *a,**kw:(iter([
                segment(0,2,[(1,2,'こんにちは')]), segment(2,20,[(2,20,'bad')])]),NS(language='ja')))
            with patch.object(sut,'load_faster_whisper',return_value=lambda *a,**kw:backend), \
                 patch.object(sut,'find_model_dir',return_value=Path(d)):
                with self.assertRaises(ValueError):
                    sut.extract(Path(d)/'movie.mp4','test','ja',False,out)
            self.assertEqual(out.read_text(encoding='utf-8'),'previous complete subtitle')

    def test_real_extraction_requests_word_timing_and_keeps_bridged_short_line(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'movie.srt'
            calls=[]
            def transcribe(*a,**kw):
                calls.append(kw)
                return iter([segment(0,40,[(38,39,'こんにちは')],logprob=-1.5)]),NS(language='ja')
            with patch.object(sut,'load_faster_whisper',return_value=lambda *a,**kw:NS(transcribe=transcribe)), \
                 patch.object(sut,'find_model_dir',return_value=Path(d)):
                sut.extract(Path(d)/'movie.mp4','test','ja',False,out)
            self.assertTrue(calls[0]['word_timestamps'])
            self.assertIn('00:00:38,000 --> 00:00:39,000',out.read_text(encoding='utf-8'))


    def test_rejected_low_confidence_bad_alignment_does_not_abort_valid_speech(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'movie.srt'
            backend=NS(transcribe=lambda *a,**kw:(iter([
                segment(0,1,[(0,1,'你好')]),
                segment(2,20,[],text='unreliable',logprob=-2.61178),
                segment(21,22,[(21,22,'再见')])]),NS(language='zh')))
            with patch.object(sut,'load_faster_whisper',return_value=lambda *a,**kw:backend), \
                 patch.object(sut,'find_model_dir',return_value=Path(d)):
                sut.extract(Path(d)/'movie.mp4','test','zh',False,out)
            text=out.read_text(encoding='utf-8')
            self.assertIn('你好',text);self.assertIn('再见',text)
            self.assertNotIn('unreliable',text)

    def test_confidence_boundary_still_rejects_invalid_alignment(self):
        for confidence in (-2.5,-1.5):
            with self.subTest(confidence=confidence),tempfile.TemporaryDirectory() as d:
                out=Path(d)/'movie.srt';out.write_text('previous',encoding='utf-8')
                backend=NS(transcribe=lambda *a,**kw:(iter([
                    segment(0,1,[(0,1,'你好')]),
                    segment(2,20,[],text='unreliable',logprob=confidence)]),NS(language='zh')))
                with patch.object(sut,'load_faster_whisper',return_value=lambda *a,**kw:backend), \
                     patch.object(sut,'find_model_dir',return_value=Path(d)),self.assertRaises(ValueError):
                    sut.extract(Path(d)/'movie.mp4','test','zh',False,out)
                self.assertEqual(out.read_text(encoding='utf-8'),'previous')

    def test_only_rejected_low_confidence_speech_preserves_existing_subtitle(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'movie.srt';out.write_text('previous',encoding='utf-8')
            backend=NS(transcribe=lambda *a,**kw:(iter([
                segment(0,20,[],text='unreliable',logprob=-3)]),NS(language='zh')))
            with patch.object(sut,'load_faster_whisper',return_value=lambda *a,**kw:backend), \
                 patch.object(sut,'find_model_dir',return_value=Path(d)),self.assertRaises(ValueError):
                sut.extract(Path(d)/'movie.mp4','test','zh',False,out)
            self.assertEqual(out.read_text(encoding='utf-8'),'previous')


if __name__=='__main__':
    unittest.main()
