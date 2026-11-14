import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import source_review as s
from translate_srt_batch import write_srt


class SourceReviewTests(unittest.TestCase):
    entry=('28','00:00:10,000','00:00:12,000','急にと回りに来た')

    def witnesses(self):
        return {(model,'28',window):dict(text='急に泊まりに来た',words=[
            dict(text='泊まり',start=10.4,end=10.9,probability=.95)])
            for model in ('large-v3','qwen') for window in ('near','context')}

    def test_supported_edit_needs_both_architectures_and_windows(self):
        witnesses=self.witnesses()
        result=s.supported_edit(self.entry,'と回り','泊まり',witnesses)
        self.assertEqual(result['text'],'急に泊まりに来た')
        self.assertFalse(result['human_audio_verified'])
        for key in witnesses:
            damaged=copy.deepcopy(witnesses);damaged[key]['text']='別の語句'
            with self.subTest(key=key),self.assertRaises(ValueError):
                s.supported_edit(self.entry,'と回り','泊まり',damaged)

    def test_neighbor_support_low_confidence_or_clock_disagreement_does_not_repair(self):
        for change in (dict(start=9.,end=9.5),dict(probability=.1),dict(start=11.2,end=11.9)):
            witnesses=self.witnesses();witnesses['large-v3','28','context']['words'][0].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                s.supported_edit(self.entry,'と回り','泊まり',witnesses)
        with self.assertRaises(ValueError):s.supported_edit(self.entry,'と回り','',self.witnesses())

    def test_confidence_triage_preserves_nonverbal_and_is_not_edit_authority(self):
        entries=[('1','00:00:01,000','00:00:02,000','普通の家です'),
                 ('2','00:00:03,000','00:00:04,000','ちゅっちゅっ')]
        raw=dict(blocks=[dict(start=0,segments=[dict(words=[
            dict(text='家',start=1.2,end=1.5,probability=.2),
            dict(text='ちゅっ',start=3.2,end=3.5,probability=.01)])])])
        risks=s.triage(entries,raw)
        self.assertEqual([r['id'] for r in risks],['1'])
        self.assertTrue(risks[0]['hard_risk'])
        self.assertNotIn('replacement',risks[0])

    def test_report_binds_exact_source_audio_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.srt';audio=root/'audio.wav';report=root/'review.json'
            write_srt(source,[self.entry]);audio.write_bytes(b'original audio')
            s.make_report(source,audio,report,explicit_ids=['28'])
            self.assertEqual(s.load_report(report,source)['hold_ids'],['28'])
            audio.write_bytes(b'changed audio')
            with self.assertRaises(ValueError):s.load_report(report,source)
            audio.write_bytes(b'original audio');source.write_text('changed source')
            with self.assertRaises(ValueError):s.load_report(report,source)

    def test_stale_asr_audio_is_rejected_before_making_a_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.srt';audio=root/'audio.wav';asr=root/'asr.json'
            write_srt(source,[self.entry]);audio.write_bytes(b'audio')
            asr.write_text(json.dumps(dict(input_sha1='wrong',blocks=[])))
            with self.assertRaises(ValueError):s.make_report(source,audio,root/'report.json',asr)
            self.assertFalse((root/'report.json').exists())

    def test_slice_inherits_only_matching_parent_cues_and_tracks_parent_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'all.srt';part=root/'part.srt';audio=root/'audio.wav'
            other=('29','00:00:13,000','00:00:14,000','分かりました')
            write_srt(source,[self.entry,other]);write_srt(part,[other]);audio.write_bytes(b'audio')
            s.make_report(source,audio,root/'all.json',explicit_ids=['28'])
            result=s.subset_report(root/'all.json',part,root/'part.json')
            self.assertEqual(result['hold_ids'],[]);self.assertEqual(result['reference_hold_ids'],['28'])
            s.load_report(root/'part.json',part)
            parent=json.loads((root/'all.json').read_text());parent['hold_ids']=[]
            (root/'all.json').write_text(json.dumps(parent))
            with self.assertRaises(ValueError):s.load_report(root/'part.json',part)

    def test_review_ids_cannot_be_used_as_clip_paths(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(s.subprocess,'run') as hardware:
                with self.assertRaises(ValueError):
                    s.decode_review(root/'missing.srt',root/'missing.wav',root/'review',
                                    cue_ids=['../outside'])
                hardware.assert_not_called()
            self.assertFalse((root/'review').exists())
