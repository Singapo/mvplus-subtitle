"""Source gaps stay explicit; unreviewed timing failures cannot be laundered."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import native_subtitles as native
import source_subset as subset
from audio_archive import digest
from aligned_asr.utterance import pipeline_revision,review_digest
from reprocess_subtitles import require_current_timing


class SourceSubsetTests(unittest.TestCase):
    def fixture(self,root):
        prior=root/'prior';job=root/'job';prior.mkdir();job.mkdir()
        audio=job/'full_audio.wav'
        with wave.open(str(audio),'wb') as f:
            f.setparams((1,2,16000,0,'NONE','not compressed'));f.writeframes(b'\0'*320000)
        source=prior/'draft.srt'
        source.write_text('1\n00:00:01,000 --> 00:00:01,500\n一\n\n2\n00:00:03,000 --> 00:00:03,500\n二\n\n3\n00:00:02,900 --> 00:00:03,400\n三\n\n4\n00:00:05,000 --> 00:00:05,500\n四\n',encoding='utf-8')
        report=prior/'report.json'
        value=dict(timing_policy='utterance_clock_v2',pipeline_revision=pipeline_revision(),
            quality_passed=False,quality_issues=[dict(type='invalid_cue',id=3,start=2.9,end=3.4),
                dict(type='backward_cue',start=2.9,text='三')],source_sha1=digest(source),
            audio_sha1=digest(audio),audio_seconds=10.,manual_review_digest=review_digest(prior),cues=4)
        native.save(report,value)
        return source,report,audio,job/'subset.srt'

    def test_partial_source_is_checked_and_original_failure_remains(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,report,audio,output=self.fixture(Path(tmp));original=report.read_bytes()
            result=subset.build(source,report,audio,output,['2','3'],'Unresolved source clocks remain deferred')
            checked=require_current_timing(output)
            self.assertEqual(report.read_bytes(),original)
            self.assertFalse(checked['full_source_complete'])
            self.assertEqual(checked['source_deferred_cues'],2)
            self.assertEqual([e['id'] for e in result['deferred']],['2','3'])
            target=output.parent/'translation';native.prepare(output,target)
            state=native.checked(target)
            self.assertIn(native.proof(output.parent/'source_subset_review.json'),state['inputs'])

    def test_cannot_discard_only_the_flagged_cue_or_unrelated_cues(self):
        for ids in [['3'],['1','3'],['2','3','4']]:
            with tempfile.TemporaryDirectory() as tmp:
                source,report,audio,output=self.fixture(Path(tmp))
                with self.assertRaises(ValueError):subset.build(source,report,audio,output,ids,'Deferred')
                self.assertFalse(output.exists())

    def test_historical_string_cue_id_is_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,report,audio,output=self.fixture(Path(tmp));data=native.read(report)
            data['quality_issues'][0]['id']='3';native.save(report,data)
            subset.build(source,report,audio,output,['2','3'],'Deferred')
            self.assertEqual(require_current_timing(output)['source_deferred_cues'],2)

    def test_other_quality_failures_or_different_audio_remain_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,report,audio,output=self.fixture(Path(tmp));data=native.read(report)
            data['quality_issues'].append(dict(type='unresolved_events'));native.save(report,data)
            with self.assertRaises(ValueError):subset.build(source,report,audio,output,['2','3'],'Deferred')
            data['quality_issues'].pop();native.save(report,data);audio.write_bytes(audio.read_bytes()+b'changed')
            with self.assertRaises(ValueError):subset.build(source,report,audio,output,['2','3'],'Deferred')

    def test_modified_subset_or_erased_gap_record_is_rejected(self):
        for mutation in ['source','gap','metadata','evidence']:
            with tempfile.TemporaryDirectory() as tmp:
                source,report,audio,output=self.fixture(Path(tmp));subset.build(source,report,audio,output,['2','3'],'Deferred')
                meta=output.with_suffix('.alignment.json')
                if mutation=='source':output.write_text(output.read_text('utf-8').replace('一','改'),encoding='utf-8')
                elif mutation=='gap':
                    path=output.parent/'source_subset_review.json';data=native.read(path);data['deferred']=[];native.save(path,data)
                elif mutation=='metadata':
                    data=native.read(meta);data['full_source_complete']=True;native.save(meta,data)
                else:report.write_text('{}',encoding='utf-8')
                with self.assertRaises((ValueError,RuntimeError,KeyError)):require_current_timing(output)


if __name__=='__main__':unittest.main()
