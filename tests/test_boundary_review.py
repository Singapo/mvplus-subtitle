from pathlib import Path
import sys
import unittest
import json
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from boundary_review import decision,verify_manifest
from audio_archive import digest


class BoundaryReviewTests(unittest.TestCase):
    def issue(self,text='すっごい'):
        return dict(type='cross_block_source_conflict',text=text,start=10.,end=10.,boundary=10.,
            words=[dict(text=text,start=10.,end=10.)])
    def witnesses(self,text='うん'):
        return {k:dict(segments=[dict(text=text)]) for k in ('760M_near','large-v3_near','760M_owner','large-v3_owner')}
    def test_zero_truncated_sound_tail_can_be_logged_without_rewriting_as_dialogue(self):
        self.assertEqual(decision(self.issue('は'),'はぁっはぁっは',self.witnesses()),'unlocated_truncated_nonverbal_tail_after_context_review')
    def test_lexical_unit_needs_all_four_decodes_without_spelling_equivalent_support(self):
        self.assertEqual(decision(self.issue(),'ああすっごい',self.witnesses()),'unsupported_zero_time_lexical_unit_after_context_review')
        with self.assertRaises(ValueError):decision(self.issue(),'ああすっごい',self.witnesses('すごい、すごい'))
        with self.assertRaises(ValueError):decision(self.issue(),'ああすっごい',{})
    def test_positive_duration_or_wrong_source_not_reviewable(self):
        with self.assertRaises(ValueError):decision(dict(self.issue(),end=10.1),'すっごい',self.witnesses())
        with self.assertRaises(ValueError):decision(self.issue(),'違う文字',self.witnesses())

    def fixture(self,job):
        issue=self.issue('は');issue.update(block=0,segment=0)
        (job/'asr.json').write_text(json.dumps(dict(blocks=[dict(start=0.,end=10.,segments=[dict(text='はぁっはぁっは')])])),'utf-8')
        folder=job/'boundary_review';folder.mkdir()
        witnesses={}
        for key in self.witnesses():
            file=folder/(key+'.json')
            file.write_text(json.dumps(dict(audio_sha1='audio',model_identity='model',target=issue,
                owner_start=0.,owner_end=10.,clip_start=6.,clip_end=14.,segments=[dict(text='呼吸')])),'utf-8')
            witnesses[key]=dict(file=str(file.relative_to(job)),sha1=digest(file))
        review=dict(policy='four_decode_zero_boundary_review',audio_sha1='audio',asr_sha1=digest(job/'asr.json'),pipeline_revision='revision',
            resolved=[dict(issue=issue,reason='unlocated_truncated_nonverbal_tail_after_context_review',witnesses=witnesses)])
        path=job/'boundary_review.json';path.write_text(json.dumps(review),'utf-8')
        return dict(audio_sha1='audio',audio_seconds=20.,pipeline_revision='revision',boundary_review_sha1=digest(path))

    def test_review_provenance_is_revalidated_and_changed_witness_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch('boundary_review.model_identity',return_value='model'):
            job=Path(tmp);report=self.fixture(job)
            self.assertEqual(len(verify_manifest(job,report)['resolved']),1)
            (job/'boundary_review/760M_near.json').write_text('{}')
            with self.assertRaises(ValueError):verify_manifest(job,report)

    def test_changed_recognition_input_rejects_old_review(self):
        with tempfile.TemporaryDirectory() as tmp,patch('boundary_review.model_identity',return_value='model'):
            job=Path(tmp);report=self.fixture(job);(job/'asr.json').write_text('{}')
            with self.assertRaises(ValueError):verify_manifest(job,report)
