from pathlib import Path
import json
import queue
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import reprocess_subtitles as p
import nosub_pipeline as n


class StageBindingTests(unittest.TestCase):
    def test_separate_work_directory_owns_audio_and_manual_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'A.srt';job=root/'A.alignment';job.mkdir()
            source.write_text('1\n00:00:01,000 --> 00:00:02,000\n原文\n',encoding='utf-8')
            (job/'full_audio.wav').write_bytes(b'audio')
            (job/'manual_audio_review.json').write_text(json.dumps(dict(utterance_confirmations=[dict(text='evidence')])) )
            source.with_suffix('.alignment.json').write_text(json.dumps(dict(timing_policy=p.TIMING_POLICY,
                pipeline_revision=p.pipeline_revision(),manual_review_digest=p.review_digest(job),
                audio_sha1=p.sha1(job/'full_audio.wav'),source_sha1=p.sha1(source),quality_passed=True)))
            p.require_current_timing(source,work=job)
            with self.assertRaises(RuntimeError):p.require_current_timing(source)
            (job/'full_audio.wav').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):p.require_current_timing(source,work=job)

    def test_failed_reextraction_preserves_unverified_source_and_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=root/'A.mp4';video.write_bytes(b'video');source=root/'A.srt'
            source.write_bytes(b'user draft');updates=[]
            state=SimpleNamespace(set=lambda *a,**kw:updates.append((a,kw)))
            with patch.object(n.subprocess,'run',return_value=SimpleNamespace(returncode=1,stderr='failed')):
                n.do_extract(dict(stem='A'),root,state,queue.Queue())
            self.assertEqual(source.read_bytes(),b'user draft');self.assertEqual(video.read_bytes(),b'video')
            self.assertEqual(next((root/'A.alignment/prior_sources').glob('*.srt')).read_bytes(),b'user draft')
