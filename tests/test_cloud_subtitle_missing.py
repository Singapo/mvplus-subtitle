import hashlib
from pathlib import Path
import tempfile
import unittest
from test_cloud_subtitle_replace import Fake


class MissingTests(unittest.TestCase):
    def setup_cloud(self):
        c=Fake();video=dict(fid='20',cid='2',n='A.mp4',sha='VIDEO')
        c.files=[dict(video)];return c,video

    def test_publish_and_repeat_do_not_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,v=self.setup_cloud();p=Path(tmp)/'new.srt';p.write_bytes(b'new');j={}
            c.publish_missing_subtitle(v,p,j,lambda:None)
            c.publish_missing_subtitle(v,p,j,lambda:None)
            self.assertEqual(j['stage'],'complete');self.assertEqual(len(c.files),2)
            self.assertEqual(c.deleted,[])

    def test_existing_canonical_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,v=self.setup_cloud();c.files.append(dict(fid='21',n='A.srt',s=3,sha='OLD'))
            p=Path(tmp)/'new.srt';p.write_bytes(b'new')
            with self.assertRaises(RuntimeError):c.publish_missing_subtitle(v,p,{},lambda:None)
            self.assertEqual(len(c.files),2);self.assertEqual(c.renamed,[])

    def test_unknown_upload_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,v=self.setup_cloud();p=Path(tmp)/'new.srt';p.write_bytes(b'new')
            with self.assertRaises(RuntimeError):
                c.publish_missing_subtitle(v,p,dict(upload_attempted=True),lambda:None)
            self.assertEqual(len(c.files),1)

    def test_lost_rename_response_recovers_without_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,v=self.setup_cloud();p=Path(tmp)/'new.srt';p.write_bytes(b'new');j={}
            rename=c.rename
            def lost(fid,name):rename(fid,name);return False
            c.rename=lost
            with self.assertRaises(RuntimeError):c.publish_missing_subtitle(v,p,j,lambda:None)
            c.publish_missing_subtitle(v,p,j,lambda:None)
            self.assertEqual(j['stage'],'complete');self.assertEqual(len(c.files),2)

    def test_mismatched_stage_stops_before_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,v=self.setup_cloud();p=Path(tmp)/'new.srt';p.write_bytes(b'new')
            c.files.append(dict(fid='9',n='A.mvplus-stage-20.srt',s=3,sha='WRONG'))
            with self.assertRaises(RuntimeError):c.publish_missing_subtitle(v,p,{},lambda:None)
            self.assertEqual(c.renamed,[])
