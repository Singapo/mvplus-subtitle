import hashlib
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from cloud_common import Cloud115

class Fake(Cloud115):
    def __init__(self):
        self.old=b'old subtitles';self.files=[dict(fid='1',cid='2',n='A.srt',s=len(self.old),pc='old',sha=hashlib.sha1(self.old).hexdigest())]
        self.fail_rename=False;self.renamed=[];self.deleted=[]
    def list_dir(self,cid):return [dict(x) for x in self.files]
    def download_verified(self,meta,path):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(self.old)
    def upload(self,local,cid,name):
        fid='10' if any(r['fid']=='9' for r in self.files) else '9'
        self.files.append(dict(fid=fid,cid=cid,n=name,s=local.stat().st_size,sha=hashlib.sha1(local.read_bytes()).hexdigest()));return True
    def rename(self,fid,name):
        if self.fail_rename and str(fid)=='9':return False
        for row in self.files:
            if str(row['fid'])==str(fid):
                row['n']=Path(name).stem+Path(row['n']).suffix
                self.renamed.append((str(fid),name));return True
        return False
    def delete(self,fid):self.files=[x for x in self.files if str(x['fid'])!=str(fid)];self.deleted.append(str(fid));return True

class ReplacementTests(unittest.TestCase):
    def test_recovers_stage_with_wrong_extension_without_deleting_old(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Fake();old=dict(c.files[0]);c.files[0]['n']='A旧.srt'
            local=Path(tmp)/'new.srt';local.write_bytes(b'new subtitles')
            c.files.append(dict(fid='9',cid='2',n='A.mvplus-stage-1',s=local.stat().st_size,sha=hashlib.sha1(local.read_bytes()).hexdigest()))
            backup=Path(tmp)/'rollback/1/A.srt';backup.parent.mkdir(parents=True);backup.write_bytes(c.old)
            j={'stage_fid':'9','stage':'published','retained_old':[dict(fid='1',old_name='A.srt',new_name='A旧.srt',sha=old['sha'])]}
            c.replace_subtitle(old,local,backup,j,lambda:None)
            self.assertEqual({r['n'] for r in c.files},{'A旧.srt','A.srt'})
            self.assertEqual(c.deleted,['9'])
            self.assertEqual(j['stage_fid'],'10')
    def test_rename_matches_installed_p115_wire_contract(self):
        from p115sync import P115Client
        cloud=object.__new__(Cloud115);cloud.client=P115Client('UID=1');cloud.log=lambda value:None
        with patch.object(cloud.client,'_request_json',return_value={'state':True}) as request:
            self.assertTrue(cloud.rename('123','A旧.srt'))
            request.assert_called_once_with('POST','https://webapi.115.com/files/batch_rename',data={'files_new_name[123]':'A旧.srt'})
    def test_duplicate_version_and_canonical_old_both_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Fake();c.files[0]['n']='A(1).srt';old=dict(c.files[0])
            c.files.append(dict(c.files[0],fid='2',n='A.srt'))
            local=Path(tmp)/'new.srt';local.write_bytes(b'new subtitles')
            backup=Path(tmp)/'rollback/1/A(1).srt';j={}
            c.replace_subtitle(old,local,backup,j,lambda:None,target_name='A.srt')
            self.assertEqual({r['n'] for r in c.files},{'A(1)旧.srt','A旧.srt','A.srt'})
            self.assertEqual(c.deleted,[])
            c.replace_subtitle(old,local,backup,j,lambda:None,target_name='A.srt')
            self.assertEqual(len(c.files),3)
    def test_verified_swap_retains_rollback(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Fake();old=dict(c.files[0]);local=Path(tmp)/'new.srt';local.write_bytes(b'new subtitles')
            backup=Path(tmp)/'backup.srt';j={}
            c.replace_subtitle(old,local,backup,j,lambda:None)
            self.assertEqual(backup.read_bytes(),b'old subtitles')
            self.assertEqual(c.files[0]['n'],'A旧.srt');self.assertEqual(c.deleted,[])
            self.assertEqual(c.files[1]['n'],'A.srt')
            c.replace_subtitle(old,local,backup,j,lambda:None)
            self.assertEqual(len(c.files),2)
    def test_failed_new_rename_restores_old_without_deleting(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Fake();old=dict(c.files[0]);c.fail_rename=True
            local=Path(tmp)/'new.srt';local.write_bytes(b'new subtitles')
            with self.assertRaises(RuntimeError):c.replace_subtitle(old,local,Path(tmp)/'backup.srt',{},lambda:None)
            self.assertEqual(c.files[0]['n'],'A.srt');self.assertEqual(c.deleted,[])
    def test_stale_selection_refuses_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=Fake();old=dict(c.files[0]);c.files[0]['sha']='changed'
            local=Path(tmp)/'new.srt';local.write_bytes(b'new subtitles')
            with self.assertRaises(RuntimeError):c.replace_subtitle(old,local,Path(tmp)/'backup.srt',{},lambda:None)
            self.assertEqual(len(c.files),1);self.assertEqual(c.deleted,[])
