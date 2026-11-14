"""Offline regression coverage for source-bound rebuilds and cloud revisions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
import wave
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import reprocess_subtitles as p
from workflow_fixtures import review_fixture,audit_fixture
from cloud_common import Cloud115

JA='1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n'
ZH='1\n00:00:01,000 --> 00:00:02,000\n你好\n'


def source(path,text=JA,quality=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')
    path.with_suffix('.alignment.json').write_text(json.dumps(dict(timing_policy=p.TIMING_POLICY,
        pipeline_revision=p.pipeline_revision(),
        manual_review_digest=p.review_digest(path.parent),
        audio_sha1=p.sha1(path.parent/'full_audio.wav') if (path.parent/'full_audio.wav').exists() else '',
        quality_passed=quality,source_sha1=p.sha1(path),audio_seconds=4)),encoding='utf-8')


def translation(ja,zh):
    binding,cached=p.prepare_translation(ja,zh,ja.parent)
    zh.write_text(ZH,encoding='utf-8');cached['output_sha1']=p.sha1(zh)
    p.atomic_json(binding,cached)
    return cached


class MemoryCloud(Cloud115):
    """Runs the real replacement algorithm; every remote operation is in-memory."""
    def __init__(self):
        self.files=[];self.bytes={};self.next_fid=100;self.calls=[];self.fail_new_rename=False
    def add(self,fid,name,data):
        import hashlib
        self.bytes[str(fid)]=data
        row=dict(fid=str(fid),cid='2',n=name,s=len(data),sha=hashlib.sha1(data).hexdigest().upper())
        self.files.append(row);return dict(row)
    def list_dir(self,cid):self.calls.append('list');return copy.deepcopy(self.files)
    def download_verified(self,row,path):
        self.calls.append('backup');path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(self.bytes[str(row['fid'])])
    def upload(self,local,cid,name):
        self.calls.append('upload');self.add(self.next_fid,name,local.read_bytes());self.next_fid+=1;return True
    def rename(self,fid,name):
        self.calls.append(('rename',str(fid),name))
        if self.fail_new_rename and str(fid)=='100':return False
        for row in self.files:
            if row['fid']==str(fid):row['n']=name;return True
        return False
    def delete(self,fid):raise AssertionError('No subtitle may be deleted in revision tests')


class SourceBindingTests(unittest.TestCase):
    def test_audio_chunks_removed_only_after_pcm_bytes_match_full_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);folder=job/'audio_chunks';folder.mkdir()
            def wav(path,data):
                with wave.open(str(path),'wb') as handle:
                    handle.setnchannels(1);handle.setsampwidth(2);handle.setframerate(16000);handle.writeframes(data)
            one=b'\x01\x00'*16000;two=b'\x02\x00'*16000
            wav(folder/'00000000.wav',one);wav(folder/'00000001.wav',two)
            wav(job/'full_audio.wav',one+one)
            p.prune_duplicate_audio_chunks(job/'full_audio.wav')
            self.assertTrue(folder.exists())
            wav(job/'full_audio.wav',one+two)
            p.prune_duplicate_audio_chunks(job/'full_audio.wav')
            self.assertFalse(folder.exists());self.assertTrue((job/'full_audio.wav').exists())

    def test_atomic_state_write_retries_windows_reader_lock_and_preserves_existing_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);target=work/'state.json';target.write_text('{"old":true}')
            existing=work/'state.tmp';existing.write_text('preexisting evidence')
            replace=Path.replace;attempts=[]
            def locked(path,destination):
                attempts.append(path)
                if len(attempts)==1:raise PermissionError('synthetic reader lock')
                return replace(path,destination)
            with patch.object(Path,'replace',locked),patch.object(p.time,'sleep'):
                p.atomic_json(target,{'new':True})
            self.assertEqual(json.loads(target.read_text()),{'new':True})
            self.assertEqual(existing.read_text(),'preexisting evidence')
            self.assertEqual(len(attempts),2);self.assertEqual(attempts[0],attempts[1])
            self.assertEqual(list(work.glob('*.part')),[])

    def test_policy_quality_and_exact_bytes_all_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            ja=Path(tmp)/'A.srt';source(ja)
            self.assertEqual(p.require_current_timing(ja)['source_sha1'],p.sha1(ja))
            meta=ja.with_suffix('.alignment.json');report=json.loads(meta.read_text(encoding='utf-8'))
            for changed in (dict(timing_policy='coherent_block_clock_v1'),dict(quality_passed=False),
                            dict(quality_passed=1),dict(source_sha1=report['source_sha1'].lower())):
                meta.write_text(json.dumps(dict(report,**changed)))
                with self.assertRaises(RuntimeError):p.require_current_timing(ja)
            meta.write_text(json.dumps(report));ja.write_text(JA+'\n',encoding='utf-8')
            with self.assertRaises(RuntimeError):p.require_current_timing(ja)

    def test_same_cue_count_changed_source_archives_all_translation_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);ja=job/'A.srt';zh=job/'A.zh-cn.srt';source(ja)
            translation(ja,zh)
            checkpoint=zh.with_suffix(zh.suffix+'.checkpoint');checkpoint.write_text('old checkpoint')
            context=zh.with_suffix('.global_context.json');context.write_text('old context')
            old={path.name:path.read_bytes() for path in p.translation_paths(zh)}
            source(ja,JA.replace('こんにちは','こんばんは'))
            _,binding=p.prepare_translation(ja,zh,job)
            self.assertEqual(binding['source_sha1'],p.sha1(ja));self.assertNotIn('output_sha1',binding)
            for path in p.translation_paths(zh):self.assertFalse(path.exists())
            evidence={path.name:path.read_bytes() for path in (job/'obsolete').rglob('*') if path.is_file()}
            for name,data in old.items():self.assertEqual(evidence[name],data)

    def test_bound_checkpoint_resumes_but_uncommitted_or_tampered_output_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);ja=job/'A.srt';zh=job/'A.zh-cn.srt';source(ja)
            binding,_=p.prepare_translation(ja,zh,job)
            checkpoint=zh.with_suffix(zh.suffix+'.checkpoint');checkpoint.write_text('new checkpoint')
            zh.write_text(ZH,encoding='utf-8')
            p.prepare_translation(ja,zh,job)
            self.assertFalse(zh.exists());self.assertTrue(checkpoint.exists())
            translation(ja,zh);self.assertTrue(p.prepare_translation(ja,zh,job)[1]['output_sha1'])
            zh.write_text(ZH+'\n',encoding='utf-8')
            with self.assertRaises(RuntimeError):p.require_translation(ja,zh)
            p.prepare_translation(ja,zh,job);self.assertFalse(zh.exists())

    def test_missing_audio_uses_verified_duration_without_skipping_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp);ja=job/'A.srt';zh=job/'A.zh-cn.srt';source(ja);translation(ja,zh)
            ok,note=p.validate_current_subtitles(ja,zh,job/'missing.wav')
            self.assertTrue(ok,note);self.assertNotIn('未做',note)
            source(ja,JA.replace('00:00:02,000','00:00:20,000'))
            zh.write_text(ZH.replace('00:00:02,000','00:00:20,000'),encoding='utf-8')
            self.assertFalse(p.validate_current_subtitles(ja,zh,job/'missing.wav')[0])


class RevisionTests(unittest.TestCase):
    def setup_revision(self,work):
        cloud=MemoryCloud();original=cloud.add('1','A.srt',b'frozen original')
        cloud.files[0]['n']='A旧.srt';cloud.add('9','A.srt',b'old broken translation')
        backup=work/'rollback/1/A.srt';backup.parent.mkdir(parents=True);backup.write_bytes(b'frozen original')
        local=work/'A.zh-cn.srt';local.write_bytes(b'correct new translation')
        old=dict(stage='complete',stage_fid='9',sha1=cloud.files[1]['sha'],target_name='A.srt',
                 retained_old=[dict(fid='1',old_name='A.srt',new_name='A旧.srt',sha=original['sha'])])
        record=dict(status='uploaded',replacement=old,cloud_fid='9',cloud_sha1=old['sha1'])
        return cloud,original,backup,local,record

    def test_completed_old_journal_rotates_and_both_old_srts_survive_real_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud,original,backup,local,record=self.setup_revision(work)
            old=copy.deepcopy(record['replacement'])
            binding=dict(source_sha1='SOURCE',output_sha1=p.sha1(local))
            journal=p.revision_journal(record,binding);journal['target_name']='A.srt'
            self.assertEqual(record['prior_publications'][0]['replacement'],old)
            self.assertEqual(record['prior_publications'][0]['cloud_fid'],'9')
            self.assertNotIn('stage_fid',journal)
            p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            cloud.replace_subtitle(original,local,backup,journal,lambda:None,target_name='A.srt')
            self.assertEqual({row['n'] for row in cloud.files},{'A旧.srt','A(9)旧.srt','A.srt'})
            self.assertEqual(cloud.bytes['1'],b'frozen original')
            self.assertEqual(cloud.bytes['9'],b'old broken translation')
            self.assertEqual(journal['sha1'],p.sha1(local));self.assertEqual(journal['stage_fid'],'100')
            self.assertEqual((work/'rollback/9/A.srt').read_bytes(),cloud.bytes['9'])
            p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            cloud.replace_subtitle(original,local,backup,journal,lambda:None,target_name='A.srt')
            self.assertEqual(len(cloud.files),3);self.assertEqual(len(record['prior_publications']),1)
            self.assertIs(p.revision_journal(record,binding),journal)

    def test_failed_revision_rename_keeps_original_retained_and_restores_prior_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud,original,backup,local,record=self.setup_revision(work)
            journal=p.revision_journal(record,dict(source_sha1='SOURCE',output_sha1=p.sha1(local)))
            journal['target_name']='A.srt';p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            cloud.fail_new_rename=True
            with self.assertRaises(RuntimeError):cloud.replace_subtitle(original,local,backup,journal,lambda:None)
            self.assertEqual(next(row['n'] for row in cloud.files if row['fid']=='1'),'A旧.srt')
            self.assertEqual(next(row['n'] for row in cloud.files if row['fid']=='9'),'A.srt')
            cloud.fail_new_rename=False
            cloud.replace_subtitle(original,local,backup,journal,lambda:None)
            self.assertEqual(journal['stage'],'complete');self.assertEqual(len(cloud.files),3)

    def test_incompatible_old_staged_bytes_are_backed_up_and_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud,original,backup,local,record=self.setup_revision(work)
            cloud.add('8','A.mvplus-stage-1.srt',b'failed earlier revision')
            journal=p.revision_journal(record,dict(source_sha1='SOURCE',output_sha1=p.sha1(local)))
            journal['target_name']='A.srt';p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            cloud.replace_subtitle(original,local,backup,journal,lambda:None)
            self.assertEqual(len(cloud.files),4)
            self.assertEqual(cloud.bytes['8'],b'failed earlier revision')
            self.assertTrue(next(row['n'] for row in cloud.files if row['fid']=='8').endswith('旧.srt'))

    def test_tampered_frozen_backup_prevents_any_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud,original,backup,local,record=self.setup_revision(work)
            backup.write_bytes(b'tampered')
            journal={'target_name':'A.srt'}
            with self.assertRaises(RuntimeError):p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            self.assertNotIn('upload',cloud.calls)

    def test_resume_refuses_unknown_canonical_added_after_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud,original,backup,local,record=self.setup_revision(work)
            journal=p.revision_journal(record,dict(source_sha1='SOURCE',output_sha1=p.sha1(local)))
            journal['target_name']='A.srt';p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            cloud.add('80','A.srt',b'externally added subtitle')
            with self.assertRaises(RuntimeError):p.seed_revision(cloud,original,local,backup,journal,lambda:None)
            self.assertNotIn('upload',cloud.calls)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        fixture=patch('source_review.ensure_review',side_effect=review_fixture)
        fixture.start();self.addCleanup(fixture.stop)
    def test_already_uploaded_legacy_source_rebuilds_and_publishes_new_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud,original,backup,local,record=RevisionTests().setup_revision(work)
            plan=work/'plan.json';plan.write_text(json.dumps({'items':[dict(subtitle=original,
                video=dict(fid='v',n='A.mp4',s=1,pc='A'))]}));plan_bytes=plan.read_bytes()
            ja=work/'jobs/1/A.srt';ja.parent.mkdir(parents=True);ja.write_text('legacy source',encoding='utf-8')
            (ja.parent/'full_audio.wav').write_bytes(b'cached audio')
            zh=ja.with_name('A.zh-cn.srt');zh.write_text('legacy translation',encoding='utf-8')
            cp=zh.with_suffix(zh.suffix+'.checkpoint');cp.write_text('legacy checkpoint')
            p.atomic_json(work/'state.json',{'items':{'1':record}})
            calls=[]
            def command(cmd,**kwargs):
                target=Path(cmd[cmd.index('--output')+1]);calls.append(Path(cmd[1]).name)
                if 'extract_aligned_srt.py' in cmd[1]:source(target)
                else:
                    self.assertFalse(cp.exists());self.assertFalse(zh.exists())
                    target.write_text(ZH,encoding='utf-8')
                    audit_fixture(target.with_name(target.name.replace('.zh-cn.srt','.srt')),target)
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(p,'Cloud115',return_value=cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p.subprocess,'run',side_effect=command),\
                 patch.object(p,'validate_subtitles',return_value=(True,'valid cached audio')),\
                 patch.object(p,'archive_original',return_value=(work/'archive','archived')):
                state=p.run(plan,budget_file=work/'cost.json',budget_cny='20')
            rebuilt=state['items']['1']
            self.assertEqual(calls,['extract_aligned_srt.py','translate_srt_batch.py'])
            self.assertEqual(rebuilt['status'],'uploaded');self.assertTrue(state['batch_complete'])
            self.assertEqual(state['completed_current'],1);self.assertEqual(plan.read_bytes(),plan_bytes)
            self.assertEqual(rebuilt['prior_publications'][0]['cloud_fid'],'9')
            self.assertEqual(rebuilt['replacement']['sha1'],p.sha1(zh))
            self.assertTrue(p.publication_complete(rebuilt,ja,zh))
            self.assertEqual({row['n'] for row in cloud.files},{'A旧.srt','A(9)旧.srt','A.srt'})
            evidence=[path.read_bytes() for path in (ja.parent/'obsolete').rglob('*') if path.is_file()]
            self.assertIn(b'legacy source',evidence);self.assertIn(b'legacy translation',evidence)
            self.assertIn(b'legacy checkpoint',evidence)

    def test_stale_uploaded_reextracts_and_quality_failure_does_not_block_good_cached_film(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud=MemoryCloud();rows=[]
            for fid,name in (('1','Bad'),('2','Good')):
                old=cloud.add(fid,name+'.srt',b'old')
                rows.append(dict(subtitle=old,video=dict(fid='v'+fid,n=name+'.mp4',s=1,pc=name)))
            plan=work/'plan.json';plan.write_text(json.dumps({'items':rows}));frozen=plan.read_bytes()
            bad=work/'jobs/1/Bad.srt';bad.parent.mkdir(parents=True);bad.write_text(JA,encoding='utf-8')
            (bad.parent/'full_audio.wav').write_bytes(b'cached audio')
            good=work/'jobs/2/Good.srt';source(good);translation(good,good.with_name('Good.zh-cn.srt'))
            p.atomic_json(work/'state.json',{'items':{'1':{'status':'uploaded'}}})
            def command(cmd,**kwargs):
                self.assertIn('extract_aligned_srt.py',cmd[1])
                self.assertEqual(Path(cmd[2]).read_bytes(),b'cached audio')
                source(Path(cmd[cmd.index('--output')+1]),quality=False)
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(p,'Cloud115',return_value=cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p.subprocess,'run',side_effect=command),\
                 patch.object(p,'archive_original',return_value=(work/'archive','archived')):
                p.run(plan,budget_file=work/'cost.json',budget_cny='20')
            state=json.loads((work/'state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['items']['1']['status'],'failed_extract')
            self.assertEqual(state['items']['2']['status'],'uploaded')
            self.assertFalse(state['batch_complete']);self.assertEqual(state['completed_current'],1)
            self.assertEqual(plan.read_bytes(),frozen)
            self.assertTrue((bad.parent/'full_audio.wav').exists())
            self.assertTrue((good.parent/'Good.zh-cn.srt').exists())
            self.assertEqual(cloud.calls.count('upload'),1)
            # Once all new-policy publication evidence agrees, rerun is local-only.
            rows[0:1]=[];plan.write_text(json.dumps({'items':rows}));calls=list(cloud.calls)
            with patch.object(p,'Cloud115',return_value=cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p.subprocess,'run',side_effect=AssertionError('No cached stage should rerun')):
                p.run(plan)
            self.assertEqual(cloud.calls,calls)

    def test_quality_hold_blocks_translation_and_publication_until_explicit_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);cloud=MemoryCloud();old=cloud.add('1','A.srt',b'old')
            plan=work/'plan.json';plan.write_text(json.dumps({'items':[dict(subtitle=old,
                video=dict(fid='v',n='A.mp4',s=1,pc='A'))]}))
            ja=work/'jobs/1/A.srt';source(ja)
            hold=work/'quality_hold.json';hold.write_text('{"reason":"review"}');hold_bytes=hold.read_bytes()
            translated=threading.Event()
            def command(cmd,**kwargs):
                self.assertFalse(hold.exists());self.assertIn('translate_srt_batch.py',cmd[1])
                target=Path(cmd[cmd.index('--output')+1]);target.write_text(ZH,encoding='utf-8')
                audit_fixture(ja,target);translated.set()
                return SimpleNamespace(returncode=0)
            real_sleep=time.sleep
            with patch.object(p,'Cloud115',return_value=cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p.subprocess,'run',side_effect=command),patch.object(p.time,'sleep',side_effect=lambda _:real_sleep(.01)),\
                 patch.object(p,'archive_original',return_value=(work/'archive','archived')):
                runner=threading.Thread(target=p.run,args=(plan,),kwargs=dict(budget_file=work/'cost.json',budget_cny='20'));runner.start()
                try:
                    deadline=time.monotonic()+3
                    while time.monotonic()<deadline:
                        try:status=json.loads((work/'state.json').read_text(encoding='utf-8'))['items']['1']['status']
                        except (OSError,ValueError,KeyError):status=''
                        if status=='waiting_quality_review':break
                        real_sleep(.01)
                    self.assertEqual(status,'waiting_quality_review',(work/'state.json').read_text(encoding='utf-8'))
                    self.assertFalse(translated.is_set());self.assertNotIn('upload',cloud.calls)
                    self.assertEqual(hold.read_bytes(),hold_bytes)
                finally:
                    hold.unlink();runner.join(5)
                self.assertFalse(runner.is_alive())
            self.assertTrue(translated.is_set())
            self.assertEqual(json.loads((work/'state.json').read_text(encoding='utf-8'))['items']['1']['status'],'uploaded')


if __name__=='__main__':unittest.main()
