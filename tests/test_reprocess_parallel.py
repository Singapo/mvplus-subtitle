"""Prove stages overlap rather than just checking worker configuration."""
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import reprocess_subtitles as p
from workflow_fixtures import review_fixture,audit_fixture

class ParallelTests(unittest.TestCase):
    def test_native_export_allows_translation_before_later_asr_finishes(self):
        asr_a=threading.Event();download_b=threading.Event();ready_a=threading.Event()
        translating=threading.Event();asr_c=threading.Event();translated=threading.Event()
        errors=[]
        class Cloud:
            def __init__(self,*a):pass
            def extract_audio(self,pc,path):
                if pc=='B':
                    if not asr_a.wait(3):raise AssertionError('Next download did not overlap A ASR')
                    download_b.set()
                path.write_bytes(b'audio')
        def extract(cmd,**kwargs):
            target=Path(cmd[cmd.index('--output')+1]);letter=target.stem
            if letter=='A':
                asr_a.set()
                if not download_b.wait(3):raise AssertionError('B download blocked behind A ASR')
            if letter=='C':
                if not translating.wait(3):raise AssertionError('Native translation waited for all sources')
                asr_c.set()
            target.write_text('source');return SimpleNamespace(returncode=0,stdout='',stderr='')
        def export(row,source):
            if row['video']['fid']=='vA':ready_a.set()
        def native_translation():
            try:
                if not ready_a.wait(3):raise AssertionError('A source was not exported immediately')
                translating.set()
                if not asr_c.wait(3):raise AssertionError('Native translation blocked later ASR')
                translated.set()
            except Exception as exc:errors.append(exc)
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);plan=work/'plan.json'
            plan.write_text(json.dumps(dict(items=[dict(subtitle=dict(fid=c,n=c+'.srt'),
                video=dict(fid='v'+c,n=c+'.mp4',pc=c,s=i)) for i,c in enumerate('ABC')])))
            worker=threading.Thread(target=native_translation);worker.start()
            with patch.object(p,'Cloud115',Cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p,'free_gb',return_value=100),patch.object(p.subprocess,'run',side_effect=extract),\
                 patch.object(p,'publication_complete',return_value=False),patch.object(p,'timing_is_current',return_value=False),\
                 patch.object(p,'require_current_timing',return_value={'source_sha1':'verified'}),\
                 patch.object(p,'archive_audio',side_effect=ValueError('test has no actual WAV')),\
                 patch.object(p,'archive_original',return_value=(work/'archive','okay')):
                state=p.run(plan,extract_only=True,on_extracted=export)
            worker.join(4)
            self.assertFalse(worker.is_alive());self.assertFalse(errors)
            self.assertTrue(download_b.is_set());self.assertTrue(translated.is_set())
            self.assertTrue(all(r['status']=='extracted' for r in state['items'].values()))

    def test_publish_only_withholds_source_context_review(self):
        class Cloud:
            def __init__(self,*a):pass
            def replace_subtitle(self,*a,**kw):raise AssertionError('Held source must not publish')
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);job=work/'jobs/1';job.mkdir(parents=True)
            (job/'A.srt').write_text('source');(job/'A.zh-cn.srt').write_text('中文')
            (job/'translation_content_hold.json').write_text('{}')
            plan=work/'plan.json';plan.write_text(json.dumps(dict(items=[dict(
                subtitle=dict(fid='1',n='A.srt'),video=dict(fid='v1',n='A.mp4'))])))
            with patch.object(p,'Cloud115',Cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p,'publication_complete',return_value=False),patch.object(p,'timing_is_current',return_value=True),\
                 patch.object(p,'require_current_timing',return_value={'source_sha1':'verified'}),patch.object(p,'require_translation',return_value={}),\
                 patch.object(p,'archive_original',return_value=(work/'archive','okay')),\
                 patch.object(p.subprocess,'run',side_effect=AssertionError('No generation allowed')):
                state=p.run(plan,publish_only=True)
            self.assertEqual(state['items']['1']['status'],'failed_publish')
            self.assertIn('context review',state['items']['1']['error'])

    def test_publish_only_can_forward_agent_outputs_without_any_generation(self):
        class Cloud:
            def __init__(self,*a):pass
            def extract_audio(self,*a):raise AssertionError('No audio download allowed')
            def replace_subtitle(self,original,local,backup,journal,save,target_name=None):
                journal.update(stage='complete',stage_fid='new',sha1=p.sha1(local));save()
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);job=work/'jobs/1';job.mkdir(parents=True)
            ja=job/'A.srt';zh=job/'A.zh-cn.srt';ja.write_text('日文');zh.write_text('中文')
            plan=work/'plan.json';plan.write_text(json.dumps(dict(items=[dict(subtitle=dict(fid='1',n='A.srt'),video=dict(fid='v1',n='A.mp4'))])))
            (work/'translation_credit_hold.json').write_text('{}');(work/'translation_provider_override.json').write_text('{}')
            binding=dict(source_sha1=p.sha1(ja),output_sha1=p.sha1(zh),timing_policy=p.TIMING_POLICY)
            with patch.object(p,'Cloud115',Cloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p,'publication_complete',return_value=False),patch.object(p,'timing_is_current',return_value=True),\
                 patch.object(p,'require_current_timing',return_value=binding),patch.object(p,'require_translation',return_value=binding),\
                 patch.object(p,'archive_original',return_value=(work/'archive','okay')),patch.object(p,'seed_revision'),\
                 patch.object(p,'validate_current_subtitles',return_value=(True,'valid')),\
                 patch.object(p.subprocess,'run',side_effect=AssertionError('No generation subprocess allowed')):
                state=p.run(plan,publish_only=True)
            self.assertEqual(state['items']['1']['status'],'uploaded')
            self.assertTrue((work/'translation_credit_hold.json').exists())

    def test_user_provider_override_blocks_deepseek_before_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);(work/'translation_provider_override.json').write_text('{}')
            with patch.object(p,'load_env') as credentials:
                with self.assertRaisesRegex(RuntimeError,'Codex agent'):p.run(work/'plan.json')
                credentials.assert_not_called()

    def test_credit_hold_blocks_paid_pipeline_before_credentials_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);plan=work/'plan.json'
            plan.write_text(json.dumps(dict(items=[])))
            (work/'translation_credit_hold.json').write_text('{}')
            with patch.object(p,'load_env') as credentials:
                with self.assertRaisesRegex(RuntimeError,'balance unavailable'):p.run(plan)
                credentials.assert_not_called()
                self.assertFalse(p.run(plan,extract_only=True)['active'])
                credentials.assert_not_called()

    def test_extract_only_drains_more_than_queue_capacity_with_quality_hold(self):
        class FakeCloud:
            def __init__(self,*args):pass
            def extract_audio(self,pc,path):path.write_bytes(b'audio')
            def replace_subtitle(self,*args,**kwargs):raise AssertionError('Unexpected publication')
        def extract(cmd,**kwargs):
            self.assertIn('extract_aligned_srt.py',cmd[1])
            target=Path(cmd[cmd.index('--output')+1]);target.write_text('source')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);manifest=work/'plan.json'
            rows=[dict(subtitle=dict(fid=str(i),n=f'{i}.srt'),video=dict(fid=f'v{i}',n=f'{i}.mp4',s=i,pc=str(i))) for i in range(10)]
            manifest.write_text(json.dumps(dict(items=rows)))
            (work/'quality_hold.json').write_text('{}')
            with patch.object(p,'Cloud115',FakeCloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p,'free_gb',return_value=100),patch.object(p.subprocess,'run',side_effect=extract),\
                 patch.object(p,'publication_complete',return_value=False),patch.object(p,'timing_is_current',return_value=False),\
                 patch.object(p,'require_current_timing',return_value={'source_sha1':'verified'}),\
                 patch.object(p,'archive_original',return_value=(work/'archive','archived')):
                state=p.run(manifest,extract_only=True)
            self.assertFalse(state['active'])
            self.assertEqual([r['status'] for r in state['items'].values()],['extracted']*10)
            self.assertTrue(all(r['translation_pending'] for r in state['items'].values()))
            self.assertTrue((work/'quality_hold.json').exists())

    def test_download_overlaps_asr_and_asr_continues_while_translation_waits(self):
        started_a=threading.Event();downloaded_b=threading.Event();extracted_c=threading.Event()
        overlap=[]
        class FakeCloud:
            def __init__(self,*args):pass
            def download_verified(self,meta,path):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'old')
            def extract_audio(self,pc,path):
                if pc=='B':
                    if not started_a.wait(3):raise AssertionError('ASR did not start during next download')
                    overlap.append('B download during A ASR');downloaded_b.set()
                if pc=='C' and not downloaded_b.wait(3):raise AssertionError('B download blocked')
                path.write_bytes(b'audio')
            def replace_subtitle(self,original,local,backup,journal,save,target_name=None):
                journal.update(stage='complete',stage_fid='new-'+original['fid'],sha1=p.sha1(local));save()
        def subprocess(cmd,**kwargs):
            target=Path(cmd[cmd.index('--output')+1])
            stem=target.name.split('.')[0]
            if 'extract_aligned_srt.py' in cmd[1]:
                if stem=='A':
                    started_a.set()
                    if not downloaded_b.wait(3):raise AssertionError('Next download did not overlap ASR')
                target.write_text('1\n00:00:00,000 --> 00:00:01,000\ntext\n',encoding='utf-8')
                target.with_suffix('.alignment.json').write_text(json.dumps({'timing_policy':p.TIMING_POLICY,
                    'pipeline_revision':p.pipeline_revision(),
                    'manual_review_digest':p.review_digest(target.parent),'audio_sha1':p.sha1(target.parent/'full_audio.wav'),
                    'quality_passed':True,'source_sha1':p.sha1(target),'audio_seconds':2}))
                if stem=='C':extracted_c.set()
            else:
                if stem=='A':
                    if not extracted_c.wait(3):raise AssertionError('Translation wait blocked subsequent ASR')
                    overlap.append('C ASR completed during A translation')
                target.write_text('1\n00:00:00,000 --> 00:00:01,000\n中文\n',encoding='utf-8')
                audit_fixture(target.with_name(target.name.replace('.zh-cn.srt','.srt')),target)
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);manifest=work/'plan.json'
            rows=[{'subtitle':{'fid':letter,'n':letter+'.srt'},'video':{'fid':'v'+letter,'n':letter+'.mp4','s':i,'pc':letter}}
                  for i,letter in enumerate('ABC')]
            manifest.write_text(json.dumps({'items':rows}),encoding='utf-8')
            with patch.object(p,'Cloud115',FakeCloud),patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p,'free_gb',return_value=100),patch.object(p.subprocess,'run',side_effect=subprocess),\
                 patch.object(p,'validate_subtitles',return_value=(True,'valid')),\
                 patch.object(p,'seed_revision'),\
                 patch('source_review.ensure_review',side_effect=review_fixture),\
                 patch.object(p,'archive_original',return_value=(work/'archive','archived')):
                p.run(manifest,budget_file=work/'cost.json',budget_cny='20')
            state=json.loads((work/'state.json').read_text(encoding='utf-8'))
            self.assertEqual([state['items'][x]['status'] for x in 'ABC'],['uploaded']*3)
            self.assertIn('B download during A ASR',overlap)
            self.assertIn('C ASR completed during A translation',overlap)
