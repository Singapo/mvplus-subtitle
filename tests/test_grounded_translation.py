import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import grounded_translation as g
from audio_archive import atomic_json
from translate_srt_batch import parse_srt

SETTINGS=dict(model='deepseek-flash',reasoning_effort='high',max_output_tokens=16384)


def targets(prompt):
    return json.loads(prompt.split('仅翻译以下ID，不输出参考中的其他ID：\n')[1].split('\n返回JSON')[0])


def save_response(prompt,path,items):
    content=json.dumps(dict(items=items),ensure_ascii=False)
    atomic_json(path,dict(SETTINGS,prompt_sha256=g.sha(prompt.encode('utf-8')),content=content,
        content_sha256=g.sha(content.encode('utf-8')),finish_reason='stop',response_format='json'))
    return content


class GroundedTests(unittest.TestCase):
    def entries(self,count=6):
        return [(str(i),f'00:00:{i:02},000',f'00:00:{i+1:02},000',f'原文{i}') for i in range(1,count+1)]

    def run_translation(self,entries,path,ask,**kwargs):
        return g.translate(entries,path,'忠实翻译并返回JSON',ask,SETTINGS,max_workers=1,**kwargs)

    def complete(self,calls):
        def ask(prompt,path):
            selected=targets(prompt);calls.append([e['id'] for e in selected])
            return save_response(prompt,path,[dict(id=e['id'],zh='中文'+e['id'],review='none') for e in selected])
        return ask

    def test_same_reference_prefix_and_exact_ids_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';calls=[];prompts=[]
            handler=self.complete(calls)
            def ask(prompt,path):prompts.append(prompt);return handler(prompt,path)
            data=self.entries(4)
            result=self.run_translation(data,output,ask,batch_size=2,frame_size=4,margin=0)
            self.assertTrue(result['quality_passed'])
            self.assertEqual([e[:3] for e in parse_srt(output)],[e[:3] for e in data])
            self.assertEqual(prompts[0].split('仅翻译以下ID')[0],prompts[1].split('仅翻译以下ID')[0])

    def test_resume_does_not_repay_success_and_source_change_only_invalidates_local_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';calls=[];data=self.entries()
            kwargs=dict(batch_size=2,frame_size=2,margin=0)
            self.run_translation(data,output,self.complete(calls),**kwargs)
            self.assertEqual(len(calls),3)
            result=self.run_translation(data,output,self.complete(calls),**kwargs)
            self.assertEqual(len(calls),3);self.assertEqual(result['cache_hits'],3)
            changed=list(data);changed[2]=(*changed[2][:3],'修正后的原文')
            self.run_translation(changed,output,self.complete(calls),**kwargs)
            self.assertEqual(calls[-1],['3','4']);self.assertEqual(len(calls),4)

    def test_partial_response_then_failure_is_recovered_from_raw_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';calls=[];data=self.entries(2)
            def partial(prompt,path):
                ids=[e['id'] for e in targets(prompt)];calls.append(ids)
                if len(ids)==1:raise ConnectionError('lost reply')
                return save_response(prompt,path,[dict(id='1',zh='中文',review='none')])
            first=self.run_translation(data,output,partial)
            self.assertFalse(first['quality_passed']);self.assertFalse(output.exists())
            last=self.run_translation(data,output,self.complete(calls))
            self.assertTrue(last['quality_passed']);self.assertEqual(calls,[['1','2'],['2'],['2']])

    def test_blocked_source_is_not_sent_or_used_as_reference_and_old_output_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';output.write_text('previous verified output')
            prompts=[];calls=[];handler=self.complete(calls)
            def ask(prompt,path):prompts.append(prompt);return handler(prompt,path)
            report=self.run_translation(self.entries(3),output,ask,blocked=['2'])
            self.assertFalse(report['quality_passed']);self.assertNotIn('原文2',prompts[0])
            self.assertEqual(calls,[['1','3']]);self.assertEqual(output.read_text(),'previous verified output')

    def test_source_review_flag_never_becomes_a_final_subtitle_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt'
            def ask(prompt,path):
                return save_response(prompt,path,[dict(id='1',zh='',review='source',span='原文1')])
            report=self.run_translation(self.entries(1),output,ask)
            self.assertEqual(report['review_ids'],['1']);self.assertFalse(output.exists())

    def test_unreliable_reference_outside_target_is_omitted_without_blocking_clear_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';calls=[];prompts=[];handler=self.complete(calls)
            def ask(prompt,path):prompts.append(prompt);return handler(prompt,path)
            report=self.run_translation(self.entries(2),output,ask,blocked=['3'],reference_entries=self.entries(3))
            self.assertTrue(report['quality_passed']);self.assertNotIn('原文3',prompts[0])
            self.assertEqual(report['source_hold_ids'],[])

    def test_malformed_or_corrupted_response_cache_is_not_paid_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';calls=[]
            self.run_translation(self.entries(1),output,self.complete(calls))
            response=next(output.with_suffix('.srt.translation').joinpath('requests').glob('*.json'))
            data=json.loads(response.read_text(encoding='utf-8'));data['content']='{"items":[]}'
            atomic_json(response,data)
            report=self.run_translation(self.entries(1),output,self.complete(calls))
            self.assertFalse(report['quality_passed']);self.assertEqual(len(calls),1)

    def test_strict_json_contract(self):
        entry=self.entries(1)
        invalid=['{"items":[],"items":[]}',
            '{"items":[{"id":"1","id":"1","zh":"中","review":"none"}]}',
            json.dumps(dict(items=[dict(id='2',zh='中',review='none')])),
            json.dumps(dict(items=[dict(id='1',zh='',review='none')])),
            json.dumps(dict(items=[dict(id='1',zh='かな',review='none')])),
            json.dumps(dict(items=[dict(id='1',zh='[原文待核]',review='none')])),
            json.dumps(dict(items=[dict(id='1',zh='',review='source',span='不存在')])),
            json.dumps(dict(items=[dict(id='1',zh='中',review='none',thought='not allowed')]))]
        for raw in invalid:
            with self.subTest(raw=raw),self.assertRaises(ValueError):g.parse_reply(raw,entry)

    def test_cache_diagnostics_do_not_change_acceptance_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);report=dict(quality_passed=True,output_updated=True,cache_hits=0,protocol='grounded-json-v1')
            g.write_certificate(report,folder);first=(folder/'certificate.json').read_bytes()
            report['cache_hits']=100;g.write_certificate(report,folder)
            self.assertEqual(first,(folder/'certificate.json').read_bytes())

    def test_missing_limiter_is_reviewed_not_silently_approved_or_rewritten(self):
        self.assertEqual(g.semantic_risks('一人だけ','一个人'),['source_limiter_needs_review'])
        self.assertEqual(g.semantic_risks('一人だけ','只有一个人'),[])
        self.assertEqual(g.semantic_risks('だけど','但是'),[])
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'out.srt';data=[('1','00:00:01,000','00:00:02,000','一人だけ')]
            def ask(prompt,path):return save_response(prompt,path,[dict(id='1',zh='一个人',review='none')])
            report=self.run_translation(data,output,ask)
            self.assertEqual(report['review_ids'],['1']);self.assertFalse(output.exists())

    def test_failed_recheck_cannot_leave_old_positive_certificate_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);g.write_certificate(dict(quality_passed=True,output_updated=True),folder)
            g.write_certificate(dict(quality_passed=False,output_updated=False,review_ids=['1']),folder)
            self.assertIs(json.loads((folder/'certificate.json').read_text())['quality_passed'],False)

    def test_per_film_source_hold_does_not_pause_other_films_financial_ledger(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            pause=Mock()
            def held(prompt,path):raise g.SourceContextHold('context held')
            report=self.run_translation(self.entries(1),Path(tmp)/'out.srt',held,failure_callback=pause)
            self.assertFalse(report['quality_passed']);pause.assert_not_called()
