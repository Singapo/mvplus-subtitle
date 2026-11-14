from pathlib import Path
import json
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import translate_srt_batch as t


class ParallelTranslationTests(unittest.TestCase):
    def test_disabled_checkpoint_not_mixed_with_enabled_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'cp';entries=self.entries(2)
            cp.write_text(json.dumps(dict(alignment_version=3,input_mtime=t.entries_mtime(entries),model='test',
                target='zh-cn',thinking_mode='disabled',next_index=2,entries=['旧译1','旧译2'])),'utf-8')
            with patch.object(t,'translate_batch_with_splits',return_value=(['新译1','新译2'],0)) as api:
                result=self.run_translation(entries,cp,True)
            api.assert_called_once();self.assertEqual([e[3] for e in result],['新译1','新译2'])

    def entries(self,n=6):return [(str(i),'00:00:01,000','00:00:02,000',f'原文{i}') for i in range(n)]
    def run_translation(self,entries,checkpoint,resume=False):
        return t.adaptive_batch_translate(entries,None,'test','https://api.deepseek.com','ja','zh-cn',2,2,2,checkpoint,resume)

    def test_fast_batch_refills_slot_before_first_batch_finishes_and_output_stays_ordered(self):
        third_started=threading.Event();lock=threading.Lock();active=0;peak=0;contexts={}
        def request(batch,context,*args,**kwargs):
            nonlocal active,peak
            idx=int(batch[0][0])
            with lock:active+=1;peak=max(peak,active);contexts[idx]=context
            try:
                if idx==0:self.assertTrue(third_started.wait(3),'Slow first batch blocked new work')
                if idx==4:third_started.set()
                return [f'中文{e[0]}' for e in batch],0
            finally:
                with lock:active-=1
        with tempfile.TemporaryDirectory() as tmp,patch.object(t,'translate_batch_with_splits',side_effect=request):
            entries=self.entries();out=self.run_translation(entries,Path(tmp)/'cp')
        self.assertEqual([e[3] for e in out],[f'中文{i}' for i in range(6)])
        self.assertEqual([e[:3] for e in out],[e[:3] for e in entries])
        self.assertEqual(contexts[4],['2. 原文2','3. 原文3'])
        self.assertEqual(peak,2)

    def test_failure_retains_out_of_order_success_and_resume_sends_only_missing_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'cp';entries=self.entries(4)
            def request(batch,*args,**kwargs):
                if batch[0][0]=='0':
                    deadline=time.monotonic()+3
                    while time.monotonic()<deadline:
                        if checkpoint.exists() and json.loads(checkpoint.read_text('utf-8'))['entries'][2]:break
                        time.sleep(.01)
                    else:raise AssertionError('Out-of-order success not checkpointed')
                    raise ValueError('Model failed')
                return [f'中文{e[0]}' for e in batch],0
            with patch.object(t,'translate_batch_with_splits',side_effect=request):
                with self.assertRaises(RuntimeError):self.run_translation(entries,checkpoint)
            saved=json.loads(checkpoint.read_text('utf-8'))
            self.assertEqual(saved['next_index'],0)
            self.assertEqual(saved['entries'],['','','中文2','中文3'])
            sent=[]
            def recover(batch,*args,**kwargs):
                sent.extend(e[0] for e in batch);return [f'中文{e[0]}' for e in batch],0
            with patch.object(t,'translate_batch_with_splits',side_effect=recover):out=self.run_translation(entries,checkpoint,True)
            self.assertEqual(sent,['0','1']);self.assertEqual(len(out),4)

    def test_short_batch_response_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(t,'translate_batch_with_splits',return_value=(['中文'],0)):
            with self.assertRaises(RuntimeError):self.run_translation(self.entries(2),Path(tmp)/'cp')

    def test_inflight_success_after_another_batch_fails_is_still_checkpointed(self):
        second_started=threading.Event()
        def request(batch,*args,**kwargs):
            if batch[0][0]=='0':
                self.assertTrue(second_started.wait(3))
                raise ValueError('First batch failed')
            second_started.set();time.sleep(.05)
            return [f'中文{e[0]}' for e in batch],0
        with tempfile.TemporaryDirectory() as tmp,patch.object(t,'translate_batch_with_splits',side_effect=request):
            cp=Path(tmp)/'cp'
            with self.assertRaises(RuntimeError):self.run_translation(self.entries(4),cp)
            self.assertEqual(json.loads(cp.read_text('utf-8'))['entries'],['','','中文2','中文3'])

