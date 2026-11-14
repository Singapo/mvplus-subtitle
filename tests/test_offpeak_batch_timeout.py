import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import translate_srt_batch as t

class IdleFuture:
    def __init__(self,fn,args):self.fn,self.args=fn,args
    def result(self,timeout=None):
        if timeout is not None:raise TimeoutError('Simulated discount wait exceeds batch deadline')
        return self.fn(*self.args)
    def cancel(self):return True

class IdleExecutor:
    def __init__(self,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def submit(self,fn,*args):return IdleFuture(fn,args)

def simulated_wait(futures,timeout=None,return_when=None):
    return (set(futures),set()) if timeout is None else (set(),set(futures))

class TimeoutTests(unittest.TestCase):
    def test_discount_wait_does_not_discard_batch_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'DEEPSEEK_OFF_PEAK_ONLY':'1'}),\
             patch.object(t.concurrent.futures,'ThreadPoolExecutor',IdleExecutor),\
             patch.object(t.concurrent.futures,'wait',side_effect=simulated_wait) as waiting,\
             patch.object(t,'translate_batch',return_value=(['中文'],0)):
            checkpoint=Path(tmp)/'test.checkpoint'
            result=t.adaptive_batch_translate([('1','00:00:00,000','00:00:01,000','こんにちは')],None,
                'deepseek-flash','https://api.deepseek.com','ja','zh-cn',128,100,1,checkpoint,False,timeout=300)
            self.assertEqual(result[0][3],'中文')
            self.assertEqual(json.loads(checkpoint.read_text(encoding='utf-8'))['next_index'],1)
            self.assertIsNone(waiting.call_args.kwargs['timeout'])
    def test_non_gated_channel_keeps_outer_deadline(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'DEEPSEEK_OFF_PEAK_ONLY':'1'}),\
             patch.object(t.concurrent.futures,'ThreadPoolExecutor',IdleExecutor),\
             patch.object(t.concurrent.futures,'wait',side_effect=simulated_wait) as waiting:
            with self.assertRaises(RuntimeError):
                t.adaptive_batch_translate([('1','0','1','a')],None,'local','http://localhost:11434',
                    'ja','zh-cn',128,100,1,Path(tmp)/'cp',False,timeout=300)
            self.assertEqual(waiting.call_args.kwargs['timeout'],360)
    def test_http_request_still_has_timeout(self):
        from translation_budget import TranslationBudget
        with tempfile.TemporaryDirectory() as tmp,\
             patch.object(t,'API_BUDGET',TranslationBudget(Path(tmp)/'cost.json','20')),\
             patch.object(t,'check_account_balance'),\
             patch.dict(os.environ,{'DEEPSEEK_OFF_PEAK_ONLY':'1','DEEPSEEK_API_KEY':'test-only'}),\
             patch.object(t,'_load_env_file'),patch('deepseek_offpeak.wait_offpeak') as gate,\
             patch.object(t.urllib.request,'urlopen') as request:
            request.return_value.__enter__.return_value.read1.side_effect=[
                b'{"choices":[{"message":{"content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":20}}',b'']
            self.assertEqual(t.call_ollama('test','deepseek-flash','https://api.deepseek.com',timeout=300),'OK')
            gate.assert_called_once_with(300)
            self.assertEqual(request.call_args.kwargs['timeout'],300)
            payload=json.loads(request.call_args.args[0].data)
            self.assertEqual(payload['thinking'],{'type':'enabled'})
