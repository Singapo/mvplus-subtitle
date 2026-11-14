import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from translation_budget import BudgetExceeded, TranslationBudget
import translate_srt_batch as t


class TranslationBudgetTests(unittest.TestCase):
    def test_scoped_cap_preserves_prior_unknown_and_is_shared_across_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cost.json';shared=TranslationBudget(path,'500')
            old=shared.reserve('old unknown',100,'deepseek-flash')
            before=json.loads(path.read_text())['requests'][old]
            def reserve(_):
                budget=TranslationBudget(path,'500',scope='new-batch',scope_limit_cny='.10')
                try:return budget.reserve('new translation',4096,'deepseek-flash')
                except BudgetExceeded:return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results=list(pool.map(reserve,range(8)))
            self.assertEqual(sum(v is not None for v in results),2)
            scoped=TranslationBudget(path,'500',scope='new-batch',scope_limit_cny='.10')
            self.assertLessEqual(scoped.summary()['scope_spent_or_reserved_cny'],.10)
            self.assertEqual(json.loads(path.read_text())['requests'][old],before)
            with self.assertRaises(ValueError):TranslationBudget(path,'500',scope='new-batch',scope_limit_cny='20')

    def setUp(self):
        work=tempfile.TemporaryDirectory();self.addCleanup(work.cleanup)
        budget=TranslationBudget(Path(work.name)/'api.json','20')
        active=patch.object(t,'API_BUDGET',budget);active.start();self.addCleanup(active.stop)
        account=patch.object(t,'check_account_balance');account.start();self.addCleanup(account.stop)
    def test_parallel_reservations_cannot_exceed_shared_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'cost.json'
            budget = TranslationBudget(ledger, '.10')
            def reserve(_):
                try:
                    return budget.reserve('test', 4096, 'deepseek-flash')
                except BudgetExceeded:
                    return None
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(reserve, range(20)))
            self.assertEqual(sum(r is not None for r in results), 2)
            resumed = TranslationBudget(ledger, '.10')
            self.assertLessEqual(resumed.summary()['conservative_spent_or_reserved_cny'], .10)
            with self.assertRaises(ValueError):
                TranslationBudget(ledger, '20')

    def test_authorized_limit_increase_keeps_existing_costs_and_halt(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cost.json';budget=TranslationBudget(path,'10')
            ident=budget.reserve('test',100,'deepseek-flash')
            budget.settle(ident,dict(prompt_tokens=10,completion_tokens=20))
            budget.halt('keep this hold')
            before=json.loads(path.read_text(encoding='utf-8'))
            result=budget.increase_limit('20','User explicitly raised cumulative budget to 20 CNY')
            after=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(after['requests'],before['requests'])
            self.assertEqual(result['limit_cny'],20)
            self.assertEqual(result['halt_reason'],'keep this hold')
            self.assertEqual(TranslationBudget(path,'20').summary()['requests'],1)

    def test_limit_increase_does_not_release_unknown_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cost.json';budget=TranslationBudget(path,'10')
            budget.reserve('test',100,'deepseek-flash');before=path.read_bytes()
            with self.assertRaises(BudgetExceeded):budget.increase_limit('20','User authorization')
            self.assertEqual(path.read_bytes(),before)

    def test_windows_pending_lock_deletion_is_retried_without_extra_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20');real_open=os.open;attempts=[]
            def pending_delete(path,*args,**kwargs):
                if str(path).endswith('.lock'):
                    attempts.append(path)
                    if len(attempts)==1:raise PermissionError('Windows lock deletion still pending')
                return real_open(path,*args,**kwargs)
            with patch('translation_budget.os.open',side_effect=pending_delete),patch('translation_budget.time.sleep'):
                budget.reserve('test',100,'deepseek-flash')
            self.assertGreaterEqual(len(attempts),2);self.assertEqual(budget.summary()['requests'],1)

    def test_usage_settles_and_invalid_usage_keeps_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = TranslationBudget(Path(tmp) / 'cost.json', '20')
            rid = budget.reserve('abc', 100, 'deepseek-flash')
            before = budget.summary()
            with self.assertRaises(RuntimeError):
                budget.settle(rid, {})
            self.assertEqual(before['conservative_spent_or_reserved_cny'],budget.summary()['conservative_spent_or_reserved_cny'])
            self.assertIsNotNone(budget.summary()['halt_reason'])
            with self.assertRaises(RuntimeError):
                budget.settle(rid, dict(prompt_tokens=1, completion_tokens=101))
            budget.settle(rid, dict(prompt_tokens=20, completion_tokens=30))
            self.assertEqual(budget.summary()['conservative_spent_or_reserved_cny'], .00028)
            self.assertEqual(budget.summary()['unknown_requests'], 0)

    def test_unknown_request_is_reserved_and_blocks_automatic_resend(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = TranslationBudget(Path(tmp) / 'cost.json', '.015')
            with patch.object(t, 'API_BUDGET', budget), patch.object(t, 'API_MAX_TOKENS', 1000), \
                 patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test', 'DEEPSEEK_OFF_PEAK_ONLY': '0'}), \
                 patch.object(t, '_load_env_file'), patch.object(t.time, 'sleep'), \
                 patch.object(t.urllib.request, 'urlopen', side_effect=http.client.RemoteDisconnected()) as api:
                with self.assertRaises(http.client.RemoteDisconnected):
                    t.call_ollama('test', 'deepseek-flash', 'https://api.deepseek.com')
                with self.assertRaises(BudgetExceeded):
                    t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
                self.assertEqual(api.call_count, 1)
                self.assertEqual(budget.summary()['unknown_requests'], 1)
                payload = json.loads(api.call_args.args[0].data)
                self.assertEqual(payload['max_tokens'], 1000)

    def test_length_finish_stops_without_paid_split_recovery(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test', 'DEEPSEEK_OFF_PEAK_ONLY': '0'}), \
             patch.object(t, '_load_env_file'), \
             patch.object(t.urllib.request, 'urlopen') as api:
            api.return_value.__enter__.return_value.read1.side_effect = [json.dumps(dict(
                choices=[dict(message=dict(content='1. test'), finish_reason='length')],
                usage=dict(prompt_tokens=10,completion_tokens=30))).encode(),b'']
            with self.assertRaises(RuntimeError):
                t.call_ollama('test', 'deepseek-flash', 'https://api.deepseek.com')
            api.assert_called_once()

    def test_unknown_model_and_nonfinite_budget_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for limit in ('NaN', 'Infinity', '0', '-1'):
                with self.assertRaises(ValueError):
                    TranslationBudget(Path(tmp) / 'cost.json', limit)
            budget = TranslationBudget(Path(tmp) / 'cost.json', '20')
            with self.assertRaises(ValueError):
                budget.reserve('test', 100, 'deepseek-v4-pro')

    def test_low_effort_keeps_thinking_and_is_sent_to_api(self):
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test', 'DEEPSEEK_OFF_PEAK_ONLY': '0'}), \
             patch.object(t, '_load_env_file'), patch.object(t, 'API_REASONING_EFFORT', 'low'), \
             patch.object(t.urllib.request, 'urlopen') as api:
            api.return_value.__enter__.return_value.read1.side_effect = [json.dumps(dict(
                choices=[dict(message=dict(content='1. 中文'), finish_reason='stop')],
                usage=dict(prompt_tokens=10,completion_tokens=30))).encode(),b'']
            t.call_ollama('test', 'deepseek-flash', 'https://api.deepseek.com')
            payload = json.loads(api.call_args.args[0].data)
            self.assertEqual(payload['reasoning_effort'], 'low')
            self.assertEqual(payload['thinking'], {'type': 'enabled'})

    def test_english_prompt_does_not_load_japanese_name_table(self):
        with patch.object(t, 'load_name_map') as names:
            t.build_prompt([('12', 'a', 'b', 'Hello')], [], 'en', 'zh-cn', system_prompt='Translate')
            names.assert_not_called()

    def test_keepalive_cannot_extend_wall_clock_deadline(self):
        from unittest.mock import Mock
        response = Mock()
        response.read1.return_value = b' '
        with patch.object(t.time, 'monotonic', side_effect=[0, 1, 3]):
            with self.assertRaisesRegex(RuntimeError, 'wall-clock'):
                t.read_budgeted_response(response, 2)
        self.assertEqual(response.read1.call_count, 2)

    def test_socket_read_receives_remaining_deadline_not_a_fresh_full_timeout(self):
        from unittest.mock import MagicMock
        from types import SimpleNamespace
        response=MagicMock(spec=http.client.HTTPResponse);sock=MagicMock()
        response.fp=SimpleNamespace(raw=SimpleNamespace(_sock=sock))
        response.read1.side_effect=[b' ',b'{}',b'']
        with patch.object(t.time,'monotonic',side_effect=[0,1,1.9]):
            self.assertEqual(t.read_budgeted_response(response,2),b' {}')
        values=[call.args[0] for call in sock.settimeout.call_args_list]
        self.assertEqual(values[:2],[2,1]);self.assertAlmostEqual(values[2],.1)

    def test_successful_budgeted_api_response_is_settled(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = TranslationBudget(Path(tmp) / 'cost.json', '20')
            result = dict(choices=[dict(message=dict(content='12. 中文'), finish_reason='stop')],
                          usage=dict(prompt_tokens=100, completion_tokens=200))
            with patch.object(t, 'API_BUDGET', budget), \
                 patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test', 'DEEPSEEK_OFF_PEAK_ONLY': '0'}), \
                 patch.object(t, '_load_env_file'), patch.object(t.urllib.request, 'urlopen') as api:
                api.return_value.__enter__.return_value.read1.side_effect = [json.dumps(result).encode(), b'']
                self.assertEqual(t.call_ollama('test', 'deepseek-flash', 'https://api.deepseek.com'), '12. 中文')
            self.assertEqual(budget.summary()['conservative_spent_or_reserved_cny'], .0018)
            self.assertEqual(budget.summary()['unknown_requests'], 0)

    def test_aborted_reply_is_not_accepted_even_if_some_text_and_usage_exist(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test','DEEPSEEK_OFF_PEAK_ONLY':'0'}),\
             patch.object(t,'_load_env_file'),patch.object(t.urllib.request,'urlopen') as api:
            api.return_value.__enter__.return_value.read1.side_effect=[json.dumps(dict(
                choices=[dict(message=dict(content='1. partial'),finish_reason='aborted')],
                usage=dict(prompt_tokens=10,completion_tokens=20))).encode(),b'']
            with self.assertRaises(RuntimeError):t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            self.assertIsNotNone(t.API_BUDGET.summary()['halt_reason']);api.assert_called_once()

    def test_thinking_only_cutoff_still_persists_receipt_and_cost(self):
        with tempfile.TemporaryDirectory() as tmp,\
             patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test','DEEPSEEK_OFF_PEAK_ONLY':'0'}),\
             patch.object(t,'_load_env_file'),patch.object(t.urllib.request,'urlopen') as api:
            api.return_value.__enter__.return_value.read1.side_effect=[json.dumps(dict(
                choices=[dict(message=dict(content=None),finish_reason='length')],
                usage=dict(prompt_tokens=10,completion_tokens=100))).encode(),b'']
            receipt=Path(tmp)/'reply.json'
            with self.assertRaises(ValueError):t.call_ollama('test','deepseek-flash','https://api.deepseek.com',response_path=receipt)
            self.assertEqual(json.loads(receipt.read_text())['content'],'')
            self.assertEqual(t.API_BUDGET.summary()['unknown_requests'],0)
            self.assertIsNotNone(t.API_BUDGET.summary()['halt_reason'])

    def test_changed_effort_does_not_reuse_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'out.checkpoint'
            entries = [('12', '00:00:01,000', '00:00:02,000', 'Hello')]
            checkpoint.write_text(json.dumps(dict(alignment_version=3,
                input_mtime=t.entries_mtime(entries), model='deepseek-flash', target='zh-cn',
                thinking_mode='enabled', next_index=1, entries=['旧译'])), encoding='utf-8')
            with patch.object(t, 'API_REASONING_EFFORT', 'low'), \
                 patch.object(t, 'translate_batch_with_splits', return_value=(['新译'], 0)) as api:
                out = t.adaptive_batch_translate(entries, None, 'deepseek-flash', 'https://api.deepseek.com',
                    'en', 'zh-cn', 4, 0, 1, checkpoint, True)
            api.assert_called_once()
            self.assertEqual(out[0][3], '新译')
