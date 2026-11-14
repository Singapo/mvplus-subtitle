import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock,patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from translation_budget import TranslationBudget,BudgetExceeded
import translate_srt_batch as t


class AccountGuardTests(unittest.TestCase):
    def test_shared_confirmation_preserves_costs_and_ignores_other_clients_debits(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20');budget.observe_balance('100')
            budget.reserve('unknown response',1000,'deepseek-flash')
            with self.assertRaises(BudgetExceeded):budget.observe_balance('70')
            before=json.loads(budget.path.read_text())
            budget.confirm_shared_account('User confirmed other clients use this account')
            budget.observe_balance('60')
            after=json.loads(budget.path.read_text())
            self.assertEqual(before['requests'],after['requests'])
            self.assertEqual(before['limit_micro_cny'],after['limit_micro_cny'])
            self.assertEqual(before['account']['initial_micro_cny'],after['account']['initial_micro_cny'])
            self.assertEqual(budget.summary()['unknown_requests'],1)
            self.assertEqual(budget.summary()['account_mode'],'shared')
            self.assertIsNone(budget.summary()['halt_reason'])
            self.assertEqual(budget.summary()['observed_account_debit_cny'],40)
            with self.assertRaises(ValueError):budget.observe_balance('60',initial_balance='60')

    def test_shared_account_still_enforces_task_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','.01')
            budget.confirm_shared_account('User confirmation')
            budget.reserve('test',900,'deepseek-flash')
            budget.observe_balance('100')  # Existing shared ledger needs no exclusive baseline.
            with self.assertRaises(BudgetExceeded):budget.reserve('test',900,'deepseek-flash')
            self.assertEqual(budget.summary()['requests'],1)

    def test_confirmation_only_clears_account_attribution_halts(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20')
            reason='Missing/invalid API usage; retain reservations and inspect before resuming'
            budget.halt(reason)
            budget.confirm_shared_account('User confirmation')
            self.assertEqual(budget.summary()['halt_reason'],reason)
            with self.assertRaises(BudgetExceeded):budget.reserve('test',1000,'deepseek-flash')
            with self.assertRaises(ValueError):budget.confirm_shared_account(' ')

    def test_shared_account_unavailable_stops_before_paid_post(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20');budget.confirm_shared_account('User confirmation')
            response=MagicMock()
            response.__enter__.return_value.read.return_value=b'{"is_available":false}'
            with patch.object(t,'API_BUDGET',budget),patch.object(t,'_load_env_file'),\
                 patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test','DEEPSEEK_OFF_PEAK_ONLY':'0'}),\
                 patch.object(t.urllib.request,'urlopen',return_value=response) as network:
                with self.assertRaisesRegex(RuntimeError,'balance unavailable'):
                    t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            self.assertEqual(network.call_count,1)
            self.assertEqual(network.call_args.args[0].get_method(),'GET')
            self.assertEqual(budget.summary()['requests'],0)
            self.assertIsNotNone(budget.summary()['halt_reason'])

    def test_balance_observation_does_not_replace_another_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20');budget.observe_balance('100')
            budget.halt('API token usage exceeded reserved bounds')
            with self.assertRaises(BudgetExceeded):budget.observe_balance('60')
            budget.confirm_shared_account('User confirmation')
            self.assertEqual(budget.summary()['halt_reason'],'API token usage exceeded reserved bounds')

    def test_existing_ledger_cannot_silently_start_a_new_balance_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20')
            budget.reserve('old request',8192,'deepseek-flash')
            with self.assertRaises(BudgetExceeded):budget.observe_balance('84.40')
            self.assertEqual(budget.summary()['requests'],1)

    def test_unattributed_account_debit_halts_and_topup_does_not_erase_observed_debit(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20');budget.observe_balance('100.00')
            rid=budget.reserve('test',1000,'deepseek-flash')
            budget.settle(rid,dict(prompt_tokens=10,completion_tokens=20))
            with self.assertRaises(BudgetExceeded):budget.observe_balance('99.90')
            self.assertAlmostEqual(budget.summary()['observed_account_debit_cny'],.1)
            with self.assertRaises(BudgetExceeded):budget.observe_balance('200.00')
            self.assertAlmostEqual(budget.summary()['observed_account_debit_cny'],.1)
            with self.assertRaises(ValueError):budget.observe_balance('200.00',initial_balance='200.00')

    def test_valid_paid_response_is_retained_but_next_post_stops_on_balance_discrepancy(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);budget=TranslationBudget(work/'cost.json','20');balances=iter(['100.00','99.90'])
            calls=[]
            def request(req,**kwargs):
                calls.append(req.get_method());response=MagicMock();body=response.__enter__.return_value
                if req.get_method()=='GET':
                    body.read.return_value=json.dumps(dict(is_available=True,balance_infos=[dict(
                        currency='CNY',total_balance=next(balances))])).encode()
                else:
                    body.read1.side_effect=[json.dumps(dict(id='receipt',model='deepseek-flash',
                        choices=[dict(message=dict(content='OK'),finish_reason='stop')],
                        usage=dict(prompt_tokens=10,completion_tokens=20))).encode(),b'']
                return response
            with patch.object(t,'API_BUDGET',budget),patch.object(t,'_load_env_file'),\
                 patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test','DEEPSEEK_OFF_PEAK_ONLY':'0'}),\
                 patch.object(t.urllib.request,'urlopen',side_effect=request):
                self.assertEqual(t.call_ollama('test','deepseek-flash','https://api.deepseek.com',response_path=work/'reply.json'),'OK')
                with self.assertRaises(BudgetExceeded):t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            self.assertEqual(calls,['GET','POST','GET'])
            self.assertEqual(json.loads((work/'reply.json').read_text())['provider_request_id'],'receipt')
            recorded=next(iter(json.loads(budget.path.read_text())['requests'].values()))
            self.assertEqual(recorded['provider_request_id'],'receipt')
            self.assertIsNotNone(budget.summary()['halt_reason'])

    def test_unsupported_provider_is_rejected_before_sending_a_balance_credential(self):
        with patch.object(t.urllib.request,'urlopen') as network:
            with self.assertRaises(ValueError):t.call_ollama('test','deepseek-flash','https://api.openai.com')
            network.assert_not_called()

    def test_deepseek_does_not_fall_back_to_another_providers_credential(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20')
            with patch.object(t,'API_BUDGET',budget),patch.object(t,'_load_env_file'),\
                 patch.dict(os.environ,{'DEEPSEEK_API_KEY':'','OPENAI_API_KEY':'other-provider-test-only'}),\
                 patch.object(t.urllib.request,'urlopen') as network:
                with self.assertRaisesRegex(RuntimeError,'DEEPSEEK_API_KEY'):
                    t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
                network.assert_not_called()
