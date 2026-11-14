import http.client
import json
import os
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch
import urllib.error
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from translation_budget import TranslationBudget
import translate_srt_batch as t


class NetworkRetryTests(unittest.TestCase):
    def setUp(self):
        self.work=tempfile.TemporaryDirectory();self.addCleanup(self.work.cleanup)
        active=patch.object(t,'API_BUDGET',TranslationBudget(Path(self.work.name)/'cost.json','20'))
        active.start();self.addCleanup(active.stop)
        account=patch.object(t,'check_account_balance');account.start();self.addCleanup(account.stop)

    def test_truncated_paid_response_stops_without_resending_and_keeps_discount_gate(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-only','DEEPSEEK_OFF_PEAK_ONLY':'1'}),\
             patch.object(t,'_load_env_file'),patch('deepseek_offpeak.wait_offpeak') as gate,\
             patch.object(t.time,'sleep'),patch.object(t.urllib.request,'urlopen') as request:
            response=request.return_value.__enter__.return_value
            response.read1.side_effect=http.client.IncompleteRead(b'partial')
            with self.assertRaises(http.client.IncompleteRead):t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            self.assertEqual(request.call_count,1);self.assertEqual(gate.call_count,1)
            self.assertEqual(json.loads(request.call_args.args[0].data)['thinking'],{'type':'enabled'})

    def test_network_retries_are_bounded_and_bad_credentials_are_not_retried(self):
        with patch.dict(os.environ,{'DEEPSEEK_API_KEY':'test-only','DEEPSEEK_OFF_PEAK_ONLY':'0'}),\
             patch.object(t,'_load_env_file'),patch.object(t.time,'sleep'),\
             patch.object(t.urllib.request,'urlopen',side_effect=http.client.RemoteDisconnected('closed')) as request:
            with self.assertRaises(http.client.RemoteDisconnected):t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            self.assertEqual(request.call_count,1)
            request.reset_mock();request.side_effect=urllib.error.HTTPError('https://api.deepseek.com',401,'bad key',{},None)
            with patch.object(t,'API_BUDGET',TranslationBudget(Path(self.work.name)/'auth.json','20')):
                with self.assertRaises(urllib.error.HTTPError):t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            request.assert_called_once()
