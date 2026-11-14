import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import translate_srt_batch as t
import reprocess_subtitles as p
import nosub_pipeline as n
from translation_budget import TranslationBudget,BudgetExceeded


class PaidWorkflowGuards(unittest.TestCase):
    def test_imported_client_cannot_bypass_budget(self):
        with patch.object(t,'API_BUDGET',None),patch.object(t.urllib.request,'urlopen') as network:
            with self.assertRaisesRegex(RuntimeError,'budget'):t.call_ollama('test','deepseek-flash','https://api.deepseek.com')
            network.assert_not_called()

    def test_reprocess_requires_budget_before_any_cloud_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan=Path(tmp)/'plan.json'
            plan.write_text(json.dumps(dict(items=[dict(subtitle=dict(fid='1',n='A.srt'),video=dict(n='A.mp4'))])))
            with patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),patch.object(p,'Cloud115') as cloud:
                with self.assertRaisesRegex(ValueError,'budget'):p.run(plan)
                cloud.assert_not_called()

    def test_nosub_rejects_unbudgeted_start_before_env_or_cloud(self):
        with patch.object(n,'load_env') as env:
            self.assertEqual(n.cmd_run(SimpleNamespace(budget_file=None,budget_cny=None)),2)
            env.assert_not_called()

    def test_resume_keeps_unknown_charges_and_does_not_reset_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget=TranslationBudget(Path(tmp)/'cost.json','20')
            budget.reserve('request',8192,'deepseek-flash');before=budget.summary()['conservative_spent_or_reserved_cny']
            budget.halt('unknown response')
            with self.assertRaises(BudgetExceeded):budget.reserve('another',8192,'deepseek-flash')
            budget.resume('Explicit controlled continuation after checking unknown reservation')
            self.assertEqual(budget.summary()['conservative_spent_or_reserved_cny'],before)
            self.assertEqual(budget.summary()['limit_cny'],20)

    def test_foreign_reference_facts_are_rejected_before_network_even_if_targets_match(self):
        from translate_srt_batch import write_srt
        from source_review import make_report,subset_report
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);full=work/'full.srt';part=work/'part.srt';reference=work/'reference.srt';audio=work/'audio.wav'
            one=('1','00:00:01,000','00:00:02,000','こんにちは')
            two=('2','00:00:03,000','00:00:04,000','おかえり')
            write_srt(full,[one,two]);write_srt(part,[one]);audio.write_bytes(b'synthetic')
            write_srt(reference,[one,(*two[:3],'別の映画の話')])
            make_report(full,audio,work/'full.json');subset_report(work/'full.json',part,work/'part.json')
            argv=['translate',str(part),'--grounded','--model','deepseek-flash','--url','https://api.deepseek.com',
                  '--budget-file',str(work/'cost.json'),'--budget-cny','20',
                  '--reference-srt',str(reference),'--source-review',str(work/'part.json')]
            with patch.object(sys,'argv',argv),patch.object(t,'API_BUDGET',None),\
                 patch.object(t,'API_MAX_TOKENS',8192),patch.object(t,'API_REASONING_EFFORT',None),\
                 patch.object(t,'REFERENCE_TEXT',''),patch.object(t.urllib.request,'urlopen') as network:
                with self.assertRaises(ValueError):t.main()
                network.assert_not_called()

    def test_known_context_hold_is_preserved_before_any_source_or_generation_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);job=work/'jobs/1';job.mkdir(parents=True)
            (job/'translation_content_hold.json').write_text('{}')
            plan=work/'plan.json';plan.write_text(json.dumps(dict(items=[dict(
                subtitle=dict(fid='1',n='A.srt'),video=dict(n='A.mp4'))])))
            (work/'state.json').write_text(json.dumps(dict(items={'1':dict(status='blocked_content',error='retain evidence')})))
            with patch.object(p,'load_env',return_value={'P115_COOKIES':'fake'}),\
                 patch.object(p,'Cloud115') as cloud,patch.object(p.subprocess,'run') as process,\
                 patch('source_review.ensure_review') as acoustic:
                state=p.run(plan,budget_file=work/'cost.json',budget_cny='20')
                process.assert_not_called();acoustic.assert_not_called()
                cloud.return_value.extract_audio.assert_not_called()
            self.assertEqual(state['items']['1']['status'],'blocked_content')
            self.assertEqual(state['items']['1']['error'],'retain evidence')
            self.assertFalse(state['batch_complete'])
