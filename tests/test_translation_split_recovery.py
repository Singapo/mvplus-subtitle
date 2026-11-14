import sys
from pathlib import Path
from unittest.mock import patch
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import translate_srt_batch as t

class SplitRecoveryTests(unittest.TestCase):
    def test_split_preserves_ids_order_and_source_context(self):
        entries=[(str(i),'0','1',f'原文{i}') for i in range(500,628)]
        contexts=[]
        def response(batch,context,*args,**kwargs):
            contexts.append(list(context))
            if len(batch)>32:raise ValueError('Missing subtitle IDs')
            return [f'中文{x[0]}' for x in batch],0
        with patch.object(t,'translate_batch',side_effect=response) as request:
            texts,hits=t.translate_batch_with_splits(entries,['499. 前文'],None,'test','local','ja','zh-cn')
        self.assertEqual(texts,[f'中文{i}' for i in range(500,628)])
        self.assertEqual(request.call_count,7)
        self.assertEqual(hits,0)
        self.assertTrue(any('531. 原文531' in context for context in contexts))
        self.assertFalse(any('中文' in line for context in contexts for line in context))
    def test_persistent_failure_is_bounded_and_not_filled(self):
        entries=[(str(i),'0','1','原文') for i in range(128)]
        with patch.object(t,'translate_batch',side_effect=ValueError('Unnumbered response')) as request:
            with self.assertRaises(ValueError):t.translate_batch_with_splits(entries,[],None,'test','local','ja','zh-cn')
        self.assertEqual(request.call_count,3)
    def test_network_failure_is_not_blindly_split(self):
        entries=[(str(i),'0','1','原文') for i in range(128)]
        with patch.object(t,'translate_batch',side_effect=TimeoutError('network')) as request:
            with self.assertRaises(TimeoutError):t.translate_batch_with_splits(entries,[],None,'test','local','ja','zh-cn')
        request.assert_called_once()
