import sys
from pathlib import Path
import queue
import tempfile
from unittest.mock import patch
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import nosub_pipeline as p

class FailedStageTests(unittest.TestCase):
    def test_failed_extraction_still_closes_translation_and_upload(self):
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp);state=p.State(work/'state.json')
            source=queue.Queue();target=queue.Queue();uploads=queue.Queue()
            source.put({'stem':'A'});source.put(None)
            with patch.object(p,'do_extract',side_effect=RuntimeError('failed')):
                p._extract_loop(source,work,state,target,p.StageGate(1,'extract'),1)
            with patch.object(p,'do_translate') as translate:
                p._translate_loop(target,work,state,uploads,p.StageGate(1,'translate'),1)
                translate.assert_not_called()
            self.assertIsNone(uploads.get_nowait())
            self.assertEqual(state.get('A')['status'],'skipped_extract_failed')
