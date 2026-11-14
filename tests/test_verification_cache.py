import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile

TOOLS=Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'
sys.path.insert(0,str(TOOLS))


class Inputs(dict):
    def to(self,*args):return self


class VerificationCacheTests(unittest.TestCase):
    def test_valid_evidence_survives_repeated_fresh_alignment_selection(self):
        loaded=[]
        processor=SimpleNamespace(apply_transcription_request=lambda **k:Inputs(input_ids=np.zeros((1,2))),
            decode=lambda *a,**k:[dict(language=None,transcription='そうですね')])
        class Model:
            def to(self,*a):return self
            def eval(self):return self
            device='cpu';dtype=None
            def generate(self,**kw):return np.zeros((1,3))
        def load(*a,**kw):loaded.append(1);return Model()
        torch=SimpleNamespace(bfloat16=None,cuda=SimpleNamespace(synchronize=lambda:None),
            inference_mode=contextlib.nullcontext)
        spec=importlib.util.spec_from_file_location('test_verifier',TOOLS/'aligned_asr/verify_content.py')
        verifier=importlib.util.module_from_spec(spec)
        old_path=list(sys.path)
        with patch.dict(sys.modules,{'torch':torch,'transformers':SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=lambda *a,**k:processor),
            AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=load))}):
            try:spec.loader.exec_module(verifier)
            finally:sys.path[:]=old_path
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);audio=dest/'full_audio.wav'
            wavfile.write(audio,16000,np.zeros(3*16000,dtype=np.int16))
            model_dir=dest/'models/qwen-asr-verifier';model_dir.mkdir(parents=True)
            (model_dir/'config.json').write_text('{}')
            (dest/'asr.json').write_text(json.dumps(dict(blocks=[dict(start=0.,end=3.,text='そうですね',segments=[])])))
            issue=dict(block=0,unit=0,type='unresolved_local_alignment',text='そうですね',start=1.,end=2.)
            (dest/'alignment.json').write_text(json.dumps(dict(issues=[issue])))
            with patch.object(verifier,'MODEL_HOME',dest):
                verifier.run(dest,audio)
                verifier.run(dest,audio)
                self.assertEqual(len(loaded),1)
                evidence=json.loads((dest/'content_verification.json').read_text('utf-8'))
                self.assertEqual(len(evidence['unit_checks']),1)
                self.assertEqual(evidence['audio_sha1'],hashlib.sha1(audio.read_bytes()).hexdigest().upper())
                wavfile.write(audio,16000,np.ones(3*16000,dtype=np.int16))
                verifier.run(dest,audio)
                self.assertEqual(len(loaded),2)
                evidence=json.loads((dest/'content_verification.json').read_text('utf-8'))
                evidence['cache_identity']='obsolete';(dest/'content_verification.json').write_text(json.dumps(evidence))
                verifier.run(dest,audio)
                self.assertEqual(len(loaded),3)


if __name__=='__main__':unittest.main()
