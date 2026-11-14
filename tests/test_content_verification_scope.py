"""No-GPU integration checks for the staged verifier scope change."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

TOOLS=Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'
sys.path.insert(0,str(TOOLS))
DEFAULT=TOOLS/'aligned_asr/verify_content.py'


class Inputs(dict):
    def to(self,*args):return self


class Tokens:
    shape=(1,1)
    def __getitem__(self,key):return self


def load():
    calls=[]
    processor=SimpleNamespace(apply_transcription_request=lambda **kw:Inputs(input_ids=Tokens()),
        decode=lambda *a,**kw:[dict(language=None,transcription='テストです')])
    class Model:
        device='cpu';dtype=None
        def to(self,*args):return self
        def eval(self):return self
        def generate(self,**kw):calls.append(kw['max_new_tokens']);return Tokens()
    torch=SimpleNamespace(bfloat16=None,cuda=SimpleNamespace(synchronize=lambda:None),inference_mode=contextlib.nullcontext)
    factory=SimpleNamespace(from_pretrained=lambda *a,**kw:processor)
    model_factory=SimpleNamespace(from_pretrained=lambda *a,**kw:Model())
    path=Path(os.environ.get('SUBTITLE_VERIFY_IMPL',str(DEFAULT)))
    spec=importlib.util.spec_from_file_location('verification_scope_candidate',path);module=importlib.util.module_from_spec(spec)
    old_path=list(sys.path)
    with patch.dict(sys.modules,{'torch':torch,'transformers':SimpleNamespace(
        AutoProcessor=factory,AutoModelForMultimodalLM=model_factory)}):
        try:spec.loader.exec_module(module)
        finally:sys.path[:]=old_path
    return module,calls


def setup(tmp_path,risky=False,clock=True):
    module,calls=load();model_home=tmp_path/'models';(model_home/'models/qwen-asr-verifier').mkdir(parents=True)
    (model_home/'models/qwen-asr-verifier/config.json').write_text('{}',encoding='utf-8')
    module.MODEL_HOME=model_home;module.model_identity=lambda path:'test-model'
    audio=tmp_path/'full_audio.wav';wavfile.write(audio,16000,np.zeros(4*16000,dtype=np.int16))
    block=dict(start=0.,end=4.,text='テストです',segments=[dict(text='テストです',
        avg_logprob=-2 if risky else 0,no_speech_prob=.9 if risky else 0,
        words=[dict(text='テストです',start=1.,end=2.,probability=.9)])])
    (tmp_path/'asr.json').write_text(json.dumps(dict(blocks=[block])),encoding='utf-8')
    issues=[dict(type='unresolved_local_alignment',block=0,unit=0,start=1.,end=2.,text='テストです')] if clock else []
    (tmp_path/'alignment.json').write_text(json.dumps(dict(issues=issues)),encoding='utf-8')
    return module,calls,audio


def report(path):return json.loads((path/'content_verification.json').read_text(encoding='utf-8'))


def test_alignment_only_keeps_unit_evidence_without_whole_block_decode(tmp_path):
    module,calls,audio=setup(tmp_path)
    module.run(tmp_path,audio)
    assert calls==[128]
    assert report(tmp_path)['checks']==[] and len(report(tmp_path)['unit_checks'])==1


def test_real_content_risk_keeps_whole_block_and_unit_checks(tmp_path):
    module,calls,audio=setup(tmp_path,risky=True)
    module.run(tmp_path,audio)
    assert calls==[256,128]
    assert len(report(tmp_path)['checks'])==len(report(tmp_path)['unit_checks'])==1


def test_explicit_all_blocks_still_checks_every_block(tmp_path):
    module,calls,audio=setup(tmp_path)
    module.run(tmp_path,audio,all_blocks=True)
    assert calls==[256,128]


def test_no_targets_loads_no_model(tmp_path):
    module,calls,audio=setup(tmp_path,clock=False)
    module.run(tmp_path,audio)
    assert calls==[] and report(tmp_path)['checks']==[]


def cached_unit(module,tmp_path,audio):
    return dict(asr_sha1=hashlib.sha1((tmp_path/'asr.json').read_bytes()).hexdigest().upper(),
        audio_sha1=hashlib.sha1(audio.read_bytes()).hexdigest().upper(),
        cache_identity=module.LEGACY_VERIFIER_CACHE_IDENTITY,checks=[],
        unit_checks=[dict(block=0,unit=0,native_text='テストです',unit_start=1.,unit_end=2.,
            clip_start=.65,clip_end=2.35,hit_token_limit=False,
            independent=dict(language=None,transcription='テストです',requested_language='Japanese'))])


def test_matching_legacy_unit_evidence_is_reused_without_model_call(tmp_path):
    module,calls,audio=setup(tmp_path);module.model_identity=lambda path:module.LEGACY_VERIFIER_MODEL_IDENTITY
    cached=cached_unit(module,tmp_path,audio)
    (tmp_path/'content_verification.json').write_text(json.dumps(cached),encoding='utf-8')
    module.run(tmp_path,audio)
    assert calls==[] and report(tmp_path)['unit_checks']==cached['unit_checks']


def test_changed_model_does_not_reuse_legacy_evidence(tmp_path):
    module,calls,audio=setup(tmp_path)
    (tmp_path/'content_verification.json').write_text(json.dumps(cached_unit(module,tmp_path,audio)),encoding='utf-8')
    module.run(tmp_path,audio)
    assert calls==[128]
