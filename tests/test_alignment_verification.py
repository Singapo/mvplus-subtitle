"""Exercise alignment orchestration without GPU/model inference."""
import contextlib
import hashlib
import importlib.util
import io
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


class AlignmentVerificationTests(unittest.TestCase):
    def test_bad_next_native_boundary_cannot_silently_delete_clear_dialogue(self):
        align=self.load_align()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);audio=dest/'full_audio.wav'
            wavfile.write(audio,16000,np.zeros(5*16000,dtype=np.int16))
            first=dict(start=1.,end=3.,owned_end=3.,sample='test',text='おはよう',segments=[dict(text='おはよう',
                avg_logprob=0,no_speech_prob=0,words=[dict(text='おはよう',start=.5,end=1.,probability=.99)])])
            second=dict(start=2.,end=4.,owned_end=4.,sample='test',text='次です',segments=[dict(text='次です',
                avg_logprob=0,no_speech_prob=0,words=[dict(text='次です',start=-1.,end=0.,probability=.99)])])
            (dest/'asr.json').write_text(json.dumps(dict(blocks=[first,second])),'utf-8')
            with contextlib.redirect_stdout(io.StringIO()):align.run(dest,audio,False)
            result=json.loads((dest/'alignment.json').read_text('utf-8'))
            self.assertTrue(any(r['type']=='cross_block_source_conflict' and r['text']=='おはよう' for r in result['issues']))
            self.assertFalse(any(r['text']=='おはよう' for r in result['omissions']))

    def test_recovered_fragment_cannot_start_after_the_next_block(self):
        align=self.load_align()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);audio=dest/'full_audio.wav'
            wavfile.write(audio,16000,np.zeros(5*16000,dtype=np.int16))
            block=lambda a,t:dict(start=a,end=a+2.,owned_end=a+2.,sample='test',text=t,
                segments=[dict(text=t,avg_logprob=0,no_speech_prob=0,
                    words=[dict(text=t,start=0.,end=0.,probability=.9)])])
            (dest/'asr.json').write_text(json.dumps(dict(blocks=[block(1.,'ち'),block(2.,'次です')])),'utf-8')
            evidence=dict(asr_sha1=hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper(),
                audio_sha1=hashlib.sha1(audio.read_bytes()).hexdigest().upper(),recoveries=[dict(block=0,unit=0,
                    source_text='ち',source_start=1.,source_end=1.,
                    method='two_model_contextual_native_clock',words=[dict(text='ち',start=2.1,end=2.2)])])
            (dest/'word_recovery.json').write_text(json.dumps(evidence),'utf-8')
            with contextlib.redirect_stdout(io.StringIO()):align.run(dest,audio,False)
            result=json.loads((dest/'alignment.json').read_text('utf-8'))
            self.assertEqual(result['blocks'][0]['words'],[])
            self.assertTrue(any(r['reason']=='very_low_confidence_cross_block_location' for r in result['omissions']))

    def load_align(self):
        processor=SimpleNamespace(
            prepare_forced_aligner_inputs=lambda **kw:(Inputs(input_ids=[]),[]),
            decode_forced_alignment=lambda **kw:[[]])
        model=SimpleNamespace(device='cpu',dtype=None,config=SimpleNamespace(timestamp_token_id=1))
        model.to=lambda *args:model
        model.eval=lambda:model
        class Model:
            @classmethod
            def from_pretrained(cls,*args,**kwargs):return cls()
            def to(self,*args):return self
            def eval(self):return self
            device=model.device;dtype=model.dtype;config=model.config
            def __call__(self,**kwargs):return SimpleNamespace(logits=[])
        factory=SimpleNamespace(from_pretrained=lambda *a,**kw:processor)
        torch=SimpleNamespace(bfloat16=None,cuda=SimpleNamespace(synchronize=lambda:None),
            inference_mode=contextlib.nullcontext)
        tagging=lambda text:SimpleNamespace(words=[text],postags=['名詞'])
        spec=importlib.util.spec_from_file_location('test_local_align',TOOLS/'aligned_asr/align.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'torch':torch,'nagisa':SimpleNamespace(tagging=tagging),
            'transformers':SimpleNamespace(AutoProcessor=factory,AutoModelForTokenClassification=Model)}):
            old_path=list(sys.path)
            try:spec.loader.exec_module(module)
            finally:sys.path[:]=old_path
        return module

    def test_alignment_partial_human_review_cannot_release_other_repeats(self):
        align=self.load_align()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);audio=dest/'full_audio.wav'
            wavfile.write(audio,16000,np.zeros(5*16000,dtype=np.int16))
            text='おはようございます'
            segments=[dict(text=text,avg_logprob=-2,no_speech_prob=.9,
                words=[dict(text=text,start=float(i+1),end=float(i+1)+.5)]) for i in range(3)]
            block=dict(start=0.,end=5.,owned_end=5.,sample='test',text=text*3,segments=segments)
            (dest/'asr.json').write_text(json.dumps(dict(blocks=[block])))
            reviews=[dict(human_verified=True,source_text=text,source_start=float(i+1),
                source_end=float(i+1)+.5,text=text,start=float(i+1),end=float(i+1)+.5) for i in range(3)]
            for count in (2,3):
                (dest/'manual_audio_review.json').write_text(json.dumps(dict(utterance_confirmations=reviews[:count])))
                with contextlib.redirect_stdout(io.StringIO()):align.run(dest,audio,True)
                result=json.loads((dest/'alignment.json').read_text('utf-8'))
                if count==2:
                    risks=[i for i in result['issues'] if i['type']=='unverified_content']
                    self.assertEqual(len(risks),2)
                    self.assertTrue(all(r['units']==[2] for r in risks))
                else:
                    self.assertEqual(result['issues'],[])
                    self.assertTrue(all(w['clock']=='human_confirmed' for w in result['blocks'][0]['words']))

    def test_two_runs_regenerate_and_retain_local_fallback_evidence(self):
        align=self.load_align()
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp);audio=dest/'full_audio.wav'
            wavfile.write(audio,16000,np.zeros(3*16000,dtype=np.int16))
            seg=dict(text='そうですね',words=[dict(text='そうですね',start=1.,end=2.)])
            asr=dict(blocks=[dict(start=0.,end=3.,owned_end=3.,sample='test',text=seg['text'],segments=[seg])])
            (dest/'asr.json').write_text(json.dumps(asr))
            audio_sha=hashlib.sha1(audio.read_bytes()).hexdigest().upper()
            (dest/'vad_evidence.json').write_text(json.dumps(dict(audio_sha1=audio_sha,speech_regions=[dict(start=1.,end=2.)])))
            evidence=dict(asr_sha1=hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper(),
                audio_sha1=audio_sha,checks=[],unit_checks=[dict(block=0,unit=0,native_text=seg['text'],
                    unit_start=1.,unit_end=2.,clip_start=.65,clip_end=2.35,hit_token_limit=False,
                    independent=dict(requested_language='Japanese',transcription=seg['text']))])
            verify=dest/'content_verification.json'
            for _ in range(2):
                verify.write_text(json.dumps(evidence))
                with contextlib.redirect_stdout(io.StringIO()):align.run(dest,audio,True)
                first=json.loads((dest/'alignment.json').read_text('utf-8'))
                self.assertEqual(first['issues'][0]['type'],'unresolved_local_alignment')
                # The verifier sees the unresolved sentence on each run.
                self.assertEqual((first['issues'][0]['block'],first['issues'][0]['unit']),(0,0))
                verify.write_text(json.dumps(evidence))
                with contextlib.redirect_stdout(io.StringIO()):align.run(dest,audio)
                final=json.loads((dest/'alignment.json').read_text('utf-8'))
                self.assertEqual(final['issues'],[])
                self.assertEqual(final['fallbacks'][0]['clock'],'independent_text_native_speech_interval')
            evidence['audio_sha1']='STALE'
            verify.write_text(json.dumps(evidence))
            with contextlib.redirect_stdout(io.StringIO()):align.run(dest,audio)
            stale=json.loads((dest/'alignment.json').read_text('utf-8'))
            self.assertTrue(stale['omissions'])
            self.assertFalse(any(w['clock']=='independent_text_native_speech_interval' for b in stale['blocks'] for w in b['words']))


if __name__=='__main__':unittest.main()
