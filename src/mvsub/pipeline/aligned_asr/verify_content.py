"""Independent local ASR evidence for risky content, never a forced transcript."""
import json
import os
from pathlib import Path
import sys
import time
import hashlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asr_config import MODEL_HOME, ALIGN_LIBS
from aligned_asr.utterance import block_content_risks,model_identity,normalize
sys.path.insert(0,str(ALIGN_LIBS))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
from scipy.io import wavfile
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM


# Reuse existing evidence only for the identical prior model and input hashes.
LEGACY_VERIFIER_MODEL_IDENTITY='711A3B49D1C9C808C10809E765DDC5546F7E0F9D'
LEGACY_VERIFIER_CACHE_IDENTITY='6BDF19F7EA03CC757D956B03543BA6C147E98BBA'

def run(dest,audio,all_blocks=False):
    data=json.loads((dest/'asr.json').read_text('utf-8'))
    alignment=json.loads((dest/'alignment.json').read_text('utf-8')) if (dest/'alignment.json').exists() else {}
    unresolved={i['block'] for i in alignment.get('issues',[]) if i['type']=='unresolved_local_alignment'}
    unit_targets=[i for i in alignment.get('issues',[]) if i['type']=='unresolved_local_alignment']
    content_issues=[i for i in alignment.get('issues',[]) if i['type']=='unverified_content'
        and i['reason']=='low_confidence_speech']
    if content_issues:
        import nagisa
        from aligned_asr.utterance import make_units
        review_path=dest/'manual_audio_review.json'
        review=json.loads(review_path.read_text('utf-8')) if review_path.exists() else {}
        for issue in content_issues:
            units,_=make_units(data['blocks'][issue['block']],nagisa.tagging)
            units=[u for u in units if not any(r.get('human_verified') is True and
                r.get('classification')=='background_music_only' and r['start']<=u['start'] and u['end']<=r['end']
                for r in review.get('non_speech_regions',[]))]
            for ui in issue['units']:
                if ui<len(units):unit_targets.append(dict(units[ui],block=issue['block'],unit=ui))
    selected=[(i,b) for i,b in enumerate(data['blocks']) if all_blocks
        or set(block_content_risks(b))-{'long_native_word'}]
    if not selected and not unit_targets:
        (dest/'content_verification.json').write_text(json.dumps({'checks':[],'seconds':0}),encoding='utf-8');return
    model_path=MODEL_HOME/'models/qwen-asr-verifier'
    if not (model_path/'config.json').exists():raise RuntimeError('Local independent verifier missing; no auto-download')
    asr_sha=hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper()
    audio_sha=hashlib.sha1(audio.read_bytes()).hexdigest().upper()
    current_model_identity=model_identity(model_path)
    cache_identity=hashlib.sha1(Path(__file__).read_bytes()+current_model_identity.encode()).hexdigest().upper()
    compatible_cache_identities={cache_identity}
    if current_model_identity==LEGACY_VERIFIER_MODEL_IDENTITY:
        compatible_cache_identities.add(LEGACY_VERIFIER_CACHE_IDENTITY)
    previous=json.loads((dest/'content_verification.json').read_text('utf-8')) if (dest/'content_verification.json').exists() else {}
    if (previous.get('asr_sha1')!=asr_sha or previous.get('audio_sha1')!=audio_sha
        or previous.get('cache_identity') not in compatible_cache_identities):previous={}
    # An unchanged deterministic failure is cached as a failure, never as
    # positive evidence. Retrying the same long hallucination wastes minutes.
    old_blocks={c['block']:c for c in previous.get('checks',[])}
    old_units={(c['block'],c['unit']):c for c in previous.get('unit_checks',[])}
    sr,wav=wavfile.read(audio)
    if sr!=16000:raise ValueError('Expected 16kHz')
    wav=wav.astype(np.float32)/32768;started=time.perf_counter()
    processor=model=None
    def decode(left,right,limit):
        nonlocal processor,model
        if model is None:
            processor=AutoProcessor.from_pretrained(model_path,local_files_only=True)
            model=AutoModelForMultimodalLM.from_pretrained(model_path,dtype=torch.bfloat16,
                attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
        clip=wav[round(left*sr):round(right*sr)]
        for token_limit in (limit,limit*2):
            inputs=processor.apply_transcription_request(audio=clip,language='Japanese').to(model.device,model.dtype)
            with torch.inference_mode():ids=model.generate(**inputs,max_new_tokens=token_limit,do_sample=False)
            generated=ids[:,inputs['input_ids'].shape[1]:]
            parsed=processor.decode(generated,return_format='parsed')[0]
            parsed['requested_language']='Japanese'
            hit=generated.shape[1]>=token_limit
            if not hit:break
        return parsed,hit
    checks=[];unit_checks=[]
    skipped=[]
    def save():
        report=dict(model='Qwen/Qwen3-ASR-0.6B-hf',asr_sha1=asr_sha,audio_sha1=audio_sha,
            cache_identity=cache_identity,checks=checks,unit_checks=unit_checks,skipped_checks=skipped,
            seconds=time.perf_counter()-started,note='Independent local model estimates; not human hearing.',verification_scope='content_risk_blocks_and_required_units')
        temp=dest/'content_verification.part.json'
        temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(dest/'content_verification.json')
    def collapsed_loop(block):
        words=[w for s in block.get('segments',[]) for w in s.get('words',[]) if normalize(w['text'])]
        return ('repeated_lexical_loop' in block_content_risks(block) and len(words)>12
            and sum(w['end']-w['start']>.02 for w in words)/len(words)<.5)
    for bi,block in selected:
        if collapsed_loop(block):
            skipped.append(dict(block=bi,reason='collapsed_repetition_clock',stage='block'));save();continue
        cached=old_blocks.get(bi,{})
        if (cached.get('native_text')==block['text'] and cached.get('start')==block['start']
            and cached.get('end')==block['end'] and cached.get('independent',{}).get('requested_language')=='Japanese'):
            checks.append(cached);save();continue
        parsed,hit=decode(block['start'],block['end'],1024)
        # A forced-language prefix is part of INPUT, so generated-only parsing
        # normally has language=None. Record the actual request separately.
        parsed['requested_language']='Japanese'
        checks.append(dict(block=bi,start=block['start'],end=block['end'],native_text=block['text'],
            independent=parsed,hit_token_limit=hit))
        print('CONTENT_CHECK',bi,flush=True)
        save()
    seen=set()
    for issue in unit_targets:
        key=(issue['block'],issue['unit'])
        if key in seen:continue
        seen.add(key)
        bi=issue['block'];block=data['blocks'][bi]
        if collapsed_loop(block):
            skipped.append(dict(block=bi,unit=issue['unit'],reason='collapsed_repetition_clock',stage='unit'));save();continue
        left=max(block['start'],issue['start']-.35);right=min(block['end'],issue['end']+.35)
        if right-left<=.02:continue
        cached=old_units.get((bi,issue['unit']),{})
        if (cached.get('native_text')==issue['text'] and cached.get('unit_start')==issue['start']
            and cached.get('unit_end')==issue['end'] and cached.get('clip_start')==left
            and cached.get('clip_end')==right and cached.get('independent',{}).get('requested_language')=='Japanese'):
            unit_checks.append(cached);save();continue
        parsed,hit=decode(left,right,512)
        unit_checks.append(dict(block=bi,unit=issue['unit'],native_text=issue['text'],
            unit_start=issue['start'],unit_end=issue['end'],clip_start=left,clip_end=right,
            independent=parsed,hit_token_limit=hit))
        save()
    if model is not None:torch.cuda.synchronize()
    save()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),'--all' in sys.argv)
