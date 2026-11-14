"""Align dialogue inside its native utterance; retain sound events locally."""
import json
import math
import os
from pathlib import Path
import sys
import time
import hashlib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asr_config import ALIGN_MODEL, ALIGN_LIBS
sys.path.insert(0,str(ALIGN_LIBS))
from aligned_asr.utterance import POLICY, CONTENT_POLICY, make_units, map_aligned_words, unconfirmed_content_risks, human_confirmation, verified_native_interval, sound_omission_indices
from aligned_asr.acoustic import locate_sound
from aligned_asr.recovery import vocal_event_supported,nonvocal_background_supported,native_clock_supported,very_low_content_confidence
from aligned_asr.timing import normalize
from aligned_asr.clock_order import reconcile,zero_phonetic_fragment
os.environ['HF_HUB_OFFLINE']='1'
import nagisa
import numpy as np
from scipy.io import wavfile
import torch
from transformers import AutoProcessor, AutoModelForTokenClassification


def run(dest,audio,fresh_verification=False):
    asr=json.loads((dest/'asr.json').read_text(encoding='utf-8'))
    sr,wav=wavfile.read(audio)
    if sr!=16000 or wav.ndim!=1:raise ValueError('Expected mono 16kHz PCM')
    duration=len(wav)/sr;wav=wav.astype(np.float32)/32768
    verification_path=dest/'content_verification.json'
    verification=json.loads(verification_path.read_text('utf-8')) if not fresh_verification and verification_path.exists() else {}
    verified_input=(verification.get('asr_sha1')==hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper()
                    and verification.get('audio_sha1')==hashlib.sha1(audio.read_bytes()).hexdigest().upper())
    independent={c['block']:c['independent']['transcription'] for c in verification.get('checks',[]) if verified_input
                 if (c['independent'].get('language')=='Japanese' or
                     c['independent'].get('requested_language')=='Japanese') and not c.get('hit_token_limit')}
    unit_checks={(c['block'],c['unit']):c for c in verification.get('unit_checks',[]) if verified_input
                 and c['independent'].get('requested_language')=='Japanese' and not c.get('hit_token_limit')}
    vad=json.loads((dest/'vad_evidence.json').read_text('utf-8')) if (dest/'vad_evidence.json').exists() else {}
    if vad.get('audio_sha1')!=hashlib.sha1(audio.read_bytes()).hexdigest().upper():vad={}
    review=json.loads((dest/'manual_audio_review.json').read_text('utf-8')) if (dest/'manual_audio_review.json').exists() else {}
    non_speech=[r for r in review.get('non_speech_regions',[]) if r.get('human_verified') is True
                and r.get('classification')=='background_music_only']
    recovery={};event_checks={};background_checks={}
    if not fresh_verification:
        asr_sha=hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper()
        audio_sha=hashlib.sha1(audio.read_bytes()).hexdigest().upper()
        for name,key in [('word_recovery.json','recoveries'),('event_verification.json','checks')]:
            path=dest/name
            evidence=json.loads(path.read_text('utf-8')) if path.exists() else {}
            if evidence.get('asr_sha1')!=asr_sha or evidence.get('audio_sha1')!=audio_sha:continue
            target=recovery if key=='recoveries' else event_checks
            for r in evidence.get(key,[]):
                dest_checks=background_checks if r.get('kind')=='background' else target
                dest_checks[r['block'],r['unit']]=r
    started=time.perf_counter()
    processor=AutoProcessor.from_pretrained(ALIGN_MODEL,local_files_only=True)
    model=AutoModelForTokenClassification.from_pretrained(ALIGN_MODEL,dtype=torch.bfloat16,
        attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
    torch.cuda.synchronize();load=time.perf_counter()-started
    started=time.perf_counter();blocks=[];issues=[];decisions=[];omissions=[];warnings=[];order_repairs=[]
    for bi,block in enumerate(asr['blocks']):
        boundary=duration
        if bi+1<len(asr['blocks']) and asr['blocks'][bi+1]['sample']==block['sample']:
            next_block=asr['blocks'][bi+1]
            starts=[next_block['start']+w['start'] for s in next_block.get('segments',[])
                for w in s.get('words',[]) if normalize(w['text'])]
            boundary=min(duration,next_block['start'],min(starts,default=duration))
        units,errors=make_units(block,nagisa.tagging);word_rows=[]
        original_units=units
        units=[]
        for unit in original_units:
            if any(r['start']<=unit['start'] and unit['end']<=r['end'] for r in non_speech):
                omissions.append(dict(block=bi,reason='human_confirmed_no_voice',**unit))
            else:units.append(unit)
        issues.extend(dict(block=bi,**e) for e in errors)
        # Remove sound-only units before acoustic localization, recovery and
        # subtitle export. Keep their slots so cached evidence indexes and
        # adjacent dialogue boundaries retain the original ownership.
        resolved=sound_omission_indices(units,review)
        omissions.extend(dict(block=bi,unit=ui,reason='non_dialogue_sound_filtered',**units[ui])
                         for ui in sorted(resolved))
        for ui,unit in enumerate(units):
            if ui in resolved:continue
            check=background_checks.get((bi,ui),{})
            if (check.get('source_text')==unit['text'] and check.get('start')==unit['start']
                and check.get('end')==unit['end'] and nonvocal_background_supported(check)):
                resolved.add(ui)
                omissions.append(dict(block=bi,unit=ui,reason='independent_nonvocal_background',**unit))
        for risk in unconfirmed_content_risks(block,units,review,resolved):
            if normalize(block['text']) and normalize(block['text']) in normalize(independent.get(bi,'')):
                decisions.append(dict(block=bi,clock='independent_content_supported',**risk));continue
            if fresh_verification:
                issues.append(dict(block=bi,type='unverified_content',**risk))
            else:
                for ui in risk['units']:
                    unit=units[ui]
                    check=unit_checks.get((bi,ui),{})
                    local_text=''
                    if (check.get('native_text')==unit['text'] and check.get('unit_start')==unit['start']
                        and check.get('unit_end')==unit['end']
                        and check.get('clip_start',-1)>=unit['start']-.351
                        and check.get('clip_end',float('inf'))<=unit['end']+.351):
                        local_text=check['independent']['transcription']
                    if very_low_content_confidence(unit,risk['reason'],local_text):
                        if ui not in resolved:
                            omissions.append(dict(block=bi,unit=ui,reason='very_low_confidence_content',risk=risk['reason'],**unit))
                        resolved.add(ui)
                    else:warnings.append(dict(block=bi,unit=ui,type='content_disagreement_source_retained',reason=risk['reason'],text=unit['text']))
        for ui,unit in enumerate(units):
            if ui in resolved:continue
            owned=[i for i,w in enumerate(unit['words']) if (w['start']+w['end'])/2<block['owned_end']
                   or block['owned_end']>=block['end']]
            if not owned:continue
            mapped=unit['words'];method='native_sound';error=None
            confirmed=human_confirmation(unit,review)
            if confirmed:
                a,b=confirmed['start'],confirmed['end']
                if not math.isfinite(a+b) or not 0<=a<b<=duration:raise ValueError('Invalid human-confirmed utterance bounds')
                word_rows.append(dict(start=a,end=b,text=confirmed['text'],kind=unit['kind'],
                    utterance=f'{bi}:{unit["segment"]}:{ui}',utterance_end=True,clock='human_confirmed'))
                decisions.append(dict(block=bi,unit=ui,clock='human_confirmed',source_text=unit['text']))
                continue
            if unit['start']>=boundary:
                if zero_phonetic_fragment(unit):
                    omissions.append(dict(block=bi,unit=ui,reason='unowned_cross_block_fragment',**unit));continue
                issues.append(dict(block=bi,unit=ui,type='cross_block_source_conflict',boundary=boundary,**unit));continue
            if unit['kind']=='sound' and unit['end']-unit['start']<=.02:
                left=block['start'] if not ui else max(block['start'],units[ui-1]['end'])
                right=min(boundary,block['end']) if ui+1==len(units) else min(boundary,block['end'],units[ui+1]['start'])
                found=locate_sound(wav,sr,unit['start'],left,right)
                if found:
                    mapped=[dict(start=found[0],end=found[1],text=unit['text'])]
                    method='measured_native_sound_event'
                    event=event_checks.get((bi,ui),{})
                    if (event.get('source_text')==unit['text'] and event.get('start')==found[0]
                        and event.get('end')==found[1] and vocal_event_supported(event)):
                        method='independently_classified_native_sound_event'
                    else:
                        if not fresh_verification:
                            omissions.append(dict(block=bi,unit=ui,reason='very_low_confidence_sound_location',**unit));continue
                        issues.append(dict(block=bi,unit=ui,type='unverified_acoustic_sound_event',
                            start=found[0],end=found[1],text=unit['text'],reason='Measured energy is not proof of a nonverbal voice'))
                elif not fresh_verification:
                    omissions.append(dict(block=bi,unit=ui,reason='unlocated_sound',**unit));continue
            if unit['kind']=='dialogue':
                method='local_qwen'
                if unit['end']-unit['start']>30:
                    error='long_native_utterance'
                else:
                    for pad in (.35,.8):
                        left=max(block['start'],unit['start']-pad)
                        right=min(block['end'],boundary,unit['end']+pad)
                        if ui:left=max(left,units[ui-1]['end'])
                        if ui+1<len(units):right=min(right,units[ui+1]['start'])
                        if right-left<=.02:error='unlocated_native_utterance';break
                        clip=wav[round(left*sr):round(right*sr)]
                        inputs,words=processor.prepare_forced_aligner_inputs(audio=clip,transcript=unit['text'],language='Japanese')
                        inputs=inputs.to(model.device,model.dtype)
                        with torch.inference_mode():result=model(**inputs)
                        aligned=processor.decode_forced_alignment(logits=result.logits,input_ids=inputs['input_ids'],
                            word_lists=words,timestamp_token_id=model.config.timestamp_token_id)[0]
                        candidate,error=map_aligned_words(aligned,unit,left,right)
                        if not error:mapped=candidate;break
                if error:
                    r=recovery.get((bi,ui),{})
                    if (r.get('source_text')==unit['text'] and r.get('source_start')==unit['start']
                        and r.get('source_end')==unit['end'] and r.get('words')
                        and normalize(''.join(w['text'] for w in r['words']))==normalize(unit['text'])
                        and (not ui or min(w['start'] for w in r['words'])>=units[ui-1]['start'])
                        and (ui+1==len(units) or min(w['start'] for w in r['words'])<=units[ui+1]['start'])):
                        method=r.get('method','two_model_contextual_native_clock')
                        if method=='contextual_redecode_measured_speech':
                            mapped=[dict(text=unit['text'],start=min(w['start'] for w in r['words']),
                                end=max(w['end'] for w in r['words']))];error=None
                        else:
                            mapped,error=map_aligned_words([dict(text=w['text'],start_time=w['start'],
                                end_time=w['end']) for w in r['words']],unit,0,duration)
                        if not error and method=='contextual_redecode_measured_speech':
                            warnings.append(dict(block=bi,unit=ui,type='source_retained_with_context_and_speech_support',
                                start=min(w['start'] for w in mapped),end=max(w['end'] for w in mapped),text=unit['text']))
                if error:
                    check=unit_checks.get((bi,ui),{})
                    verified=verified_native_interval(unit,check,vad.get('speech_regions',[]))
                    if verified:mapped=verified;error=None;method='independent_text_native_speech_interval'
                if error and not fresh_verification:
                    omissions.append(dict(block=bi,unit=ui,reason='very_low_confidence_dialogue_location',
                        alignment_error=error,**unit));continue
                if error:
                    method='native_unresolved'
                    issues.append(dict(block=bi,unit=ui,type='unresolved_local_alignment',reason=error,
                        start=unit['start'],end=unit['end'],text=unit['text'],words=unit['words'],
                        confidence=unit['confidence'],no_speech_prob=unit['no_speech_prob']))
            if min(w['start'] for w in mapped)>=boundary:
                if not fresh_verification and zero_phonetic_fragment(unit):
                    omissions.append(dict(block=bi,unit=ui,reason='very_low_confidence_cross_block_location',**unit));continue
                issues.append(dict(block=bi,unit=ui,type='cross_block_location_conflict',reason='cross_block_location',
                    start=unit['start'],end=unit['end'],text=unit['text'],words=unit['words'],
                    confidence=unit['confidence'],no_speech_prob=unit['no_speech_prob']))
            for wi,w in enumerate(mapped):
                if method not in {'independent_text_native_speech_interval','measured_native_sound_event',
                    'independently_classified_native_sound_event','two_model_contextual_native_clock',
                    'contextual_redecode_measured_speech'} and wi not in owned:continue
                a,b=w['start'],w['end']
                if not math.isfinite(a+b) or not 0<=a<=b<=duration+.001:
                    issues.append(dict(block=bi,unit=ui,type='invalid_word_time',word=w));continue
                word_rows.append(dict(w,kind=unit['kind'],utterance=f'{bi}:{unit["segment"]}:{ui}',
                    utterance_end=wi==len(mapped)-1,clock=method))
            decisions.append(dict(block=bi,unit=ui,clock=method,start=unit['start'],end=unit['end']))
        source_units={f'{bi}:{u["segment"]}:{ui}':u for ui,u in enumerate(units)}
        def realign_pair(first,second):
            left=max(block['start'],min(first['start'],second['start'])-.35)
            right=min(block['end'],boundary,max(first['end'],second['end'])+.35)
            if not .02<right-left<=8:return None
            combined=dict(text=first['text']+second['text'],words=first['words']+second['words'])
            clip=wav[round(left*sr):round(right*sr)]
            inputs,words=processor.prepare_forced_aligner_inputs(audio=clip,transcript=combined['text'],language='Japanese')
            inputs=inputs.to(model.device,model.dtype)
            with torch.inference_mode():result=model(**inputs)
            aligned=processor.decode_forced_alignment(logits=result.logits,input_ids=inputs['input_ids'],
                word_lists=words,timestamp_token_id=model.config.timestamp_token_id)[0]
            mapped,error=map_aligned_words(aligned,combined,left,right)
            if error:return None
            count=len(first['words']);a,b=mapped[:count],mapped[count:]
            if any(max(w['end'] for w in part)-min(w['start'] for w in part)<=.02 for part in (a,b)):return None
            return a,b
        word_rows,dropped,repaired=reconcile(word_rows,source_units,realign_pair)
        omissions.extend(dict(block=bi,**r) for r in dropped)
        order_repairs.extend(dict(block=bi,**r) for r in repaired)
        blocks.append(dict(sample=block['sample'],block=bi,original_text=block['text'],words=word_rows,cues=[]))
        if bi%20==0:print('LOCAL_ALIGN',bi,'/',len(asr['blocks']),flush=True)
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    report=dict(policy=POLICY,aligner='Qwen3-ForcedAligner-0.6B-hf',load_seconds=load,align_seconds=seconds,
        content_policy=CONTENT_POLICY,blocks=blocks,issues=issues,warnings=warnings,fallbacks=decisions,
        omissions=omissions,order_repairs=order_repairs,status='needs_review' if issues else 'automatic_checks_only')
    (dest/'alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('RESULT',round(seconds,2),'issues',len(issues),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve(),'--fresh-verification' in sys.argv)
