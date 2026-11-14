"""Lexical source review outside the ASR clock revision; never claim human hearing."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import os
import subprocess
import sys
import sysconfig
import wave

from audio_archive import atomic_json, digest
from aligned_asr.timing import normalize
from aligned_asr.utterance import model_identity, NONVERBAL
from asr_config import ASR_MODEL, MODEL_HOME, ALIGN_PYTHON
import config


def decode_review(source, audio, folder, cue_ids, device='cuda'):
    """Finite, cached acoustic probes; call on the existing single GPU stage."""
    from translate_srt_batch import parse_srt
    wanted=set(cue_ids)
    if any(not isinstance(idx,str) or not idx.isdecimal() for idx in wanted):
        raise ValueError('Acoustic review requires decimal SRT IDs')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    try:
        windows,_=load_evidence(folder,source,audio)
        if {w['cue_id'] for w in windows['windows']}==wanted:return folder
    except (OSError,ValueError,KeyError):
        pass
    if any(folder.iterdir()):
        # Never overwrite a different source/model/window experiment.
        identity=hashlib.sha256((sha256(source)+digest(audio)+','.join(sorted(wanted))+
            model_identity(ASR_MODEL)+model_identity(MODEL_HOME/'models/whisper-large-v3')+
            model_identity(MODEL_HOME/'models/qwen-asr-verifier')+sha256(__file__)).encode()).hexdigest()
        folder=folder/identity;folder.mkdir(exist_ok=True)
        try:
            old,_=load_evidence(folder,source,audio)
            if {w['cue_id'] for w in old['windows']}==wanted:return folder
        except (OSError,ValueError,KeyError):
            if any(folder.iterdir()):raise RuntimeError('Incomplete/corrupt review evidence retained; use a clean review directory')
    active=subprocess.run(['nvidia-smi','--query-compute-apps=process_name','--format=csv,noheader'],
                          capture_output=True,text=True,check=True)
    if 'jasna' in active.stdout.lower():raise RuntimeError('Jasna is using GPU; defer source review')
    handles=[];bins=[]
    for base in (Path(sysconfig.get_path('purelib'))/'nvidia',config.cuda_libs()/'nvidia'):
        if base.is_dir():
            for package in base.iterdir():
                dll=package/'bin'
                if dll.is_dir():
                    bins.append(str(dll))
                    if hasattr(os,'add_dll_directory'):handles.append(os.add_dll_directory(str(dll)))
    os.environ['PATH']=os.pathsep.join(bins+[os.environ.get('PATH','')])
    os.environ['HF_HUB_OFFLINE']='1'
    import numpy as np
    from faster_whisper import WhisperModel
    source_entries=parse_srt(Path(source));entries={e[0]:e for e in source_entries}
    if len(entries)!=len(source_entries):raise ValueError('Duplicate source IDs')
    if not wanted<=entries.keys():raise ValueError('Unknown source review ID')
    windows=[];audio_sha=digest(audio)
    with wave.open(str(audio),'rb') as stream:
        if (stream.getframerate(),stream.getnchannels(),stream.getsampwidth())!=(16000,1,2):
            raise ValueError('Review requires mono PCM16 16kHz')
        duration=stream.getnframes()/16000
        for idx in sorted(wanted,key=int):
            entry=entries[idx];a,b=seconds(entry[1]),seconds(entry[2])
            for label,pad in [('near',1.5),('context',4.)]:
                left,right=max(0,a-pad),min(duration,b+pad)
                if not 0<right-left<=38:raise ValueError('Review clip too long; inspect source first')
                stream.setpos(round(left*16000));raw=stream.readframes(round((right-left)*16000))
                clip=folder/f'cue{idx}.{label}.wav'
                with wave.open(str(clip),'wb') as out:
                    out.setnchannels(1);out.setsampwidth(2);out.setframerate(16000);out.writeframes(raw)
                windows.append(dict(cue_id=idx,label=label,start=a,end=b,clip_start=left,clip_end=right,
                    source_text=entry[3],file=clip.name,clip_sha1=digest(clip)))
    atomic_json(folder/'windows.json',dict(source_sha256=sha256(source),audio_sha1=audio_sha,windows=windows))
    for label,path in [('760M',ASR_MODEL),('large-v3',MODEL_HOME/'models/whisper-large-v3')]:
        if not (path/'model.bin').exists():raise FileNotFoundError('Required offline review model: '+str(path))
        model=WhisperModel(str(path),device=device,compute_type='float16' if device=='cuda' else 'int8',
                           local_files_only=True)
        checks=[]
        for window in windows:
            with wave.open(str(folder/window['file']),'rb') as stream:
                samples=np.frombuffer(stream.readframes(stream.getnframes()),dtype=np.int16).astype(np.float32)/32768
            segments,_=model.transcribe(samples,language='ja',beam_size=5,temperature=0,
                word_timestamps=True,vad_filter=False,condition_on_previous_text=False,no_speech_threshold=.95)
            segments=list(segments)
            words=[dict(text=w.word,start=window['clip_start']+w.start,end=window['clip_start']+w.end,
                        probability=w.probability) for s in segments for w in (s.words or [])]
            checks.append(dict(window,text=''.join(s.text for s in segments),words=words))
        del model
        atomic_json(folder/(label+'.json'),dict(model=label,model_identity=model_identity(path),
            source_sha256=sha256(source),audio_sha1=audio_sha,checks=checks,human_audio_verified=False))
    atomic_json(folder/'asr.json',dict(blocks=[dict(sample='full_audio',start=w['clip_start'],end=w['clip_end'],
        text=w['source_text'],segments=[]) for w in windows]))
    if not ALIGN_PYTHON.exists():raise FileNotFoundError('Independent verifier runtime missing')
    # Qwen verification is scheduled on the single GPU stage, even for CPU Whisper.
    with (folder/'qwen.log').open('w',encoding='utf-8') as log:
        result=subprocess.run([str(ALIGN_PYTHON),str(Path(__file__).parent/'aligned_asr/verify_content.py'),
            str(folder),str(Path(audio).resolve()),'--all'],stdout=log,stderr=subprocess.STDOUT,timeout=2400)
    if result.returncode:raise RuntimeError('Independent acoustic review failed; see qwen.log')
    load_evidence(folder,source,audio)
    return folder


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest()


def seconds(value):
    h,m,s=value.replace(',','.').split(':')
    return int(h)*3600+int(m)*60+float(s)


def triage(entries, asr):
    """Scores choose acoustic checks; a score alone does not authorize edits."""
    words=[]
    for block in asr.get('blocks',[]):
        for segment in block.get('segments',[]):
            for word in segment.get('words',[]):
                value=normalize(word['text'])
                if value:
                    words.append(dict(text=value,start=block['start']+word['start'],
                        end=block['start']+word['end'],probability=word.get('probability',1)))
    risks=[]
    for idx,a,b,text in entries:
        if not normalize(text) or NONVERBAL.fullmatch(normalize(text)):continue
        left,right=seconds(a),seconds(b)
        own=[w for w in words if left-.25<=(w['start']+w['end'])/2<=right+.25]
        low=[w for w in own if re.search('[一-鿿]',w['text']) and w['probability']<.65]
        if low:
            risks.append(dict(id=idx,start=left,end=right,source_text=text,
                score=round(1-min(w['probability'] for w in low),4),
                hard_risk=any(w['probability']<.35 for w in low),
                reason='low_confidence_lexical_word',words=low))
    return sorted(risks,key=lambda r:(not r['hard_risk'],-r['score'],int(r['id'])))


def load_evidence(folder, source, audio):
    """Reject stale or edited witnesses; paths are confined to the review folder."""
    folder=Path(folder).resolve()
    windows=json.loads((folder/'windows.json').read_text(encoding='utf-8'))
    if windows['source_sha256']!=sha256(source) or windows['audio_sha1']!=digest(audio):
        raise ValueError('Review source/audio changed')
    from translate_srt_batch import parse_srt
    entries={e[0]:e for e in parse_srt(Path(source))};expected={}
    with wave.open(str(audio),'rb') as raw_audio:
        duration=raw_audio.getnframes()/raw_audio.getframerate()
        for window in windows['windows']:
            entry=entries[window['cue_id']];pad={'near':1.5,'context':4.}[window['label']]
            a,b=seconds(entry[1]),seconds(entry[2])
            if (window['source_text']!=entry[3] or window['start']!=a or window['end']!=b
                    or window['clip_start']!=max(0,a-pad) or window['clip_end']!=min(duration,b+pad)):
                raise ValueError('Review window does not match the original cue')
            clip=(folder/window['file']).resolve()
            if not clip.is_relative_to(folder) or digest(clip)!=window['clip_sha1']:
                raise ValueError('Review clip changed or escaped its folder')
            raw_audio.setpos(round(window['clip_start']*16000))
            expected_pcm=raw_audio.readframes(round((window['clip_end']-window['clip_start'])*16000))
            with wave.open(str(clip),'rb') as cut:
                if (cut.getframerate(),cut.getnchannels(),cut.getsampwidth())!=(16000,1,2) or cut.readframes(cut.getnframes())!=expected_pcm:
                    raise ValueError('Review clip is not the declared source audio interval')
            key=window['cue_id'],window['label']
            if key in expected:raise ValueError('Duplicate review window')
            expected[key]=window
    witnesses={}
    for label,model_path in [('760M',ASR_MODEL),('large-v3',MODEL_HOME/'models/whisper-large-v3')]:
        data=json.loads((folder/(label+'.json')).read_text(encoding='utf-8'))
        if (data['audio_sha1']!=windows['audio_sha1'] or data['source_sha256']!=windows['source_sha256']
                or data['model_identity']!=model_identity(model_path)):
            raise ValueError('Review model/input identity changed')
        for check in data['checks']:
            key=check['cue_id'],check['label']
            if key not in expected or any(check[k]!=expected[key][k] for k in
                    ('clip_start','clip_end','start','end','source_text','clip_sha1')):
                raise ValueError('Witness window identity changed')
            witnesses[label,*key]=dict(text=check['text'],words=check['words'])
    qwen=json.loads((folder/'content_verification.json').read_text(encoding='utf-8'))
    verifier=Path(__file__).parent/'aligned_asr/verify_content.py'
    revision=hashlib.sha1(verifier.read_bytes()+model_identity(MODEL_HOME/'models/qwen-asr-verifier').encode()).hexdigest().upper()
    if (qwen['audio_sha1']!=windows['audio_sha1'] or qwen['asr_sha1']!=digest(folder/'asr.json')
            or qwen['cache_identity']!=revision):
        raise ValueError('Independent verifier identity changed')
    for check in qwen['checks']:
        window=windows['windows'][check['block']]
        # A verifier that runs into its token limit on a long sound-only window still
        # proves the window identity; truncation alone must not block the whole film.
        if (check['independent'].get('requested_language')!='Japanese'
                or check['start']!=window['clip_start'] or check['end']!=window['clip_end']
                or check['native_text']!=window['source_text']):
            raise ValueError('Independent witness incomplete or bound to another window')
        witnesses['qwen',window['cue_id'],window['label']]=dict(text=check['independent']['transcription'])
    for cue_id,label in expected:
        if any((model,cue_id,label) not in witnesses for model in ('760M','large-v3','qwen')):
            raise ValueError('Missing independent review witness')
    return windows,witnesses


def supported_edit(entry, old, new, witnesses):
    """Two architectures and both windows must uniquely support the new span."""
    if not old or not new or entry[3].count(old)!=1 or not normalize(new):
        raise ValueError('Edit must replace one nonempty exact source span')
    value=normalize(new); previous=normalize(old)
    for model in ('large-v3','qwen'):
        for window in ('near','context'):
            text=normalize(witnesses[model,entry[0],window]['text'])
            if text.count(value)!=1 or previous in text:
                raise ValueError('Independent architectures/windows do not support this edit')
    # The changed lexical span must occur inside this cue, not a neighbor.
    bounds=[]
    for window in ('near','context'):
        words=witnesses['large-v3',entry[0],window]['words']
        tape='';spans=[]
        for word in words:
            text=normalize(word['text']);spans.append((len(tape),len(tape)+len(text),word));tape+=text
        position=tape.find(value)
        if position<0 or tape.find(value,position+1)>=0:raise ValueError('No unique timed lexical support')
        own=[w for a,b,w in spans if a<position+len(value) and b>position]
        left=min(w['start'] for w in own);right=max(w['end'] for w in own)
        if (min(w.get('probability',0) for w in own)<.35
                or sum(w.get('probability',0) for w in own)/len(own)<.7):
            raise ValueError('Corrected lexical support is low confidence')
        if not seconds(entry[1])<=left<right<=seconds(entry[2]):
            raise ValueError('Corrected span lies outside existing cue; fresh alignment required')
        bounds.append([left,right])
    if any(abs(a-b)>.4 for a,b in zip(bounds[0],bounds[1])):
        raise ValueError('Independent window clocks disagree; fresh alignment required')
    return dict(id=entry[0],old=old,new=new,bounds=bounds,
                text=entry[3].replace(old,new),human_audio_verified=False)


def make_report(source, audio, output, asr_path=None, evidence_dir=None, explicit_ids=()):
    from translate_srt_batch import parse_srt
    entries=parse_srt(Path(source)); by_id={e[0]:e for e in entries}
    if len(by_id)!=len(entries) or not entries:raise ValueError('Invalid source IDs')
    if not set(explicit_ids)<=by_id.keys():raise ValueError('Unknown explicit source review ID')
    asr=json.loads(Path(asr_path).read_text(encoding='utf-8')) if asr_path else None
    if asr is not None and asr.get('input_sha1')!=digest(audio):raise ValueError('ASR and source audio differ')
    risks=triage(entries,asr) if asr is not None else []
    # Confidence selects checks. Unreviewed scores alone are not proof of error.
    holds=set(explicit_ids)
    checked=[];evidence={}
    if evidence_dir:
        windows,witnesses=load_evidence(evidence_dir,source,audio)
        ids={w['cue_id'] for w in windows['windows']}
        for idx in sorted(ids,key=int):
            native=normalize(by_id[idx][3])
            supported=all(native in normalize(witnesses[m,idx,w]['text'])
                          for m in ('large-v3','qwen') for w in ('near','context'))
            checked.append(dict(id=idx,status='native_supported' if supported else 'needs_audio_review'))
            if supported:holds.discard(idx)
            else:holds.add(idx)
        evidence={str(Path(evidence_dir).resolve()/name):sha256(Path(evidence_dir)/name)
                  for name in ('windows.json','760M.json','large-v3.json','content_verification.json','asr.json')}
    models={name:model_identity(path) for name,path in [('760M',ASR_MODEL),('large-v3',MODEL_HOME/'models/whisper-large-v3'),
             ('qwen',MODEL_HOME/'models/qwen-asr-verifier')]} if evidence else {}
    report=dict(policy='lexical-review-v1',revision=sha256(__file__),models=models,source_path=str(Path(source).resolve()),source_sha256=sha256(source),
                audio_path=str(Path(audio).resolve()),audio_sha1=digest(audio),risks=risks,
                checked=checked,hold_ids=sorted(holds,key=int),evidence=evidence,
                human_audio_verified=False,unreviewed_soft_risk_ids=[r['id'] for r in risks if r['id'] not in {c['id'] for c in checked}])
    atomic_json(Path(output),report)
    return report


def ensure_review(source,audio,work,asr_path,max_checks=8):
    path=Path(work)/'source_review.json'
    try:load_report(path,source);return path
    except (OSError,ValueError,KeyError):pass
    from translate_srt_batch import parse_srt
    asr=json.loads(Path(asr_path).read_text(encoding='utf-8'))
    risks=triage(parse_srt(Path(source)),asr)
    selected=[r['id'] for r in risks[:max_checks]]
    evidence=decode_review(source,audio,Path(work)/'source_review',selected) if selected else None
    make_report(source,audio,path,asr_path,evidence)
    return path


def load_report(path, source, _depth=0):
    if _depth>3:raise ValueError('Source review parent cycle')
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('policy')!='lexical-review-v1' or data['source_sha256']!=sha256(source):
        raise ValueError('Source review does not match current source')
    if data.get('revision')!=sha256(__file__):raise ValueError('Lexical review policy changed')
    for name,identity in data.get('models',{}).items():
        model={'760M':ASR_MODEL,'large-v3':MODEL_HOME/'models/whisper-large-v3','qwen':MODEL_HOME/'models/qwen-asr-verifier'}[name]
        if model_identity(model)!=identity:raise ValueError('Acoustic review model changed')
    if digest(data['audio_path'])!=data['audio_sha1']:
        raise ValueError('Reviewed audio changed')
    for file,checksum in data.get('evidence',{}).items():
        if sha256(file)!=checksum:raise ValueError('Acoustic review evidence changed')
    if data.get('parent_report'):
        from translate_srt_batch import parse_srt,format_reference_context
        parent=Path(data['parent_report'])
        if parent.resolve()==Path(path).resolve() or sha256(parent)!=data['parent_report_sha256']:
            raise ValueError('Parent source review changed')
        full=load_report(parent,data['parent_source'],_depth+1)
        format_reference_context(parse_srt(Path(source)),parse_srt(Path(full['source_path'])))
    return data


def subset_report(parent_path, source, output):
    """A test/repair slice inherits real evidence, never a fabricated QA pass."""
    from translate_srt_batch import parse_srt,format_reference_context
    parent=json.loads(Path(parent_path).read_text(encoding='utf-8'))
    parent=load_report(parent_path,parent['source_path'])
    entries=parse_srt(Path(source));format_reference_context(entries,parse_srt(Path(parent['source_path'])))
    ids={e[0] for e in entries}
    data=dict(parent,source_path=str(Path(source).resolve()),source_sha256=sha256(source),
        hold_ids=[i for i in parent['hold_ids'] if i in ids],reference_hold_ids=parent['hold_ids'],
        parent_report=str(Path(parent_path).resolve()),parent_report_sha256=sha256(parent_path),
        parent_source=parent['source_path'])
    atomic_json(Path(output),data)
    return data


def reference_source(report):
    """Resolve an authenticated slice back to its original film source."""
    seen=set()
    while report.get('parent_report'):
        parent=report['parent_report']
        if parent in seen:raise ValueError('Source review parent cycle')
        seen.add(parent)
        report=load_report(parent,report['parent_source'])
    return Path(report['source_path'])


def propose_patch(source, audio, evidence_dir, edits, output):
    from translate_srt_batch import parse_srt,write_srt
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source:raise ValueError('Repairs must be written to a separate candidate')
    entries=parse_srt(source);by_id={e[0]:e for e in entries}
    _,witnesses=load_evidence(evidence_dir,source,audio)
    accepted={};proposed=[]
    for edit in edits:
        idx=str(edit['id'])
        if idx in accepted:raise ValueError('Multiple edits to one cue require one combined proposal')
        if not edit['old'] or not edit['new'] or by_id[idx][3].count(edit['old'])!=1:
            raise ValueError('Invalid or ambiguous proposed span')
        try:
            item=supported_edit(by_id[idx],edit['old'],edit['new'],witnesses)
            item['status']='supported_candidate'
        except ValueError as error:
            item=dict(id=idx,old=edit['old'],new=edit['new'],text=by_id[idx][3].replace(edit['old'],edit['new']),
                      status='needs_audio_review',reason=str(error),human_audio_verified=False)
        accepted[idx]=item
        proposed.append(item)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_srt(output,[(e[0],e[1],e[2],accepted[e[0]]['text'] if e[0] in accepted else e[3]) for e in entries])
    manifest=dict(policy='supported-lexical-candidate-v1',base_source=str(source),base_sha256=sha256(source),
                  candidate_sha256=sha256(output),audio=str(Path(audio).resolve()),audio_sha1=digest(audio),
                  evidence_dir=str(Path(evidence_dir).resolve()),edits=proposed,
                  cue_geometry_unchanged=True,publishable=False,human_audio_verified=False,
                  next_step='Run source/timing validation before publication; acoustic agreement is not human hearing')
    atomic_json(output.with_suffix('.repair.json'),manifest)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path);p.add_argument('audio',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--asr-json',type=Path);p.add_argument('--evidence-dir',type=Path)
    p.add_argument('--only-id',action='append',default=[])
    p.add_argument('--decode',action='store_true',help='Bounded local acoustic checks; use in the single GPU stage')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--max-checks',type=int,default=8)
    p.add_argument('--edits',type=Path,help='Explicit local proposals; produce candidate, never overwrite source')
    args=p.parse_args()
    if not 1<=args.max_checks<=32:p.error('--max-checks must be between 1 and 32')
    if args.decode:
        from translate_srt_batch import parse_srt
        risks=triage(parse_srt(args.source),json.loads(args.asr_json.read_text(encoding='utf-8'))) if args.asr_json else []
        targets=list(dict.fromkeys(args.only_id+[r['id'] for r in risks]))[:args.max_checks]
        if targets:
            args.evidence_dir=decode_review(args.source,args.audio,args.evidence_dir or args.output.parent/'source_review',targets,args.device)
    if args.edits:
        result=propose_patch(args.source,args.audio,args.evidence_dir,
            json.loads(args.edits.read_text(encoding='utf-8')),args.output)
    else:
        result=make_report(args.source,args.audio,args.output,args.asr_json,args.evidence_dir,args.only_id)
    print(json.dumps(dict(source_review=True,hold_ids=result.get('hold_ids',[]),
                          candidate=result.get('candidate_sha256'),output=str(args.output)),ensure_ascii=False))
