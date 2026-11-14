"""Re-decode failed utterances with wider context and two native clocks."""
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig
import time
import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
from asr_config import ASR_MODEL,MODEL_HOME,ALIGN_LIBS
from aligned_asr.recovery import consistent_recovery,native_clock_supported
from aligned_asr.utterance import model_identity


def run(dest,audio):
    started=time.perf_counter();data=json.loads((dest/'asr.json').read_text('utf-8'))
    alignment=json.loads((dest/'alignment.json').read_text('utf-8'))
    targets=[i for i in alignment['issues'] if i['type']=='unresolved_local_alignment' and i['end']-i['start']<=8]
    attempts=[];recoveries=[];handles=[];bins=[]
    audio_sha=hashlib.sha1(audio.read_bytes()).hexdigest().upper()
    vad_path=dest/'vad_evidence.json'
    vad=json.loads(vad_path.read_text('utf-8')) if vad_path.exists() else {}
    if vad.get('audio_sha1')!=audio_sha:vad={}
    for base in (Path(sysconfig.get_path('purelib'))/'nvidia',config.repo_root()/'lib/cuda/nvidia'):
        if base.is_dir():
            for package in base.iterdir():
                folder=package/'bin'
                if folder.is_dir():
                    bins.append(str(folder))
                    if hasattr(os,'add_dll_directory'):handles.append(os.add_dll_directory(str(folder)))
    os.environ['PATH']=os.pathsep.join(bins+[os.environ.get('PATH','')])
    if targets:
        from faster_whisper import WhisperModel
        sr,wav=wavfile.read(audio)
        if sr!=16000 or wav.ndim!=1:raise ValueError('Expected mono 16kHz PCM')
        wav=wav.astype(np.float32)/32768
        decodes={};models=[ASR_MODEL,MODEL_HOME/'models/whisper-large-v3']
        for mi,path in enumerate(models):
            if not (path/'model.bin').exists():raise RuntimeError('Local clock recovery model missing; no download')
            model=WhisperModel(str(path),device='cuda',compute_type='float16')
            for ti,unit in enumerate(targets):
                left=max(0,unit['start']-2);right=min(len(wav)/sr,unit['end']+2)
                stream,_=model.transcribe(wav[round(left*sr):round(right*sr)],language='ja',beam_size=5,
                    temperature=0,word_timestamps=True,vad_filter=False,condition_on_previous_text=False,
                    no_speech_threshold=.6,compression_ratio_threshold=2.4)
                segments=[dict(text=s.text,avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob,
                    words=[dict(text=w.word,start=left+w.start,end=left+w.end) for w in s.words or []]) for s in stream]
                decodes[mi,ti]=segments
                attempts.append(dict(block=unit['block'],unit=unit['unit'],model=str(path),
                    model_identity=model_identity(path),clip_start=left,clip_end=right,segments=segments))
            del model
        for ti,unit in enumerate(targets):
            words=consistent_recovery(unit,decodes[0,ti],decodes[1,ti])
            method='two_model_contextual_native_clock'
            if not words and unit.get('words') and native_clock_supported(unit,len(wav)/sr):
                words=consistent_recovery(unit,[dict(words=unit['words'])],decodes[0,ti])
                if words:
                    a,b=min(w['start'] for w in words),max(w['end'] for w in words)
                    support=[(max(a,r['start']),min(b,r['end'])) for r in vad.get('speech_regions',[]) if r['start']<b and r['end']>a]
                    if support:
                        lo=max(a,min(x for x,y in support)-.1);hi=min(b,max(y for x,y in support)+.1)
                        words=[dict(text=unit['text'],start=lo,end=hi)] if hi-lo>.02 else None
                    else:words=None
                    method='contextual_redecode_measured_speech'
            if words:recoveries.append(dict(block=unit['block'],unit=unit['unit'],source_text=unit['text'],
                source_start=unit['start'],source_end=unit['end'],words=words,
                method=method))
    (dest/'word_recovery.json').write_text(json.dumps(dict(
        asr_sha1=hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper(),
        audio_sha1=audio_sha,
        recoveries=recoveries,attempts=attempts,seconds=time.perf_counter()-started),ensure_ascii=False,indent=2),encoding='utf-8')
    print('WORD_RECOVERY',len(recoveries),'/',len(targets),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
