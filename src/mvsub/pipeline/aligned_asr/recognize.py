"""Text-only ASR on identical overlapping blocks for independent alignment."""
import json
import os
from pathlib import Path
import sys
import sysconfig
import time
import hashlib
import numpy as np
from scipy.io import wavfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config

ROOT=config.repo_root()
WORK=Path(sys.argv[1]).resolve().parent
DOWNLOAD=config.model_home()
AUDIO=Path(sys.argv[2]).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding='utf-8')
handles=[];bins=[]
for base in (Path(sysconfig.get_path('purelib'))/'nvidia',config.cuda_libs()/'nvidia'):
    if base.is_dir():
        for pkg in base.iterdir():
            d=pkg/'bin'
            if d.is_dir():
                bins.append(str(d))
                if hasattr(os,'add_dll_directory'):handles.append(os.add_dll_directory(str(d)))
_current=[p for p in os.environ.get('PATH','').split(os.pathsep) if p]
_seen={p.lower() for p in _current}
_extra=[b for b in bins if b.lower() not in _seen]
# Windows 环境变量上限 32767：继承来的 PATH 过长时 setitem 会直接抛 ValueError
# （曾导致 61 部片在 recognize 阶段秒失败）。这里去重并从尾部裁掉冗余项。
_parts=_extra+_current
while len(os.pathsep.join(_parts))>30000 and len(_parts)>len(_extra):
    _parts.pop()
os.environ['PATH']=os.pathsep.join(_parts)
from faster_whisper import WhisperModel
from extract_srt import find_model_dir
from aligned_asr.utterance import block_content_risks, POLICY, recognizer_revision
label='japanese-turbo'
with_words=True
case=Path(sys.argv[1]).name
directory=WORK/case;directory.mkdir(parents=True,exist_ok=True)
from asr_config import ASR_MODEL
model_dir=ASR_MODEL
start=time.perf_counter()
model=WhisperModel(str(model_dir),device='cuda',compute_type='float16')
load=time.perf_counter()-start
result=dict(model=label,asr_load_seconds=load,settings=dict(beam_size=5,temperature=0,
    word_timestamps=with_words,vad_filter=False,condition_on_previous_text=False,no_speech_threshold=.95,
    chunk_seconds=30,overlap_seconds=0,continuous_seek=True,retain_nonverbal=True),samples=[],blocks=[])
result['policy']=POLICY
result['recognizer_revision']=recognizer_revision()
result['input_sha1']=hashlib.sha1((AUDIO/'full_audio.wav').read_bytes()).hexdigest().upper()
samples=[('continuous',600,0),('cue162',40,1235),('cue426',50,3075),('cue679',40,5390)]
if True:
    rate,full=wavfile.read(AUDIO/'full_audio.wav');duration=len(full)/rate;del full
    samples=[('full_audio',duration,0)]
for name,duration,offset in samples:
    started=time.perf_counter()
    rate,audio=wavfile.read(AUDIO/f'{name}.wav')
    if rate!=16000:raise ValueError('expected 16k mono')
    audio=audio.astype(np.float32)/32768
    blocks=[]
    # Let Whisper carry its own continuous seek clock. External overlapping
    # windows caused truncated boundary words and competing owners.
    stream,info=model.transcribe(audio,language='ja',beam_size=5,temperature=0,
        word_timestamps=with_words,vad_filter=False,condition_on_previous_text=False,no_speech_threshold=.95)
    windows=[];window=[];window_start=0.
    for segment in stream:
        if window and segment.end-window_start>30:
            windows.append((window_start,window));window=[]
        if not window:window_start=max(0.,segment.start)
        window.append(segment)
    if window:windows.append((window_start,window))
    for window_start,segments in windows:
        base=round(window_start*rate)
        window_end=min(duration,max(s.end for s in segments)+.35)
        clip=audio[base:round(window_end*rate)]
        text=''.join(s.text for s in segments).strip()
        row=dict(sample=name,start=base/rate,end=(base+len(clip))/rate,absolute_offset=offset,
                 text=text,owned_end=window_end,
                 segments=[dict(start=s.start-base/rate,end=s.end-base/rate,text=s.text,avg_logprob=s.avg_logprob,
                                no_speech_prob=s.no_speech_prob,words=[dict(start=w.start-base/rate,end=w.end-base/rate,text=w.word,
                                probability=w.probability) for w in (s.words or [])]) for s in segments])
        row['content_risks']=block_content_risks(row)
        if row['content_risks']:
            # An independent decode with short windows is evidence, not a
            # forced transcript. Keep conflicting content explicitly blocked.
            checks=[]
            for local in range(0,len(clip),10*rate):
                retry,info=model.transcribe(clip[local:local+10*rate],language='ja',beam_size=5,
                    temperature=0,word_timestamps=True,vad_filter=False,
                    condition_on_previous_text=False,no_speech_threshold=.6)
                retry=list(retry)
                checks.append(dict(start=base/rate+local/rate,text=''.join(s.text for s in retry),
                    segments=[dict(start=local/rate+s.start,end=local/rate+s.end,text=s.text,
                        avg_logprob=s.avg_logprob,no_speech_prob=s.no_speech_prob) for s in retry]))
            row['local_redecode']=checks
            # Never accept a risky loop solely because the same decoder repeats
            # it again. Full evidence remains for a dedicated content review.
            row['content_risks'].append('needs_content_review_after_local_redecode')
        blocks.append(row)
        if len(blocks)%20==0:print('ASR_BLOCK',label,name,len(blocks),flush=True)
    elapsed=time.perf_counter()-started
    result['blocks'].extend(blocks)
    result['samples'].append(dict(sample=name,duration=duration,offset=offset,seconds=elapsed,blocks=len(blocks)))
    (directory/'asr.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(label,name,f'{elapsed:.2f}s',len(blocks),flush=True)
result['asr_seconds']=sum(s['seconds'] for s in result['samples'])
(directory/'asr.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(directory/'raw_asr.json').write_bytes((directory/'asr.json').read_bytes())
print('RESULT',label,result['asr_seconds'],flush=True)
