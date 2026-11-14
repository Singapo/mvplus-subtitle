"""Tested Japanese 760M + Qwen independent alignment, atomic SRT publication."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from asr_config import ASR_MODEL, ALIGN_MODEL, ALIGN_PYTHON, ALIGN_LIBS
from nosub_pipeline import parse_srt_cues, probe_duration
from aligned_asr.utterance import POLICY, CONTENT_POLICY, pipeline_revision, recognizer_revision, review_digest

TOOLS=Path(__file__).resolve().parent
PY = sys.executable


def extract(source, output, work=None):
    source,output=Path(source).resolve(),Path(output).resolve()
    for path in (source,ASR_MODEL/'model.bin',ALIGN_MODEL/'config.json',ALIGN_PYTHON,ALIGN_LIBS):
        if not path.exists():raise FileNotFoundError(path)
    job=Path(work).resolve() if work else output.parent/(output.stem+'.alignment')
    job.mkdir(parents=True,exist_ok=True)
    audio=job/'full_audio.wav'
    if source!=audio:
        part=audio.with_suffix('.part.wav')
        result=subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),
            '-vn','-ar','16000','-ac','1','-af','aresample=16000:async=16000:first_pts=0',
            '-c:a','pcm_s16le',str(part)],capture_output=True)
        if result.returncode:raise RuntimeError('Local audio preparation failed; output retained')
        part.replace(audio)
    started=time.perf_counter();revision=pipeline_revision()
    input_audio_sha1=hashlib.sha1(audio.read_bytes()).hexdigest().upper();input_review_digest=review_digest(job)
    stages=[('recognize',PY),('vad_evidence',PY),('align',ALIGN_PYTHON),
            ('verify_content',ALIGN_PYTHON),('recover_words',PY),
            ('verify_events',ALIGN_PYTHON),('align',ALIGN_PYTHON),('caption',PY),('repair',PY)]
    for stage_index,(stage,runtime) in enumerate(stages):
        if stage=='recognize' and (job/'raw_asr.json').exists():
            cached=json.loads((job/'raw_asr.json').read_text('utf-8'))
            if (cached.get('policy')==POLICY and cached.get('input_sha1')==hashlib.sha1(audio.read_bytes()).hexdigest().upper()
                and cached.get('recognizer_revision')==recognizer_revision()):
                (job/'asr.json').write_bytes((job/'raw_asr.json').read_bytes())
                print('[aligned-asr] verified ASR cache reused',flush=True);continue
        command=[str(runtime),str(TOOLS/'aligned_asr'/f'{stage}.py'),str(job),str(audio)]
        if stage=='align' and stage_index==2:command.append('--fresh-verification')
        with (job/f'{stage}.log').open('w',encoding='utf-8') as stage_log:
            result=subprocess.run(command,stdout=stage_log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'{stage} failed; see {job/stage}.log; existing subtitle retained')
        print(f'[aligned-asr] {stage} completed',flush=True)
    candidate=job/'full_audio.candidate.ja.srt'
    cues=parse_srt_cues(candidate)
    duration=probe_duration(audio)
    if duration is None:raise ValueError('No audio duration')
    previous=-1
    flags=[];quality_issues=[]
    for i,(idx,a,b,text) in enumerate(cues):
        if not text or not 0<=a<b<=duration+.001 or a<previous:
            quality_issues.append(dict(type='invalid_cue',id=idx,start=a,end=b))
        previous=a
        reasons=[]
        if i and a<cues[i-1][2]:reasons.append('overlap')
        if b-a>8:reasons.append('long')
        if b-a<.2:reasons.append('short')
        if reasons:flags.append(dict(id=idx,start=a,end=b,flags=reasons))
    repairs=json.loads((job/'zero_event_repair.json').read_text(encoding='utf-8'))
    alignment=json.loads((job/'alignment.json').read_text(encoding='utf-8'))
    caption=json.loads((job/'event_captioning.json').read_text(encoding='utf-8'))
    if not cues and not alignment.get('omissions'):quality_issues.append(dict(type='unexpected_empty_subtitle'))
    quality_issues.extend(alignment['issues']);quality_issues.extend(caption['issues'])
    if repairs['unresolved']:quality_issues.append(dict(type='unresolved_events',count=len(repairs['unresolved'])))
    if pipeline_revision()!=revision:quality_issues.append(dict(type='pipeline_changed_during_run'))
    if review_digest(job)!=input_review_digest:quality_issues.append(dict(type='manual_review_changed_during_run'))
    if hashlib.sha1(audio.read_bytes()).hexdigest().upper()!=input_audio_sha1:
        quality_issues.append(dict(type='audio_changed_during_run'))
    report=dict(model='efwkjn/faster-whisper-ja-760M',aligner='Qwen3-ForcedAligner-0.6B-hf',
        timing_policy=POLICY,pipeline_revision=revision,quality_passed=not quality_issues,quality_issues=quality_issues,
        content_policy=CONTENT_POLICY,warnings=alignment.get('warnings',[]),omissions=alignment.get('omissions',[]),
        audio_sha1=input_audio_sha1,manual_review_digest=input_review_digest,
        source_sha1=hashlib.sha1(candidate.read_bytes()).hexdigest().upper(),
        seconds=time.perf_counter()-started,cues=len(cues),audio_seconds=duration,
        unresolved_events=repairs['unresolved'],timing_review=flags,human_audio_verified=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if quality_issues:
        raise ValueError(f'Quality blocked: {len(quality_issues)} issues; draft and evidence retained, previous subtitle not replaced')
    output.parent.mkdir(parents=True,exist_ok=True)
    # Never expose a partial SRT; keep the previous output on any earlier failure.
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=output.name+'.',suffix='.part',delete=False) as handle:
        temp=Path(handle.name);handle.write(candidate.read_bytes())
    try:temp.replace(output)
    finally:temp.unlink(missing_ok=True)
    print(f'[aligned-asr] published {len(cues)} cues; unresolved {len(repairs["unresolved"])}',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path);p.add_argument('--output',type=Path);p.add_argument('--work',type=Path)
    args=p.parse_args()
    extract(args.source,args.output or args.source.with_suffix('.srt'),args.work)
