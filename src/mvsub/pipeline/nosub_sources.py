"""Bounded local source preparation for a frozen, confirmed-missing manifest.

Japanese keeps the existing independent alignment gate. Other languages use
the installed multilingual recognizer's word timestamps, explicitly recorded
as a different source method, never as Qwen-verified Japanese alignment.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
import gc
import json
import os
import subprocess
import sys
import time
import wave

import native_subtitles as native
import config
from asr_config import MODEL_HOME
from cloud_common import Cloud115
from nosub_pipeline import load_env, parse_srt_cues, archive_original, ORIG_ARCHIVE_DIR, free_gb
from audio_archive import archive_audio, restore_audio, verified_link, digest, pcm_info
from reprocess_subtitles import runner_lock, require_current_timing, prune_duplicate_audio_chunks
import subtitle_version_reuse as versions

ROOT=Path(__file__).resolve().parents[1]
TOOLS=Path(__file__).resolve().parent
MODEL=config.multilingual_model()
WORD_POLICY='local-multilingual-word-clock-v1'


def language_review_available(target,job):
    blocked=Path(target)/'blocked.json';review=Path(job)/'language_speech_review.json'
    if not blocked.exists() or not review.exists():return False
    state=native.read(blocked)
    return (state.get('reason')=='Language uncertain; retain audio without forcing Japanese'
        and not state.get('speech_review_attempted') and native.read(review).get('status')=='language_resolved')


def choose_language(observations):
    confident=[r['language'] for r in observations if r['probability']>=.8]
    winners={v for v in confident if confident.count(v)>=2}
    if len(winners)!=1:raise ValueError('Language uncertain; retain audio without forcing Japanese')
    return winners.pop()


def validate_word_clock(source):
    source=Path(source);report=native.read(source.with_suffix('.word_clock.json'))
    if report.get('policy')!=WORD_POLICY or report.get('quality_passed') is not True:
        raise ValueError('Word-clock checks did not pass')
    native.check_proofs([report['source'],report['audio'],report['model_config']])
    if report['source']!=native.proof(source):raise ValueError('Word-clock source differs')
    if abs(pcm_info(report['audio']['path'])['duration']-report['audio_seconds'])>.0001:
        raise ValueError('Word-clock audio duration differs')
    cues=parse_srt_cues(source);previous=-1
    if not cues:raise ValueError('No usable speech; do not translate an empty source')
    for _,start,end,text in cues:
        if not text or not 0<=start<end<=report['audio_seconds']+.001 or start<previous:
            raise ValueError('Invalid word-clock cue')
        previous=start
    return report


def extract_auto(audio,source,job,repair_word_alignment=False):
    import numpy as np
    from extract_srt import load_faster_whisper,extract
    # Use the same CUDA DLL setup as the established local recognizer.
    bins=[];handles=[]
    for base in (Path(os.sys.prefix)/'Lib/site-packages/nvidia',ROOT/'subtitle/lib/cuda/nvidia'):
        if base.is_dir():
            for package in base.iterdir():
                target=package/'bin'
                if target.is_dir():
                    bins.append(str(target))
                    if hasattr(os,'add_dll_directory'):handles.append(os.add_dll_directory(str(target)))
    os.environ['PATH']=os.pathsep.join(bins+[os.environ.get('PATH','')])
    if not (MODEL/'model.bin').exists():raise FileNotFoundError('Installed multilingual model missing; no automatic download')
    review=job/'language_speech_review.json'
    reviewed=native.read(review) if review.exists() else None
    if reviewed and reviewed.get('status')=='language_resolved':
        native.check_proofs([reviewed['audio']])
        if reviewed['audio']!=native.proof(audio):raise ValueError('Language review audio differs')
        observations=reviewed['observations'];language=choose_language(observations)
        duration=pcm_info(audio)['duration']
    else:
        observations=None
    with wave.open(str(audio),'rb') as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):
            raise ValueError('Expected verified mono 16 kHz PCM')
        duration=wav.getnframes()/16000
        if observations is None:
            detector=load_faster_whisper()(str(MODEL),device='cuda',compute_type='float16')
            observations=[]
            for fraction in (.1,.4,.7):
                start=max(0,min(duration-30,duration*fraction))
                wav.setpos(int(start*16000))
                sample=np.frombuffer(wav.readframes(30*16000),dtype=np.int16).astype(np.float32)/32768
                language,probability,_=detector.detect_language(audio=sample)
                observations.append(dict(offset_seconds=start,language=language,probability=float(probability)))
            del detector;gc.collect()
    native.save(job/'language.json',dict(language='uncertain',observations=observations,model=str(MODEL),paid_requests=0))
    language=choose_language(observations)
    native.save(job/'language.json',dict(language=language,observations=observations,model=str(MODEL),paid_requests=0))
    if language=='ja':
        with (job/'extract.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(TOOLS/'extract_aligned_srt.py'),
                str(audio),'--output',str(source),'--work',str(job)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('Japanese alignment failed; see extract.log')
        require_current_timing(source)
    else:
        options={}
        if repair_word_alignment:
            from nosub_recovery import WindowAlignmentRepair
            options['repair_alignment']=WindowAlignmentRepair(audio,language,job/'local_alignment_recovery.json')
        with (job/'extract.log').open('w',encoding='utf-8') as log,redirect_stdout(log):
            extract(audio,str(MODEL),language,True,source,**options)
        report=dict(policy=WORD_POLICY,quality_passed=True,language=language,
            source=native.proof(source),audio=native.proof(audio),audio_seconds=duration,
            model_config=native.proof(MODEL/'config.json'),independent_alignment=False,
            human_audio_verified=False,method='Whisper word timestamps; structural bounds checked, no independent semantic verification')
        native.save(source.with_suffix('.word_clock.json'),report)
        validate_word_clock(source)
    return language


def prepare_sources(work,limit=3):
    import subtitle_batch as batch
    for key in list(os.environ):
        if 'proxy' in key.lower():del os.environ[key]
    work=Path(work).resolve();plan=native.read(work/'plan.json');prepared=[];failures=[];reused=[];deferred=[]
    with runner_lock(work/'extraction/plan.json'):
        priority=work/'source_priority.json'
        ranks={str(fid):i for i,fid in enumerate(native.read(priority)['video_fids'])} if priority.exists() else {}
        items=sorted(plan['items'],key=lambda row:ranks.get(str(row['video']['fid']),len(ranks)))
        selected=[]
        for row in items:
            fid=str(row['video']['fid']);target=work/'translations'/fid;job=work/'extraction/jobs'/fid
            if (target/'content_hold.json').exists():continue
            if (target/'blocked.json').exists() and not language_review_available(target,job):continue
            version=versions.decision(work,row)
            if version['action']=='complete':continue
            if version['action']=='wait':
                versions.remember_wait(work,fid,version)
                continue
            if (target/'media_identity.json').exists() and version['action']!='reuse':continue
            selected.append(row)
            if len(selected)>=min(limit,3):break
        for row in selected:
            fid=str(row['video']['fid']);blocked=work/'translations'/fid/'blocked.json'
            if blocked.exists():
                prior=native.read(blocked);native.save(work/'extraction/jobs'/fid/'language_recovery.json',dict(previous_block=prior))
                prior['speech_review_attempted']=True;native.save(blocked,prior)
        native.save(work/'source_run.json',dict(active=True,pid=os.getpid(),started_at=time.time(),video_fids=[str(r['video']['fid']) for r in selected]))
        def verify_reuse(row,job,version,peer):
            fid=str(row['video']['fid'])
            job=work/'extraction/jobs'/fid;job.mkdir(parents=True,exist_ok=True)
            tool=TOOLS/'nosub_reuse_completed_version.py'
            if not tool.exists():
                raise RuntimeError(
                    'cross-version reuse verification is not part of this distribution: '
                    'blocked before duplicate ASR (see docs/pitfalls.md, "reuse a finished version")')
            command=[sys.executable,'-u',str(tool),'--work',str(work),
                '--target',fid,'--peer',peer,'--affine']
            command.extend(['--peer-work',version['peer_work'],'--publication-journal',version['publication_journal']])
            with (job/'version_reuse.log').open('a',encoding='utf-8') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:raise ValueError('Completed version candidate requires review before duplicate ASR; see version_reuse.log')
        def download(row):
            fid=str(row['video']['fid']);job=work/'extraction/jobs'/fid;job.mkdir(parents=True,exist_ok=True)
            source=job/(Path(row['video']['n']).stem+'.srt');audio=job/'full_audio.wav'
            version=versions.decision(work,row)
            if version['action']=='complete':return row,job,None,'reused'
            peer=version.get('peer_fid') if version['action']=='reuse' else None
            if version['action']=='wait':
                versions.remember_wait(work,fid,version)
                return row,job,None,'waiting'
            if peer:
                verify_reuse(row,job,version,peer)
                return row,job,None,'reused'
            cached=batch.cache_source(row,ROOT/'subtitle/work')
            if cached:return row,cached.parent,cached,None
            if not audio.exists() and not restore_audio(audio,Path(row['video']['n']).stem,fid):
                if free_gb(work)<5:raise RuntimeError('Less than 5 GiB free; source preparation stopped')
                cookie=load_env().get('P115_COOKIES') or ''
                if not cookie:raise RuntimeError('Missing P115_COOKIES (environment or .env)')
                Cloud115(cookie).extract_audio(row['video']['pc'],audio)
            archive_audio(audio,Path(row['video']['n']).stem,fid)
            prune_duplicate_audio_chunks(audio)
            return row,job,source,audio
        try:
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures={pool.submit(download,r):r for r in selected}
                for future in as_completed(futures):
                    row=futures[future];fid=str(row['video']['fid']);target=work/'translations'/fid
                    try:
                        row,job,source,audio=future.result()
                        if source is not None:
                            version=versions.decision(work,row)
                            if version['action']=='reuse':
                                verify_reuse(row,job,version,version['peer_fid'])
                                source=None;audio='reused'
                            elif version['action']=='complete':
                                source=None;audio='reused'
                            elif version['action']=='wait':
                                versions.remember_wait(work,fid,version)
                                source=None;audio='waiting'
                        if source is None and audio in {'reused','waiting'}:
                            (reused if audio=='reused' else deferred).append(fid)
                            print(json.dumps(dict(video_fid=fid,version_action=audio)),flush=True)
                            continue
                        if source.with_suffix('.word_clock.json').exists():validate_word_clock(source)
                        elif source.with_suffix('.alignment.json').exists():require_current_timing(source)
                        else:
                            from nosub_recovery import gpu_slot,reuse_language_evidence
                            reuse_language_evidence(row,audio,job)
                            with gpu_slot():extract_auto(audio,source,job,repair_word_alignment=row.get('repair_word_alignment',False))
                        if source.with_suffix('.word_clock.json').exists():
                            language=native.read(source.with_suffix('.word_clock.json'))['language']
                            archive=ORIG_ARCHIVE_DIR/'multilingual'/fid
                            archive.mkdir(parents=True,exist_ok=True)
                            archived=archive/(source.stem+'.'+language+'.srt')
                            verified_link(source,archived,digest(source),source.stat().st_size)
                        else:
                            archived,note=archive_original(source,ORIG_ARCHIVE_DIR)
                            if archived is None:raise RuntimeError(note)
                        seed=batch.prior_seed(source,ROOT/'subtitle/work',target)
                        state=native.prepare(source,target,seed=seed)
                        native.save(target/'media_identity.json',dict(video=row['video'],source=native.proof(source),plan=native.proof(work/'plan.json')))
                        if (target/'blocked.json').exists():
                            old=native.read(target/'blocked.json')
                            if old.get('reason')!='Language uncertain; retain audio without forcing Japanese' or not old.get('speech_review_attempted'):
                                raise ValueError('Unexpected block; do not release')
                            (target/'blocked.json').unlink()
                        if not state['packets']:batch.finish_local(work,fid)
                        prepared.append(fid)
                        with (work/'source_events.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(event='translation_ready',video_fid=fid))+'\n')
                        print(json.dumps(dict(translation_ready=fid,packets=len(state['packets']))),flush=True)
                    except Exception as exc:
                        failures.append(fid);target.mkdir(parents=True,exist_ok=True)
                        native.save(target/'blocked.json',dict(video_fid=fid,status='failed_source_preparation',reason=str(exc)[:250],no_automatic_retry=True))
                        print(json.dumps(dict(source_failed=fid,error_type=type(exc).__name__)),flush=True)
        finally:
            for row in selected:
                fid=str(row['video']['fid'])
                if fid in prepared or fid in failures or fid in reused or fid in deferred:continue
                failures.append(fid);target=work/'translations'/fid;target.mkdir(parents=True,exist_ok=True)
                native.save(target/'blocked.json',dict(video_fid=fid,status='failed_source_preparation',
                    reason='Source preparation interrupted; preserve checkpoints without automatic full-film retry',no_automatic_retry=True))
            native.save(work/'source_run.json',dict(active=False,finished_at=time.time(),prepared_video_fids=prepared,failed_video_fids=failures,reused_video_fids=reused,deferred_video_fids=deferred))
        return dict(prepared_video_fids=prepared,failed_video_fids=failures,reused_video_fids=reused,deferred_video_fids=deferred)


