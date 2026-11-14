"""Resume a frozen 115 subtitle rebuild manifest; bounded GPU/audio queue."""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

from cloud_common import Cloud115
from audio_archive import archive_audio,restore_audio,digest
from aligned_asr.utterance import pipeline_revision,review_digest
from nosub_pipeline import load_env, validate_subtitles, archive_original, ORIG_ARCHIVE_DIR, free_gb, parse_srt_cues

TOOLS=Path(__file__).resolve().parent
PY=sys.executable
DL_WORKERS=3
ASR_WORKERS=1
TRANSLATE_WORKERS=2
TRANSLATE_REQUEST_WORKERS=2
UPLOAD_WORKERS=2
QUEUE_WATERMARK=12
TIMING_POLICY='utterance_clock_v2'


def sha1(path):
    return digest(path)


def prune_duplicate_audio_chunks(audio):
    """Remove only numbered PCM chunks proven byte-identical to the full WAV."""
    folder=audio.parent/'audio_chunks'
    if not folder.is_dir():return
    chunks=sorted(p for p in folder.glob('*.wav') if len(p.stem)==8 and p.stem.isdecimal())
    if not chunks:return
    digest=hashlib.sha1();frames=0
    for chunk in chunks:
        if chunk.resolve().parent!=folder.resolve():return
        with wave.open(str(chunk),'rb') as handle:
            if (handle.getframerate(),handle.getnchannels(),handle.getsampwidth())!=(16000,1,2):return
            if int(chunk.stem)*16000!=frames:return
            frames+=handle.getnframes()
            while part:=handle.readframes(262144):digest.update(part)
    actual=hashlib.sha1()
    with wave.open(str(audio),'rb') as handle:
        if (handle.getframerate(),handle.getnchannels(),handle.getsampwidth())!=(16000,1,2) or handle.getnframes()!=frames:return
        while part:=handle.readframes(262144):actual.update(part)
    if actual.digest()!=digest.digest():return
    for chunk in chunks:chunk.unlink()
    if not any(folder.iterdir()):folder.rmdir()


def require_current_timing(ja,work=None):
    ja=Path(ja)
    job=Path(work) if work is not None else ja.parent
    metadata=ja.with_suffix('.alignment.json')
    if not metadata.exists():raise RuntimeError('Missing timing policy metadata; re-extraction required')
    report=json.loads(metadata.read_text(encoding='utf-8'))
    if not isinstance(report,dict):raise ValueError('Invalid timing metadata object')
    if report.get('timing_policy')!=TIMING_POLICY:
        raise RuntimeError('Obsolete mixed-clock subtitle; re-extraction required before translation/publication')
    if report.get('quality_passed') is not True:
        raise RuntimeError('Subtitle timing quality did not pass; translation/publication refused')
    if report.get('pipeline_revision')!=pipeline_revision():
        raise RuntimeError('Subtitle extraction code revision changed; revalidation required')
    if report.get('boundary_review_sha1'):
        from boundary_review import verify_manifest
        verify_manifest(job,report)
    if report.get('source_subset_review_sha1'):
        from source_subset import verify
        verify(job,report)
    if report.get('manual_review_digest')!=review_digest(job):
        raise RuntimeError('Manual audio review changed; revalidation required')
    audio=job/'full_audio.wav'
    if audio.exists() and report.get('audio_sha1')!=sha1(audio):
        raise RuntimeError('Source audio bytes changed; re-extraction required')
    if not ja.exists() or report.get('source_sha1')!=sha1(ja):
        raise RuntimeError('Subtitle timing metadata source hash mismatch; re-extraction required')
    return report


def timing_is_current(ja):
    try:require_current_timing(ja);return True
    except (OSError,ValueError,RuntimeError):return False


def sync_manual_review(work,job,fid):
    total=work/'manual_audio_reviews.json'
    regions=json.loads(total.read_text('utf-8')).get(fid,[]) if total.exists() else []
    target=job/'manual_audio_review.json'
    desired={'non_speech_regions':[r for r in regions if r.get('classification')=='background_music_only'],
             'utterance_confirmations':[r for r in regions if r.get('classification')=='utterance_confirmed']}
    current=json.loads(target.read_text('utf-8')) if target.exists() else {'non_speech_regions':[],'utterance_confirmations':[]}
    if current!=desired:atomic_json(target,desired)


def preserve_evidence(job, paths, move=False):
    """Keep every distinct version, including checkpoints and context, intact."""
    for path in paths:
        path=Path(path)
        if not path.is_file():continue
        dest=Path(job)/'obsolete'/sha1(path)/path.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(path,dest)
        if move:path.unlink()


def translation_paths(zh):
    return [zh,zh.with_suffix(zh.suffix+'.checkpoint'),zh.with_suffix('.global_context.json')]


def prepare_translation(ja,zh,job):
    report=require_current_timing(ja)
    binding=zh.with_suffix('.translation.json')
    try:cached=json.loads(binding.read_text(encoding='utf-8'))
    except (OSError,ValueError):cached={}
    if not isinstance(cached,dict):cached={}
    current=dict(source_sha1=report['source_sha1'],timing_policy=TIMING_POLICY)
    if any(cached.get(key)!=value for key,value in current.items()):
        preserve_evidence(job,[*translation_paths(zh),binding,Path(job)/'translate.log'],move=True)
        cached=current
        atomic_json(binding,cached)
    if zh.exists() and cached.get('output_sha1')!=sha1(zh):
        # A final file without a committed output digest is not a completed result.
        preserve_evidence(job,[zh],move=True)
        cached.pop('output_sha1',None)
        atomic_json(binding,cached)
    return binding,cached


def require_translation(ja,zh):
    report=require_current_timing(ja)
    cached=json.loads(zh.with_suffix('.translation.json').read_text(encoding='utf-8'))
    if not isinstance(cached,dict):raise ValueError('Invalid translation metadata object')
    if (cached.get('source_sha1')!=report['source_sha1'] or
        cached.get('timing_policy')!=TIMING_POLICY or cached.get('output_sha1')!=sha1(zh)):
        raise RuntimeError('Translation is not bound to the current source and output bytes')
    if cached.get('protocol')=='grounded-json-v1':
        from grounded_translation import require_audit
        audit=zh.with_suffix(zh.suffix+'.translation')/'certificate.json'
        if cached.get('translation_audit_sha1')!=sha1(audit):raise RuntimeError('Translation audit changed')
        require_audit(ja,zh,ja.parent/'source_review.json')
    return cached


def publication_complete(record,ja,zh):
    try:
        binding=require_translation(ja,zh)
        journal=record.get('replacement',{})
        return (record.get('status')=='uploaded' and record.get('quality_passed') is True and
            record.get('timing_policy')==TIMING_POLICY and record.get('source_sha1')==binding['source_sha1'] and
            journal.get('timing_policy')==TIMING_POLICY and journal.get('source_sha1')==binding['source_sha1'] and
            journal.get('translation_sha1')==binding['output_sha1'] and
            journal.get('stage')=='complete' and journal.get('sha1')==binding['output_sha1'] and
            record.get('cloud_sha1')==binding['output_sha1'] and
            str(record.get('cloud_fid'))==str(journal.get('stage_fid')) and bool(journal.get('stage_fid')))
    except (OSError,ValueError,RuntimeError):return False


def revision_journal(record,binding):
    journal=record.get('replacement')
    if journal and (journal.get('source_sha1')!=binding['source_sha1'] or
                    journal.get('timing_policy')!=TIMING_POLICY or
                    journal.get('translation_sha1')!=binding['output_sha1']):
        prior=record.pop('revision_baseline',None) or {key:copy.deepcopy(record[key]) for key in ('replacement','cloud_fid','cloud_sha1','at',
            'source_sha1','timing_policy','quality_passed','original_archive') if key in record}
        record.setdefault('prior_publications',[]).append(prior)
        record.pop('replacement',None)
        record.pop('cloud_fid',None);record.pop('cloud_sha1',None)
    journal=record.setdefault('replacement',{})
    journal.update(source_sha1=binding['source_sha1'],timing_policy=TIMING_POLICY,
                   translation_sha1=binding['output_sha1'])
    return journal


def seed_revision(cloud,original,local,backup,journal,save):
    """Seed ALL retained rows; Cloud115 cannot auto-discover them once seeded."""
    fid=str(original['fid']);target=journal['target_name']
    stage_name=Path(target).stem+'.mvplus-stage-'+fid+'.srt'
    if 'retained_old' in journal:
        items=cloud.list_dir(str(original['cid']))
        allowed={str(journal.get('stage_fid'))}|{row['fid'] for row in journal['retained_old']
            if row['old_name']==target}
        if any(row.get('n')==target and str(row.get('fid')) not in allowed for row in items):
            raise RuntimeError('Canonical subtitle changed after revision snapshot; refusing publication')
        # Retry a failed pre-upload stage retain from its persisted intent.
        for row in journal['retained_old']:
            if row['old_name']!=stage_name:continue
            current=next((x for x in cloud.list_dir(str(original['cid'])) if str(x.get('fid'))==row['fid']),None)
            if not current or current['n'] not in (row['old_name'],row['new_name']):
                raise RuntimeError('Old revision stage changed during retain')
            if current['n']==row['old_name'] and not cloud.rename(row['fid'],row['new_name']):
                raise RuntimeError('Old revision stage retain failed')
        return
    items=cloud.list_dir(str(original['cid']))
    old=next((row for row in items if str(row.get('fid'))==fid),None)
    if not old:raise RuntimeError('Frozen original subtitle missing; refusing revision')
    expected_old={original['n'],Path(original['n']).stem+'旧.srt',
                  Path(original['n']).stem+'('+fid+')旧.srt'}
    if old['n'] not in expected_old:raise RuntimeError('Frozen original subtitle has an unknown retained filename')
    if original.get('sha') and str(old.get('sha','')).upper()!=original['sha'].upper():
        raise RuntimeError('Frozen original subtitle bytes changed; refusing revision')
    if not backup.exists():
        if old['n']!=original['n']:raise RuntimeError('Renamed original has no verified frozen backup')
        cloud.download_verified(old,backup)
    if original.get('sha') and sha1(backup)!=original['sha'].upper():
        raise RuntimeError('Frozen original rollback hash mismatch')
    # A failed older revision may have left incompatible staged bytes. Preserve
    # them under an old name rather than deleting or reusing them.
    retained=[];occupied={row['n'] for row in items}
    candidates=[row for row in items if str(row.get('fid'))==fid or row.get('n')==target or
        (row.get('n')==stage_name and str(row.get('sha','')).upper()!=sha1(local))]
    for row in candidates:
        rowfid=str(row['fid']);old_name=row['n']
        path=backup if rowfid==fid else backup.parent.parent/rowfid/old_name
        if not path.exists():cloud.download_verified(row,path)
        if row.get('sha') and sha1(path)!=row['sha'].upper():
            raise RuntimeError('Revision retained subtitle backup hash mismatch')
        if rowfid==fid and old_name!=original['n'] and old_name!=target:
            # Already retained: rollback must keep its existing old filename.
            new_name=old_name
        else:
            new_name=Path(old_name).stem+'旧.srt'
            if new_name in occupied:new_name=Path(old_name).stem+'('+rowfid+')旧.srt'
            if new_name in occupied:raise RuntimeError('Revision retained filename collision')
        occupied.add(new_name)
        retained.append(dict(fid=rowfid,old_name=old_name,new_name=new_name,sha=row.get('sha')))
    journal['retained_old']=retained;save()
    # Clear the fixed staging filename before Cloud115 chooses the next upload.
    for row in retained:
        if row['old_name']==stage_name:
            if not cloud.rename(row['fid'],row['new_name']):raise RuntimeError('Old revision stage retain failed')
            save()


def validate_current_subtitles(ja,zh,audio):
    report=require_current_timing(ja)
    if audio.exists():return validate_subtitles(ja,zh,audio)
    # Completed source metadata retains the measured audio duration even after
    # the old runner removed the WAV. Structural/translation checks still run.
    duration=report.get('audio_seconds')
    if isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0:
        return False,'Missing verified audio duration in current timing metadata'
    ok,note=validate_subtitles(ja,zh,None)
    if not ok:return ok,note
    for path in (ja,zh):
        if any(not 0<=start<end<=duration+5 for _,start,end,_ in parse_srt_cues(path)):
            return False,'Subtitle timing exceeds verified cached audio duration'
    return True,note.replace(' / 时长范围校验：**未做**（未提供视频 → 显式跳过；生产路径不允许跳过）',
                            f' / 起止在已验证音轨时长内（{duration:.1f}s）')


def atomic_json(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',suffix='.part',delete=False) as handle:
        temp=Path(handle.name)
        handle.write(json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'))
    try:
        # Windows readers can briefly deny replacement while opening state.json.
        # Retry the same immutable temporary file without weakening atomicity.
        for attempt in range(6):
            try:temp.replace(path);break
            except PermissionError:
                if attempt==5:raise
                time.sleep(.05*(2**attempt))
    finally:temp.unlink(missing_ok=True)


class RunnerBusy(RuntimeError):pass


@contextmanager
def runner_lock(plan_path):
    """Share the existing crash-safe Windows lock with scripted callers."""
    import msvcrt
    lockpath=Path(plan_path).resolve().parent/'runner.lock'
    with lockpath.open('a+b') as handle:
        if lockpath.stat().st_size==0:handle.write(b'0');handle.flush()
        handle.seek(0)
        try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:raise RunnerBusy('Already running; no duplicate worker started.') from None
        try:yield
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def run(plan_path,extract_only=False,only_fids=None,publish_only=False,budget_file=None,budget_cny=None,on_extracted=None):
    plan_path=Path(plan_path).resolve();work=plan_path.parent
    if extract_only and publish_only:raise ValueError('Choose extract-only or publish-only')
    if on_extracted is not None and not extract_only:raise ValueError('Native source callback requires extract-only mode')
    if not extract_only and not publish_only and (work/'translation_provider_override.json').exists():
        raise RuntimeError('User selected Codex agent translation; DeepSeek generation disabled for this batch')
    if not extract_only and not publish_only and (work/'translation_credit_hold.json').exists():
        raise RuntimeError('DeepSeek balance unavailable; preserve checkpoints and resolve translation_credit_hold.json before paid translation')
    from translation_budget import paid_cli_args
    paid_args=[]
    if (work/'extraction_pause.json').exists():
        raise RuntimeError('Batch extraction paused for defect verification; see extraction_pause.json')
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    state_path=work/'state.json'
    state=json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {'items':{},'started':time.time()}
    lock=threading.RLock()
    def save():
        with lock:
            state['updated']=time.time()
            atomic_json(state_path,state)
    def update(fid,status,**extra):
        with lock:
            record=state['items'].setdefault(fid,{})
            if not status.startswith('failed'):record.pop('error',None)
            record.update(status=status,at=time.time(),**extra)
            save()
        print(f'[{fid}] {status}',flush=True)
    def wait_quality_release(fid,stage):
        # A batch-scoped safety hold preserves download/ASR workers and all
        # checkpoints. It must be cleared after the recorded defect is verified.
        hold=work/'quality_hold.json'
        if hold.exists():
            update(fid,'waiting_quality_review',quality_stage=stage)
        while hold.exists():time.sleep(10)
    def context_held(fid,job):
        if not (job/'translation_content_hold.json').exists():return False
        with lock:status=state['items'].get(fid,{}).get('status')
        if status!='blocked_content':update(fid,'waiting_source_context',translation_pending=True)
        return True
    # Four independent stages. Backpressure limits stored audio, never GPU/LLM
    # coupling: download the next film while this film is being recognized.
    inputs=queue.Queue()
    audios=queue.Queue(maxsize=QUEUE_WATERMARK)
    subtitles=queue.Queue(maxsize=QUEUE_WATERMARK)
    translated=queue.Queue(maxsize=QUEUE_WATERMARK)
    pending=[]
    for row in plan['items']:
        fid=str(row['subtitle']['fid']);job=work/'jobs'/fid;name=Path(row['video']['n']).stem
        job.mkdir(parents=True,exist_ok=True);sync_manual_review(work,job,fid)
        if not publication_complete(state['items'].get(fid,{}),job/(name+'.srt'),job/(name+'.zh-cn.srt')):
            record=state['items'].get(fid,{})
            if record.get('replacement') and 'revision_baseline' not in record:
                record['revision_baseline']={key:copy.deepcopy(record[key]) for key in ('replacement','cloud_fid',
                    'cloud_sha1','at','source_sha1','timing_policy','quality_passed','original_archive') if key in record}
            pending.append(row)
    pending.sort(key=lambda r:(r['subtitle']['n']!='FNS-122.srt',int(r['video'].get('s') or 0)))
    state['required_total']=len(plan['items'])
    state['completed_current']=len(plan['items'])-len(pending)
    state['batch_complete']=not pending
    if not pending:
        state['active']=False;state['finished']=time.time();save()
        print(f'CURRENT POLICY COMPLETE {state["completed_current"]}/{state["required_total"]}',flush=True)
        return state
    env=load_env()
    if only_fids is not None:
        selected={str(fid) for fid in only_fids}
        pending=[row for row in pending if str(row['subtitle']['fid']) in selected]
        if not pending:
            print('No selected pending items; batch state retained.',flush=True)
            return state
    if not extract_only and not publish_only:
        paid_args=paid_cli_args(budget_file,budget_cny)
    cookie=env.get('P115_COOKIES') or ''
    if not cookie:
        raise RuntimeError('Missing P115_COOKIES (environment or .env)')
    state['pid']=os.getpid();state['active']=True
    state['execution_mode']='publish_only' if publish_only else ('extract_only' if extract_only else 'full')
    state['workers']={'download':DL_WORKERS,'asr':ASR_WORKERS,'translate':TRANSLATE_WORKERS,'upload':UPLOAD_WORKERS}
    save()
    for row in pending:inputs.put(row)
    for _ in range(DL_WORKERS):inputs.put(None)
    def download_loop():
        cloud=Cloud115(cookie)
        while True:
            row=inputs.get()
            if row is None:return
            fid=str(row['subtitle']['fid']);job=work/'jobs'/fid
            job.mkdir(parents=True,exist_ok=True)
            with lock:already_blocked=state['items'].get(fid,{}).get('status')=='blocked_content'
            if (already_blocked or (not extract_only and not publish_only)) and context_held(fid,job):continue
            name=Path(row['video']['n']).stem
            ja=job/(name+'.srt');audio=job/'full_audio.wav'
            try:
                if timing_is_current(ja):
                    update(fid,'cached_extraction',name=row['subtitle']['n'])
                    audios.put((row,job,ja,audio));continue
                if publish_only:raise RuntimeError('Publish-only requires an already validated local source; no download/extraction permitted')
                audio_started=time.perf_counter()
                audio_source='existing_local' if audio.exists() else 'remote_chunks'
                update(fid,'downloading_audio',name=row['subtitle']['n'])
                if not audio.exists():
                    while free_gb(work)<5:time.sleep(30)
                    if restore_audio(audio,name,row['video']['fid']):
                        audio_source='restored_archive'
                        update(fid,'restored_archived_audio')
                    else:cloud.extract_audio(row['video']['pc'],audio)
                update(fid,'downloaded_audio',audio_ready_seconds=time.perf_counter()-audio_started,
                       audio_source=audio_source)
                try:
                    archived,meta=archive_audio(audio,name,row['video']['fid'])
                    update(fid,'downloaded_audio',audio_archive=str(archived),audio_archive_sha1=meta['sha1'])
                except (OSError,ValueError,wave.Error,EOFError) as error:
                    update(fid,'downloaded_audio',audio_archive_error=str(error)[:200])
                try:prune_duplicate_audio_chunks(audio)
                except (OSError,EOFError,wave.Error):print(f'[{fid}] redundant chunk cleanup skipped; full audio retained',flush=True)
                audios.put((row,job,ja,audio))
            except Exception as exc:update(fid,'failed_download',error=str(exc)[:250])
    def extract_loop():
        while True:
            task=audios.get()
            if task is None:return
            row,job,ja,audio=task;fid=str(row['subtitle']['fid'])
            try:
                if not extract_only and not publish_only and context_held(fid,job):continue
                if not timing_is_current(ja):
                    sync_manual_review(work,job,fid)
                    preserve_evidence(job,[ja,ja.with_suffix('.alignment.json'),job/'extract.log',
                        *[job/name for name in ('asr.json','alignment.json','event_captioning.json',
                            'zero_event_repair.json','content_verification.json','vad_evidence.json')]])
                    update(fid,'recognizing')
                    result=subprocess.run([str(PY),str(TOOLS/'extract_aligned_srt.py'),str(audio),
                        '--output',str(ja),'--work',str(job)],capture_output=True,text=True,encoding='utf-8',errors='replace')
                    (job/'extract.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
                    if result.returncode:raise RuntimeError('ASR/align failed; see job extract.log')
                report=require_current_timing(ja)
                if not extract_only and not publish_only:
                    from source_review import ensure_review
                    ensure_review(ja,audio,job,job/'raw_asr.json')
                archived,archive_note=archive_original(ja,ORIG_ARCHIVE_DIR)
                if archived is None:raise RuntimeError('Original archive failed: '+archive_note)
                update(fid,'extracted',extracted_source_sha1=report['source_sha1'])
                subtitles.put(task)
            except Exception as exc:update(fid,'failed_extract',error=str(exc)[:250])
    def translate_loop():
        while True:
            task=subtitles.get()
            if task is None:return
            row,job,ja,audio=task;fid=str(row['subtitle']['fid'])
            zh=job/(Path(row['video']['n']).stem+'.zh-cn.srt')
            try:
                if not extract_only and not publish_only and context_held(fid,job):continue
                require_current_timing(ja)
                if extract_only:
                    if on_extracted is not None:on_extracted(row,ja)
                    update(fid,'extracted',translation_pending=True)
                    continue
                if publish_only:
                    require_translation(ja,zh)
                    update(fid,'translated',translation_provider='codex_subagent')
                    translated.put((*task,zh))
                    continue
                wait_quality_release(fid,'translate')
                binding,cached=prepare_translation(ja,zh,job)
                if not zh.exists():
                    update(fid,'translating_offpeak_only')
                    with (job/'translate.log').open('w',encoding='utf-8') as out:
                        result=subprocess.run([str(PY),str(TOOLS/'translate_srt_batch.py'),str(ja),
                            '--output',str(zh),'--model','deepseek-flash','--url','https://api.deepseek.com',
                            '--system-prompt',str(TOOLS/'prompts/system_prompt_v9_grounded.txt'),
                            '--grounded','--source-review',str(job/'source_review.json'),*paid_args,
                            '--batch-size','16','--max-workers',str(TRANSLATE_REQUEST_WORKERS),
                            '--max-output-tokens','16384','--reasoning-effort','high','--timeout','660',
                            '--off-peak-only'],stdout=out,stderr=subprocess.STDOUT)
                    if result.returncode==3:
                        update(fid,'waiting_source_review',translation_pending=True);continue
                    if result.returncode:raise RuntimeError('Translation failed; see job translate.log')
                    require_current_timing(ja)
                    if sha1(ja)!=cached['source_sha1']:raise RuntimeError('Source changed during translation')
                    cached['output_sha1']=sha1(zh)
                    audit=zh.with_suffix(zh.suffix+'.translation')/'certificate.json'
                    cached.update(protocol='grounded-json-v1',translation_audit_sha1=sha1(audit))
                    atomic_json(binding,cached)
                require_translation(ja,zh)
                update(fid,'translated')
                translated.put((*task,zh))
            except Exception as exc:update(fid,'failed_translate',error=str(exc)[:250])
    def upload_loop():
        cloud=Cloud115(cookie)
        while True:
            task=translated.get()
            if task is None:return
            row,job,ja,audio,zh=task;fid=str(row['subtitle']['fid'])
            try:
                if (job/'translation_content_hold.json').exists():
                    raise RuntimeError('Source context review pending; publication withheld')
                wait_quality_release(fid,'publish')
                require_current_timing(ja)
                binding=require_translation(ja,zh)
                ok,note=validate_current_subtitles(ja,zh,audio)
                if not ok:raise RuntimeError(note)
                archive=ORIG_ARCHIVE_DIR/(ja.stem+'.ja.srt')
                preserve_evidence(job,[archive,archive.with_name(archive.name+'.prev')])
                archived,archive_note=archive_original(ja,ORIG_ARCHIVE_DIR)
                if archived is None:raise RuntimeError(archive_note)
                update(fid,'publishing',validation=note,original_archive=str(archived))
                with lock:
                    journal=copy.deepcopy(revision_journal(state['items'][fid],binding))
                    journal['target_name']=Path(row['video']['n']).stem+'.srt'
                    state['items'][fid]['replacement']=copy.deepcopy(journal)
                    save()
                def save_journal():
                    # Cloud115 mutates this private journal; serialize a stable
                    # snapshot under the same lock as every worker status.
                    with lock:
                        state['items'][fid]['replacement']=copy.deepcopy(journal)
                        save()
                backup=work/'rollback'/fid/row['subtitle']['n']
                seed_revision(cloud,row['subtitle'],zh,backup,journal,save_journal)
                # Gate again immediately before an external publication.
                wait_quality_release(fid,'publish')
                require_translation(ja,zh)
                if (job/'translation_content_hold.json').exists():
                    raise RuntimeError('Source context review pending; publication withheld')
                cloud.replace_subtitle(row['subtitle'],zh,backup,journal,save_journal,target_name=journal['target_name'])
                require_translation(ja,zh)
                if journal.get('stage')!='complete' or journal.get('sha1')!=sha1(zh):
                    raise RuntimeError('Publication journal does not verify current Chinese subtitle bytes')
                update(fid,'uploaded',cloud_fid=journal['stage_fid'],cloud_sha1=journal['sha1'],
                       source_sha1=binding['source_sha1'],timing_policy=TIMING_POLICY,quality_passed=True)
                with lock:
                    state['items'][fid].pop('revision_baseline',None)
                    save()
                # Source, translation, duration metadata and all drafts remain
                # available for local audit and independent stage resumption.
            except Exception as exc:update(fid,'failed_publish',error=str(exc)[:250])
    groups=[]
    for name,count,target in [('download',DL_WORKERS,download_loop),('asr',ASR_WORKERS,extract_loop),
                              ('translate',TRANSLATE_WORKERS,translate_loop),('upload',UPLOAD_WORKERS,upload_loop)]:
        group=[threading.Thread(target=target,name=f'{name}-{i}') for i in range(count)]
        groups.append(group)
        for worker in group:worker.start()
    print(f'PARALLEL download={DL_WORKERS} asr={ASR_WORKERS} translate={TRANSLATE_WORKERS} upload={UPLOAD_WORKERS} watermark={QUEUE_WATERMARK}',flush=True)
    # Close each downstream stage only after every upstream worker has finished.
    for i,group in enumerate(groups):
        for worker in group:worker.join()
        if i<3:
            target_queue=(audios,subtitles,translated)[i]
            for _ in groups[i+1]:target_queue.put(None)
    state['completed_current']=sum(publication_complete(state['items'].get(str(row['subtitle']['fid']),{}),
        work/'jobs'/str(row['subtitle']['fid'])/(Path(row['video']['n']).stem+'.srt'),
        work/'jobs'/str(row['subtitle']['fid'])/(Path(row['video']['n']).stem+'.zh-cn.srt')) for row in plan['items'])
    state['batch_complete']=state['completed_current']==len(plan['items'])
    state['active']=False;state['finished']=time.time();save()
    print(json.dumps(dict(Counter(r.get('status') for r in state['items'].values())),ensure_ascii=False),flush=True)
    return state


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser()
    p.add_argument('plan',type=Path)
    p.add_argument('--extract-only',action='store_true',help='Download and extract without API translation or cloud publication')
    p.add_argument('--only-fid',action='append',help='Retry selected subtitle FID within the original frozen plan (repeatable)')
    p.add_argument('--publish-only',action='store_true',help='Validate and publish existing bound translations; never call a translation API or extract audio')
    p.add_argument('--budget-file',type=Path);p.add_argument('--budget-cny')
    args=p.parse_args()
    try:
        with runner_lock(args.plan):
            run(args.plan,extract_only=args.extract_only,only_fids=args.only_fid,publish_only=args.publish_only,
                budget_file=args.budget_file,budget_cny=args.budget_cny)
    except RunnerBusy as exc:
        print(str(exc));sys.exit(0)
