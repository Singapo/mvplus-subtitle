"""SRT-date batch: scripts own scanning, media/ASR, packets and publication.

The native agent's only task is reading translation packets and writing output.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout,contextmanager
from datetime import datetime, timezone, timedelta
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import native_subtitles as native
import config

ROOT=config.repo_root()
VIDEO={'.mp4','.mkv','.avi','.wmv','.mov','.ts','.m2ts','.m4v','.webm'}
TZ=timezone(timedelta(hours=8))
DELIVERY_LOCK=threading.RLock()


def read(path):return native.read(path)
def save(path,data):native.save(path,data)


def history_files(history,name):
    if shutil.which('rg'):
        result=subprocess.run(['rg','--files','--hidden','--no-ignore','-g',name,str(history)],
            capture_output=True,text=True,encoding='utf-8',errors='replace')
        if result.returncode in (0,1):return [Path(line) for line in result.stdout.splitlines() if line]
    return list(Path(history).rglob(name))


def stem(name):
    value=Path(name).stem
    value=re.sub(r'旧$','',value)
    value=re.sub(r'\.(ja|jp|jpn)$','',value,flags=re.I)
    return value.casefold()


def completed_sources(inventory,history):
    """Follow verified replacement chains, not Chinese-language presence."""
    files={str(r['fid']):r for f in inventory['directories'].values() for r in f['files'] if r.get('fid')}
    edges=defaultdict(set);final=set();held=set()
    journals=[]
    for path in history_files(history,'publication.json'):
        data=read(path)
        journals.extend(data.get('items',{}).values() if 'items' in data else [data])
    for path in Path(history).glob('*/state.json'):
        journals.extend(r['replacement'] for r in read(path).get('items',{}).values() if isinstance(r,dict) and r.get('status')=='uploaded' and r.get('replacement'))
    for path in history_files(history,'content_hold.json'):
        held.add(path.parent.name)
    for journal in journals:
        if journal.get('stage')!='complete' or not journal.get('stage_fid') or not journal.get('sha1'):continue
        dest=(str(journal['stage_fid']),journal['sha1'].upper())
        current=files.get(dest[0])
        if current and str(current.get('sha','')).upper()==dest[1] and not Path(current['n']).stem.endswith('旧'):
            final.add(dest)
        for old in journal.get('retained_old',[]):
            if old.get('sha'):edges[(str(old['fid']),old['sha'].upper())].add(dest)
    done=set(final)
    for _ in range(len(edges)+1):
        added={src for src,targets in edges.items() if targets & done}-done
        if not added:break
        done.update(added)
    return done,held


def plan_batch(inventory,work,start,before,completed=None,held=None):
    work=Path(work).resolve();work.mkdir(parents=True,exist_ok=True)
    lo=int(datetime.fromisoformat(start).replace(tzinfo=TZ).timestamp())
    hi=int(datetime.fromisoformat(before).replace(tzinfo=TZ).timestamp())
    if lo>=hi:raise ValueError('Invalid date interval')
    completed=completed or set();held=held or set();allvideos=[]
    for cid,folder in inventory['directories'].items():
        allvideos.extend(dict(r,cid=str(cid)) for r in folder['files'] if r.get('fid') and Path(r['n']).suffix.lower() in VIDEO)
    ledger=[];groups={}
    for cid,folder in inventory['directories'].items():
        for meta in folder['files']:
            if not meta.get('fid') or Path(meta['n']).suffix.lower()!='.srt' or not lo<=int(meta.get('tp') or 0)<hi:continue
            srt=dict(meta,cid=str(cid));record=dict(subtitle=srt,folder=folder['path'])
            if (str(srt['fid']),str(srt.get('sha','')).upper()) in completed:
                record['status']='verified_completed';ledger.append(record);continue
            candidates=[v for v in allvideos if stem(v['n'])==stem(srt['n'])]
            if not candidates:
                base=lambda n:re.sub(r'\(\d+\)$','',stem(n))
                candidates=[v for v in allvideos if base(v['n'])==base(srt['n'])]
            local=[v for v in candidates if v['cid']==str(cid)]
            candidates=local or candidates
            if len(candidates)!=1:
                record.update(status='missing_video' if not candidates else 'ambiguous_video',candidate_video_fids=[str(v['fid']) for v in candidates]);ledger.append(record);continue
            video=candidates[0];fid=str(video['fid']);record['video_fid']=fid
            if fid in held or str(srt['fid']) in held:
                record['status']='content_isolated';ledger.append(record);continue
            record['status']='queued';ledger.append(record)
            if fid not in groups:
                folder_files=inventory['directories'][video['cid']]['files']
                canonical=[dict(r,cid=video['cid']) for r in folder_files if r.get('fid') and r['n']==Path(video['n']).stem+'.srt']
                if len(canonical)>1:
                    record['status']='ambiguous_canonical_subtitle';continue
                primary=canonical[0] if canonical else dict(fid=fid,cid=video['cid'],n=Path(video['n']).stem+'.srt',placeholder=True)
                groups[fid]=dict(cid=video['cid'],video=video,subtitle=primary,targets=[])
                if canonical and (str(primary['fid']),str(primary.get('sha','')).upper()) in completed:
                    groups[fid]['publication_reuse']=primary
            groups[fid]['targets'].append(srt)
            if groups[fid].get('publication_reuse'):record['status']='queued_old_retention_only'
    result=dict(from_date=start,before_date=before,date_field='subtitle.tp',ledger=ledger,items=list(groups.values()))
    path=work/'plan.json'
    if path.exists() and read(path)!=result:raise ValueError('Frozen batch plan differs; use another work directory')
    save(path,result)
    extraction=work/'extraction';extraction.mkdir(exist_ok=True)
    save(extraction/'plan.json',dict(items=result['items'],extract_only=True))
    save(extraction/'translation_provider_override.json',dict(provider='native_subagent',deepseek_disabled=True))
    summary=dict(raw_srt_count=len(ledger),video_jobs=len(groups),statuses=dict(Counter(r['status'] for r in ledger)),cloud_writes=0)
    save(work/'summary.json',summary)
    return summary


def scan(work):
    from cloud_common import Cloud115
    from nosub_pipeline import load_env
    work=Path(work);work.mkdir(parents=True,exist_ok=True);path=work/'inventory.json'
    data=read(path) if path.exists() else dict(directories={},pending=[dict(cid='0',path='/')])
    cloud=Cloud115(load_env()['P115_COOKIES'])
    while data['pending']:
        current=data['pending'][0];rows=cloud.list_dir(current['cid'])
        data['directories'][current['cid']]=dict(path=current['path'],files=rows)
        data['pending'].pop(0)
        for row in rows:
            if not row.get('fid') and row.get('cid') and str(row['cid']) not in data['directories'] and not any(x['cid']==str(row['cid']) for x in data['pending']):
                data['pending'].append(dict(cid=str(row['cid']),path=current['path'].rstrip('/')+'/'+row['n']))
        save(path,data)
    return path


def cache_source(row,history):
    from reprocess_subtitles import require_current_timing
    for plan in history_files(history,'plan.json'):
        try:items=read(plan).get('items',[])
        except (ValueError,OSError):continue
        for prior in items:
            if str(prior.get('video',{}).get('fid'))!=str(row['video']['fid']) or prior['video'].get('sha')!=row['video'].get('sha'):continue
            source=plan.parent/'jobs'/str(prior.get('subtitle',{}).get('fid'))/(Path(row['video']['n']).stem+'.srt')
            try:require_current_timing(source)
            except (OSError,ValueError,RuntimeError):continue
            return source
    return None


def prior_seed(source,history,target):
    """Reuse only same-source decisions whose actual native receipts still verify."""
    source_hash=native.proof(source)['sha256'];best=None
    for path in history_files(history,'manifest.json'):
        if path.parent.name!='delivery' or path.is_relative_to(target):continue
        data=read(path)
        if data.get('policy')!=native.POLICY or data.get('source',{}).get('sha256')!=source_hash:continue
        try:
            native.checked(path.parent.parent);native.check_proofs(data['inputs']);native.check_proofs([data['output']])
        except (OSError,ValueError,KeyError):continue
        rows=[[r['source']['id'],r['translation'],r['status'],r['uncertain_span'],r['reason']]
              for r in data['ledger'] if r['status'] in {'ok','partial','sound','filler'}]
        if rows and (best is None or len(rows)>len(best[0])):best=(rows,path,data)
    if best is None:return None
    rows,path,data=best;target.mkdir(parents=True,exist_ok=True);output=target/'seed.json'
    save(output,dict(source_sha256=source_hash,rows=rows,proofs=[native.proof(path),native.proof(path.parent.parent/'job.json'),*data['inputs']]))
    return output


SOURCE_BACKLOG_LIMIT=500


def prepare_sources(work,limit=3):
    import reprocess_subtitles as pipeline
    work=Path(work).resolve();plan=read(work/'plan.json');prepared=[];guard=threading.Lock();progress=sys.stdout
    if plan.get('source_preparer')=='multilingual_word_clock_v1':
        from nosub_sources import prepare_sources as multilingual_sources
        return multilingual_sources(work,limit)
    extraction_plan=work/'extraction/plan.json'
    with pipeline.runner_lock(extraction_plan):
        # Reuse the documented watermark rather than creating another queue/pool.
        # Extraction is limited by disk, not by the translation queue: build the
        # daytime backlog so off-peak workers have work ready.
        slots=max(0,SOURCE_BACKLOG_LIMIT-len(ready(work)['translation_tasks']))
        eligible=[r for r in plan['items'] if not r.get('publication_reuse') and
                  not (work/'translations'/str(r['video']['fid'])/'content_hold.json').exists() and
                  not (work/'translations'/str(r['video']['fid'])/'blocked.json').exists() and not (
                  (work/'translations'/str(r['video']['fid'])/'job.json').exists() and
                  (work/'translations'/str(r['video']['fid'])/'media_identity.json').exists())]
        # Videos that never had a subtitle are the priority: nobody can watch them yet.
        eligible.sort(key=lambda r: 0 if (r.get('subtitle') or {}).get('placeholder') else 1)
        selected=eligible[:min(limit,slots)]
        save(work/'source_run.json',dict(active=True,pid=os.getpid(),
            started_at=time.time(),video_fids=[str(r['video']['fid']) for r in selected]))
        def export(row,source):
            fid=str(row['video']['fid']);target=work/'translations'/fid
            seed=prior_seed(source,ROOT/'subtitle/work',target)
            native_state=native.prepare(source,target,seed=seed)
            save(target/'media_identity.json',dict(video=row['video'],source=native.proof(source),plan=native.proof(work/'plan.json')))
            with guard:
                prepared.append(fid)
                if native_state['packets']:
                    with (work/'source_events.jsonl').open('a',encoding='utf-8') as events:
                        events.write(json.dumps(dict(event='translation_ready',video_fid=fid))+'\n')
            if native_state['packets']:
                print(json.dumps(dict(translation_ready=fid)),file=progress,flush=True)
            else:finish_local(work,fid)
        pending=[]
        try:
            for row in selected:
                source=cache_source(row,ROOT/'subtitle/work')
                if source is None:pending.append(row)
                else:export(row,source)
            if pending:
                # Reuse the EXISTING download/GPU/bounded-queue runner.
                with (work/'source_runner.log').open('a',encoding='utf-8') as log,redirect_stdout(log):
                    pipeline.run(extraction_plan,extract_only=True,
                        only_fids=[str(r['subtitle']['fid']) for r in pending],on_extracted=export)
        finally:
            failed=[str(r['video']['fid']) for r in selected if str(r['video']['fid']) not in prepared]
            for row in selected:
                fid=str(row['video']['fid'])
                if fid in failed:
                    target=work/'translations'/fid;target.mkdir(parents=True,exist_ok=True)
                    save(target/'blocked.json',dict(video_fid=fid,status='failed_source_preparation',
                        reason='Finite source preparation failed; no automatic full-film retry',
                        evidence=str(work/'extraction/state.json')))
            save(work/'source_run.json',dict(active=False,finished_at=time.time(),
                 prepared_video_fids=prepared,failed_video_fids=failed))
        if failed and not prepared:raise RuntimeError('Source preparation failed; see source_runner.log')
        return dict(prepared_video_fids=prepared,failed_video_fids=failed,buffer_slots=slots)


def ready(work):
    work=Path(work);items=[];errors=[]
    for row in read(work/'plan.json')['items']:
        target=work/'translations'/str(row['video']['fid'])
        if (target/'blocked.json').exists() or (target/'content_hold.json').exists():continue
        if not (target/'job.json').exists() or not (target/'media_identity.json').exists():continue
        if (target/'api_translation/delivery/manifest.json').exists():continue
        from subtitle_version_reuse import decision,remember_wait
        version=decision(work,row)
        if version['action']!='produce':
            if version['action']!='complete':remember_wait(work,row['video']['fid'],version)
            continue
        try:state=native.checked(target)
        except (ValueError, OSError, KeyError) as exc:
            errors.append(dict(video_fid=str(row['video']['fid']), reason=str(exc)))
            continue
        packets=[str(Path(p)/'input.json') for p in state['packets'] if not (Path(p)/'receipt.json').exists()]
        if packets:items.append(dict(video_fid=str(row['video']['fid']),rules=str(target/'rules.txt'),inputs=packets,output_names=['output.json'],model=native.MODEL,effort=native.EFFORT))
    return dict(translation_tasks=items,input_errors=errors)


def retain_old(cloud,meta,work,journal,persist):
    fid=str(meta['fid']);folder=work/'originals'/fid;folder.mkdir(parents=True,exist_ok=True)
    rows=cloud.list_dir(meta['cid']);current=next((r for r in rows if str(r.get('fid'))==fid),None)
    if not current or current.get('sha')!=meta.get('sha'):raise ValueError('Selected old subtitle changed')
    if not (folder/meta['n']).exists():cloud.download_verified(dict(meta,pc=current['pc']),folder/meta['n'])
    backup=folder/meta['n']
    if backup.stat().st_size!=int(meta['s']) or hashlib.sha1(backup.read_bytes()).hexdigest().upper()!=meta['sha'].upper():
        raise ValueError('Old subtitle backup size/hash mismatch; do not rename')
    if Path(current['n']).stem.endswith('旧'):
        journal[fid]=dict(name=current['n'],sha1=current['sha']);persist();return
    name=Path(meta['n']).stem+'旧.srt'
    if any(r['n']==name and str(r.get('fid'))!=fid for r in rows):name=Path(meta['n']).stem+f'({fid})旧.srt'
    if current['n'] not in {meta['n'],name}:raise ValueError('Selected old subtitle was unexpectedly renamed')
    if current['n']!=name and not cloud.rename(fid,name):raise RuntimeError('Old subtitle rename not confirmed')
    actual=next((r for r in cloud.list_dir(meta['cid']) if str(r.get('fid'))==fid),None)
    if not actual or actual['n']!=name or actual.get('sha')!=meta['sha']:raise ValueError('Old subtitle retention verification failed')
    journal[fid]=dict(name=name,sha1=meta['sha']);persist()


def publish(work,fid,allow_deferred=False):
    from cloud_common import Cloud115
    from nosub_pipeline import load_env
    work=Path(work).resolve();row=next(r for r in read(work/'plan.json')['items'] if str(r['video']['fid'])==fid)
    target=work/'translations'/fid;target.mkdir(parents=True,exist_ok=True)
    if (target/'blocked.json').exists():raise ValueError('Job isolated; see blocked.json')
    manifest=build_for_delivery(work,fid)
    allowed={'ready','ready_with_deferred'} if allow_deferred else {'ready'}
    if manifest['status'] not in allowed:
        raise ValueError('Important dialogue unresolved; no automatic re-review or publication')
    cloud=Cloud115(load_env()['P115_COOKIES']);actual=next((r for r in cloud.list_dir(row['cid']) if str(r.get('fid'))==fid),None)
    if not actual or actual['n']!=row['video']['n'] or actual.get('sha')!=row['video']['sha']:raise ValueError('Frozen video changed')
    journal_path=target/'publication.json';journal=read(journal_path) if journal_path.exists() else {}
    if row.get('publication_reuse'):
        prior=row['publication_reuse'];current=next((r for r in cloud.list_dir(row['cid']) if str(r.get('fid'))==str(prior['fid'])),None)
        if not current or current['n']!=prior['n'] or current.get('sha')!=prior['sha']:raise ValueError('Verified existing publication changed')
        journal.update(stage='complete',stage_fid=str(prior['fid']),sha1=prior['sha'],verified_publication_reused=True)
        save(journal_path,journal)
    else:
        output=Path(manifest['output']['path']);original=row['subtitle']
        if original.get('placeholder'):cloud.publish_missing_subtitle(row['video'],output,journal,lambda:save(journal_path,journal))
        else:cloud.replace_subtitle(original,output,target/'rollback'/original['n'],journal,lambda:save(journal_path,journal),target_name=Path(row['video']['n']).stem+'.srt')
    if journal.get('stage')!='complete':raise ValueError('Canonical publication incomplete')
    aliases=journal.setdefault('selected_old_subtitles',{})
    for meta in row['targets']:
        if str(meta['fid'])==str(journal['stage_fid']):raise ValueError('New output cannot be retired as an old source')
        retain_old(cloud,meta,work,aliases,lambda:save(journal_path,journal))
    journal['batch_targets_complete']=True;save(journal_path,journal)
    return dict(video_fid=fid,stage='complete',old_subtitles_retained=len(aliases),cloud_fid=journal['stage_fid'])


def build_for_delivery(work,fid):
    work=Path(work).resolve()
    path=work/'delivery_policy.json'
    policy=read(path) if path.exists() else {}
    target=work/'translations'/fid
    manifest_path=target/'delivery/manifest.json'
    api_manifest=target/'api_translation/delivery/manifest.json'
    try:
        if not manifest_path.exists():native.build(target)
        manifest=native.build(target,max_deferred=policy.get('max_deferred_cues',0),persist_manifest=False)
    except Exception:
        # Films translated by the budgeted API have no native packets.
        if not api_manifest.exists():raise
        manifest=read(api_manifest)
        manifest_path=api_manifest
    content_review=target/'content_review/review.json'
    if content_review.exists():
        from subtitle_content_review import apply_review
        manifest=apply_review(target,manifest,max_deferred=policy.get('max_deferred_cues',0))
    repair=target/'supplement/repair.json'
    if repair.exists():
        from subtitle_supplement import apply_repair
        manifest=apply_repair(target,manifest,max_deferred=policy.get('max_deferred_cues',0))
    cue_repair=target/'cue_repair/repair.json'
    if cue_repair.exists():
        from subtitle_cue_repair import apply_repair
        manifest=apply_repair(target,manifest,max_deferred=policy.get('max_deferred_cues',0))
    isolated=target/'isolated/repair.json'
    if isolated.exists():
        from subtitle_isolated_translation import apply_repair
        manifest=apply_repair(target,manifest,max_deferred=policy.get('max_deferred_cues',0))
    acoustic=target/'acoustic_review/repair.json'
    if acoustic.exists():
        from subtitle_acoustic_translation import apply_repair
        manifest=apply_repair(target,manifest,max_deferred=policy.get('max_deferred_cues',0))
    source_coverage=None
    timing=Path(manifest['source']['path']).with_suffix('.alignment.json')
    if timing.exists() and read(timing).get('source_subset_review_sha1'):
        from source_subset import verify
        subset=verify(timing.parent,read(timing))
        source_coverage=dict(full_source_complete=False,quality_scope='retained_cues_only',
            deferred_source_ids=[e['id'] for e in subset['deferred']],
            review=native.proof(timing.parent/'source_subset_review.json'))
    save(target/'delivery/acceptance.json',dict(
        translation_manifest=native.proof(manifest_path),
        delivery_policy=native.proof(path) if path.exists() else None,
        status=manifest['status'],counts=manifest['counts'],output=manifest['output'],
        supplement=native.proof(repair) if repair.exists() else None,
        content_review=native.proof(content_review) if content_review.exists() else None,
        cue_repair=native.proof(cue_repair) if cue_repair.exists() else None,
        isolated=native.proof(isolated) if isolated.exists() else None,
        acoustic=native.proof(acoustic) if acoustic.exists() else None,
        max_deferred=manifest['max_deferred'],deferred_ids=manifest['deferred_ids'],source_coverage=source_coverage))
    return manifest


@contextmanager
def completion_lock(work):
    """Reuse the existing crash-safe lock for cross-process local reports."""
    from reprocess_subtitles import runner_lock,RunnerBusy
    folder=Path(work)/'reporting';folder.mkdir(exist_ok=True)
    for attempt in range(200):
        lock=runner_lock(folder/'plan.json')
        try:lock.__enter__();break
        except RunnerBusy:
            if attempt==199:raise
            time.sleep(.05)
    try:yield
    finally:lock.__exit__(None,None,None)


def local_delivery(work,fid):
    with DELIVERY_LOCK,completion_lock(work):
        return _local_delivery(work,fid)


def _local_delivery(work,fid):
    """Collect verified subtitles and cloud destinations for manual upload."""
    work=Path(work).resolve()
    row=next(r for r in read(work/'plan.json')['items'] if str(r['video']['fid'])==fid)
    target=work/'translations'/fid
    identity=read(target/'media_identity.json')
    native.check_proofs([identity['source'],identity['plan']])
    manifest=build_for_delivery(work,fid)
    if manifest['status'] not in {'ready','ready_with_deferred'}:
        raise ValueError('Incomplete subtitle; no manual upload delivery')
    source=Path(manifest['output']['path'])
    destination=work/'manual_upload'/fid/(Path(row['video']['n']).stem+'.srt')
    destination.parent.mkdir(parents=True,exist_ok=True)
    index_path=work/'manual_upload/index.json'
    index=read(index_path) if index_path.exists() else dict(mode='manual_upload',jobs={})
    replace=False
    if destination.exists() and destination.read_bytes()!=source.read_bytes():
        prior=index['jobs'].get(fid);old=destination.read_bytes()
        if (not prior or Path(prior['path']).resolve()!=destination.resolve() or
            len(old)!=prior['size'] or hashlib.sha1(old).hexdigest().upper()!=prior['sha1']):
            raise ValueError('Existing manual delivery changed; do not overwrite')
        replace=True
    if not destination.exists() or replace:
        part=destination.with_name(destination.name+'.part')
        shutil.copyfile(source,part);part.replace(destination)
    content=destination.read_bytes()
    if content!=source.read_bytes():raise ValueError('Manual delivery byte verification failed')
    result=dict(video_fid=fid,status='ready_for_manual_upload',path=str(destination),
        cloud_cid=str(row['cid']),name=destination.name,size=len(content),
        sha1=hashlib.sha1(content).hexdigest().upper(),counts=manifest['counts'],
        acceptance=manifest['status'],deferred_ids=manifest.get('deferred_ids',[]),
        old_subtitles=[dict(fid=str(m['fid']),name=m['n'],size=m['s'],sha1=m['sha'],
            suggested_retained_name=Path(m['n']).stem+'旧.srt') for m in row['targets']])
    index['jobs'][fid]=result;save(index_path,index)
    return result


def report_completion(work,fid,manifest,delivery=None):
    """Completion-triggered report; no scheduler, polling or model calls."""
    work=Path(work).resolve()
    with DELIVERY_LOCK,completion_lock(work):
        event=dict(video_fid=fid,status=manifest['status'],counts=manifest['counts'],
                   output=manifest['output'],deferred_ids=manifest.get('deferred_ids',[]),
                   local_path=delivery['path'] if delivery else None)
        signature=hashlib.sha256(json.dumps(event,sort_keys=True).encode()).hexdigest()
        state_path=work/'completion_reports.json'
        reports=read(state_path) if state_path.exists() else {}
        if reports.get(fid,{}).get('signature')==signature:
            refresh_summary(work)
            return event
        reports[fid]=dict(event,signature=signature)
        save(state_path,reports)
        with (work/'completion_events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(event,event='subtitle_completed'),ensure_ascii=False)+'\n')
        print(json.dumps(dict(event,event='subtitle_completed'),ensure_ascii=True),flush=True)
        lines=['# 字幕完成报告','','每部完成后由脚本更新；可观看待补不等于无缺项。','','| 视频FID | 验收状态 | 待补ID | 本地字幕 |','|---|---|---|---|']
        for item in reports.values():
            output=item.get('local_path') or item['output']['path']
            lines.append(f"| {item['video_fid']} | {item['status']} | {', '.join(item.get('deferred_ids',[]))} | [{Path(output).name}](<{Path(output).as_posix()}>) |")
        (work/'完成报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        refresh_summary(work)
        return event


def refresh_restricted_index(work):
    """Index existing restricted records without copying their dialogue."""
    work=Path(work).resolve();rows={str(r['video']['fid']):r for r in read(work/'plan.json')['items']}
    jobs={}
    for path in (work/'translations').glob('*/delivery/manifest.json'):
        manifest=read(path)
        target=path.parent.parent
        if (target/'content_review/review.json').exists() and not (target/'content_hold.json').exists():
            from subtitle_content_review import apply_review
            manifest=apply_review(target,manifest,max_deferred=manifest['max_deferred'])
            path=target/'content_review/delivery/manifest.json'
        if not manifest['counts'].get('content_isolated'):continue
        fid=target.name
        identity=read(target/'media_identity.json') if (target/'media_identity.json').exists() else {}
        jobs[fid]=dict(video_fid=fid,name=rows[fid]['video']['n'],status='content_isolated',
            manifest=native.proof(path),source=identity.get('source'),
            isolated_cues=[dict(id=r['source']['id'],start=r['source'].get('start'),end=r['source'].get('end'))
                           for r in manifest['ledger'] if r['status']=='content_isolated'])
    for target in (work/'translations').iterdir():
        if not target.is_dir():continue
        hold=target/'content_hold.json';blocked=target/'blocked.json'
        evidence=hold if hold.exists() else (blocked if blocked.exists() and read(blocked).get('status')=='content_hold' else None)
        if evidence is None:continue
        fid=target.name;identity=read(target/'media_identity.json') if (target/'media_identity.json').exists() else {}
        jobs[fid]=dict(video_fid=fid,name=rows.get(fid,{}).get('video',{}).get('n',fid),status='content_hold',
            hold=native.proof(evidence),source=identity.get('source'),job_path=str(target/'job.json'),
            isolated_cues=jobs.get(fid,{}).get('isolated_cues',[]))
    folder=work/'restricted_review';folder.mkdir(exist_ok=True)
    index=dict(mode='references_only_no_dialogue_copies',jobs=jobs,
               content_hold_jobs=sum(r['status']=='content_hold' for r in jobs.values()),
               isolated_cue_count=sum(len(r['isolated_cues']) for r in jobs.values()))
    index_path=folder/'index.json'
    if not index_path.exists() or read(index_path)!=index:save(index_path,index)
    lines=['# 隔离记录定位清单','','原文、草稿及回执保留在原位置，本清单只记录定位与校验值，不复制受限对白。',
           '现有分类未重新做语义审校；明确受限条目不自动转其他模型。','','| 视频 | 状态 | 隔离ID | 原记录 |','|---|---|---|---|']
    for row in jobs.values():
        proof=row.get('hold') or row['manifest'];ids=', '.join(r['id'] for r in row['isolated_cues'])
        lines.append(f"| {row['name']} | {row['status']} | {ids} | [记录](<{Path(proof['path']).as_posix()}>) |")
    readme=folder/'README.md';body='\n'.join(lines)+'\n'
    if not readme.exists() or readme.read_text(encoding='utf-8')!=body:
        readme.write_text(body,encoding='utf-8')
    return dict(jobs=len(jobs),content_hold_jobs=index['content_hold_jobs'],isolated_cues=index['isolated_cue_count'])


def refresh_summary(work):
    work=Path(work);summary=read(work/'summary.json');manifests=[]
    translation_total=sum(not row.get('publication_reuse') for row in read(work/'plan.json')['items'])
    for path in (work/'translations').glob('*/delivery/manifest.json'):
        manifest=read(path);acceptance=path.with_name('acceptance.json')
        if acceptance.exists():
            decision=read(acceptance);native.check_proofs([decision['translation_manifest']])
            if decision.get('supplement'):native.check_proofs([decision['supplement']])
            if decision.get('content_review'):
                from subtitle_content_review import apply_review
                apply_review(path.parent.parent,manifest,max_deferred=decision['max_deferred'])
            manifest.update(status=decision['status'],counts=decision['counts'])
        manifests.append((path,manifest))
    blocked=[read(p) for p in (work/'translations').glob('*/blocked.json')]
    holds=[r for r in blocked if r['status']=='content_hold']
    failures=[r for r in blocked if r['status']=='failed_source_preparation']
    execution_failures=[r for r in blocked if r['status'] in {'translation_execution_failed','source_validation_failed'}]
    decisions=[dict(video_fid=p.parent.parent.name,status=m['status'],counts=m['counts'],manifest=str(p))
               for p,m in manifests if m['status'] not in {'ready','ready_with_deferred'}]
    prepared=len(list((work/'translations').glob('*/media_identity.json')))
    index=read(work/'manual_upload/index.json') if (work/'manual_upload/index.json').exists() else {'jobs':{}}
    if index['jobs']:
        lines=['# 本地待上传字幕','','文件名与视频一致。“可观看待补”保留少量缺项，待补ID如下；原翻译回执不变。',
               '','先将对应旧字幕加“旧”保留，再就地上传同名新版；不要改动视频。',
               '','| 新字幕 | 字节数 | 待补ID | 旧版保留名称 |','|---|---:|---|---|']
        for item in sorted(index['jobs'].values(),key=lambda r:r['name']):
            old='、'.join(m['suggested_retained_name'] for m in item['old_subtitles'])
            lines.append(f"| [{item['name']}](<{item['video_fid']}/{item['name']}>) | {item['size']} | {', '.join(item.get('deferred_ids',[]))} | {old} |")
        (work/'manual_upload/上传清单.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    placement=work/'cloud_placement/summary.json'
    published=sum(read(placement).get('verified_placements',{}).values()) if placement.exists() else 0
    summary['execution']=dict(prepared_jobs=prepared,
        remaining_source_jobs=translation_total-prepared-len(failures),
        translated_jobs=len(manifests),manual_ready_jobs=len(index['jobs']),new_publications=published,
        watchable_pending_jobs=sum(m['status']=='ready_with_deferred' for _,m in manifests),
        needs_source_decision=sum(r['status']=='needs_source_decision' for r in decisions),
        needs_content_decision=sum(r['status']=='needs_content_decision' for r in decisions),
        content_hold_jobs=len(holds),failed_source_jobs=len(failures),
        execution_failed_jobs=len(execution_failures),blockers=decisions+holds+failures+execution_failures)
    summary['controller_polling']=False
    save(work/'summary.json',summary)
    return summary['execution']


def finish_local(work,fid):
    manifest=build_for_delivery(work,fid)
    delivery=local_delivery(work,fid) if manifest['status'] in {'ready','ready_with_deferred'} else None
    report_completion(work,fid,manifest,delivery)
    return dict(status=manifest['status'],counts=manifest['counts'],
                local_path=delivery['path'] if delivery else None,
                deferred_ids=manifest.get('deferred_ids',[]))


def record_dispatch(work,fid,agent):
    target=Path(work).resolve()/'translations'/fid
    native.checked(target)
    if not agent:raise ValueError('Actual native Agent task required')
    record=dict(video_fid=fid,agent=agent,model=native.MODEL,effort=native.EFFORT,status='running')
    path=target/'dispatch.json'
    if path.exists():
        previous=read(path)
        if previous['status']=='running' and previous['agent']!=agent:
            raise ValueError('Another native Agent is already assigned')
    save(path,record)
    return record


def accept_translation(work,fid,agent):
    work=Path(work);target=work/'translations'/fid;state=native.checked(target)
    for name in state['packets']:
        folder=Path(name)
        if not (folder/'receipt.json').exists():native.accept(target,folder.name,agent)
    policy_path=work/'delivery_policy.json'
    if policy_path.exists() and read(policy_path).get('mode')=='local_only':
        result=finish_local(work,fid)
    else:
        manifest=native.build(target);result=dict(status=manifest['status'],counts=manifest['counts'])
    dispatch=target/'dispatch.json'
    if dispatch.exists():
        record=read(dispatch)
        if record.get('agent')==agent:
            record['status']='accepted';save(dispatch,record)
    return result


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('plan');a.add_argument('--work',type=Path,required=True);a.add_argument('--inventory',type=Path);a.add_argument('--from-date',required=True);a.add_argument('--before-date',required=True)
    a=sub.add_parser('sources');a.add_argument('--work',type=Path,required=True);a.add_argument('--limit',type=int,default=3)
    for name in ('ready','status'):
        a=sub.add_parser(name);a.add_argument('--work',type=Path,required=True)
    a=sub.add_parser('accept');a.add_argument('--work',type=Path,required=True);a.add_argument('--video-fid',required=True);a.add_argument('--agent',required=True)
    a=sub.add_parser('dispatched');a.add_argument('--work',type=Path,required=True);a.add_argument('--video-fid',required=True);a.add_argument('--agent',required=True)
    a=sub.add_parser('publish');a.add_argument('--work',type=Path,required=True);a.add_argument('--video-fid',required=True);a.add_argument('--allow-deferred',action='store_true')
    a=sub.add_parser('local');a.add_argument('--work',type=Path,required=True);a.add_argument('--video-fid',required=True)
    a=p.parse_args()
    if a.action=='plan':
        inventory=read(a.inventory or scan(a.work));done,held=completed_sources(inventory,ROOT/'subtitle/work')
        result=plan_batch(inventory,a.work,a.from_date,a.before_date,done,held)
    elif a.action=='sources':
        if a.limit<1:raise ValueError('Positive source limit required')
        result=prepare_sources(a.work,a.limit)
    elif a.action=='ready':result=ready(a.work)
    elif a.action=='accept':result=accept_translation(a.work,a.video_fid,a.agent)
    elif a.action=='dispatched':result=record_dispatch(a.work,a.video_fid,a.agent)
    elif a.action=='publish':result=publish(a.work,a.video_fid,allow_deferred=a.allow_deferred)
    elif a.action=='local':
        result=local_delivery(a.work,a.video_fid)
        report_completion(a.work,a.video_fid,build_for_delivery(a.work,a.video_fid),result)
    else:
        plan=read(a.work/'plan.json');result=dict(raw_srt_count=len(plan['ledger']),video_jobs=len(plan['items']),prepared_jobs=sum((a.work/'translations'/str(r['video']['fid'])/'job.json').exists() for r in plan['items']),published_jobs=sum((a.work/'translations'/str(r['video']['fid'])/'publication.json').exists() and read(a.work/'translations'/str(r['video']['fid'])/'publication.json').get('batch_targets_complete') is True for r in plan['items']))
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
