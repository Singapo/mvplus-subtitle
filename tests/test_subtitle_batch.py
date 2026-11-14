from datetime import datetime,timezone,timedelta
import json
import hashlib
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import subtitle_batch as batch

def stamp(date):return str(int(datetime.fromisoformat(date).replace(tzinfo=timezone(timedelta(hours=8))).timestamp()))
def media(fid,name,date='2026-09-01'):
    return dict(fid=fid,cid='1',n=name,tp=stamp(date),sha='HASH'+fid,pc='P'+fid,s=100)
def inventory(files,extra=None):
    folders={'1':dict(path='/films',files=files)}
    folders.update(extra or {})
    return dict(directories=folders)

def test_scope_is_srt_upload_not_video_or_modification_date(tmp_path):
    video=media('v','film.mp4','2025-01-01');selected=media('s','film.srt');selected['te']=stamp('2027-01-01')
    outside=media('o','outside.srt','2026-10-15')
    result=batch.plan_batch(inventory([video,selected,outside]),tmp_path,'2026-08-01','2026-10-15')
    assert result['raw_srt_count']==1 and result['video_jobs']==1

def test_chinese_presence_never_means_processed_and_aliases_share_one_job(tmp_path):
    rows=[media('v','film.mp4'),media('s','film.srt'),media('a','film.ja.srt')]
    result=batch.plan_batch(inventory(rows),tmp_path,'2026-08-01','2026-10-15')
    assert result['raw_srt_count']==2 and result['video_jobs']==1
    plan=batch.read(tmp_path/'plan.json');assert len(plan['items'][0]['targets'])==2

def test_archive_is_in_ledger_and_unique_global_video_can_match(tmp_path):
    result=batch.plan_batch(inventory([media('v','film.mp4')],{'2':dict(path='/原文',files=[dict(media('s','film.ja.srt'),cid='2')])}),tmp_path,'2026-08-01','2026-10-15')
    assert result['raw_srt_count']==1 and result['video_jobs']==1

def test_ambiguous_video_is_not_guessed(tmp_path):
    result=batch.plan_batch(inventory([media('s','film.srt')],{'2':dict(path='/a',files=[media('v','film.mp4')]),'3':dict(path='/b',files=[media('w','film.mkv')])}),tmp_path,'2026-08-01','2026-10-15')
    assert result['video_jobs']==0 and result['statuses']=={'ambiguous_video':1}

def test_verified_replacement_is_skipped_but_name_alone_is_not(tmp_path):
    s=media('s','film旧.srt');v=media('v','film.mp4')
    result=batch.plan_batch(inventory([s,v]),tmp_path/'a','2026-08-01','2026-10-15')
    assert result['video_jobs']==1
    result=batch.plan_batch(inventory([s,v]),tmp_path/'b','2026-08-01','2026-10-15',{('s','HASHs'.upper())})
    assert result['statuses']=={'verified_completed':1}

def test_completed_video_reuse_needs_no_new_translation(tmp_path):
    v=media('v','film.mp4');s=media('s','film.srt','2026-10-20');a=media('a','film.ja.srt')
    batch.plan_batch(inventory([v,s,a]),tmp_path,'2026-08-01','2026-10-15',{('s','HASHs'.upper())})
    plan=batch.read(tmp_path/'plan.json');assert plan['items'][0]['publication_reuse']['fid']=='s'

def test_sources_invokes_only_scripted_extract_only_then_exports_native_packets(tmp_path,monkeypatch):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt')]),tmp_path,'2026-08-01','2026-10-15')
    monkeypatch.setattr(batch,'cache_source',lambda *a:None)
    monkeypatch.setattr(batch,'prior_seed',lambda *a:None)
    calls=[]
    def runner(plan,**kwargs):
        calls.append(kwargs)
        source=tmp_path/'extraction/jobs/s/film.srt';source.parent.mkdir(parents=True,exist_ok=True)
        source.write_text('1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n',encoding='utf-8')
        kwargs['on_extracted'](batch.read(plan)['items'][0],source)
        return {}
    import reprocess_subtitles
    monkeypatch.setattr(reprocess_subtitles,'run',runner)
    monkeypatch.setattr(reprocess_subtitles,'require_current_timing',lambda *a:{'quality_passed':True})
    result=batch.prepare_sources(tmp_path)
    assert result['prepared_video_fids']==['v']
    assert len(calls)==1 and calls[0]['extract_only'] is True and calls[0]['only_fids']==['s']
    tasks=batch.ready(tmp_path)['translation_tasks'];assert len(tasks)==1
    assert set(tasks[0])=={'video_fid','rules','inputs','output_names','model','effort'}
    assert batch.read(tmp_path/'source_run.json')['active'] is False
    assert json.loads((tmp_path/'source_events.jsonl').read_text('utf-8'))['video_fid']=='v'

def test_source_failure_never_dispatches_translation(tmp_path,monkeypatch):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt')]),tmp_path,'2026-08-01','2026-10-15')
    monkeypatch.setattr(batch,'cache_source',lambda *a:None)
    import reprocess_subtitles
    monkeypatch.setattr(reprocess_subtitles,'run',lambda *a,**k:None)
    with pytest.raises(RuntimeError):batch.prepare_sources(tmp_path)
    assert batch.ready(tmp_path)['translation_tasks']==[]
    assert batch.read(tmp_path/'source_run.json')['active'] is False
    assert batch.read(tmp_path/'translations/v/blocked.json')['status']=='failed_source_preparation'


def test_callback_failure_closes_source_event_state(tmp_path,monkeypatch):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt')]),tmp_path,'2026-08-01','2026-10-15')
    monkeypatch.setattr(batch,'cache_source',lambda *a:tmp_path/'cached.srt')
    monkeypatch.setattr(batch,'prior_seed',lambda *a:None)
    monkeypatch.setattr(batch.native,'prepare',lambda *a,**kw:(_ for _ in ()).throw(ValueError('bad source')))
    with pytest.raises(ValueError,match='bad source'):batch.prepare_sources(tmp_path)
    assert batch.read(tmp_path/'source_run.json')['active'] is False
    assert batch.read(tmp_path/'source_run.json')['failed_video_fids']==['v']


def test_resume_accepts_only_missing_packets_and_preserves_original_worker_receipt(tmp_path):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt')]),tmp_path,'2026-08-01','2026-10-15')
    def stamp_ms(second):return f'00:{second//60:02}:{second%60:02},000'
    source=tmp_path/'film.srt'
    source.write_text('\n\n'.join(f'{i+1}\n{stamp_ms(i)} --> {stamp_ms(i+1)}\n原文' for i in range(321))+'\n',encoding='utf-8')
    target=tmp_path/'translations/v';state=batch.native.prepare(source,target)
    for folder in map(Path,state['packets']):
        ids=batch.read(folder/'control.json')['ids']
        batch.save(folder/'output.json',[[idx,'对白','ok','',''] for idx in ids])
    batch.native.accept(target,'001','/root/original_worker')
    original=(target/'packets/001/receipt.json').read_bytes()
    batch.record_dispatch(tmp_path,'v','/root/resume_worker')
    assert batch.accept_translation(tmp_path,'v','/root/resume_worker')['status']=='ready'
    assert (target/'packets/001/receipt.json').read_bytes()==original
    assert batch.read(target/'packets/002/receipt.json')['agent']=='/root/resume_worker'
    assert batch.read(target/'dispatch.json')['status']=='accepted'


def test_failed_source_is_not_retried_and_next_video_continues(tmp_path,monkeypatch):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt'),
        media('w','other.mp4'),media('t','other.srt')]),tmp_path,'2026-08-01','2026-10-15')
    target=tmp_path/'translations/v';target.mkdir(parents=True)
    batch.save(target/'blocked.json',dict(status='failed_source_preparation'))
    source=tmp_path/'cached.srt'
    source.write_text('1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n',encoding='utf-8')
    monkeypatch.setattr(batch,'cache_source',lambda *a:source)
    monkeypatch.setattr(batch,'prior_seed',lambda *a:None)
    assert batch.prepare_sources(tmp_path)['prepared_video_fids']==['w']
    assert [r['video_fid'] for r in batch.ready(tmp_path)['translation_tasks']]==['w']


def test_isolated_translation_releases_buffer_and_cannot_publish(tmp_path,monkeypatch):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt')]),tmp_path,'2026-08-01','2026-10-15')
    target=tmp_path/'translations/v';target.mkdir(parents=True)
    batch.save(target/'blocked.json',dict(status='content_isolated'))
    batch.save(target/'job.json',{})
    batch.save(target/'media_identity.json',{})
    monkeypatch.setattr(batch.native,'checked',lambda *a:pytest.fail('Isolated input must not be read'))
    assert batch.ready(tmp_path)['translation_tasks']==[]
    with pytest.raises(ValueError,match='isolated'):
        batch.publish(tmp_path,'v')


def prepare_local_fixture(tmp_path,status='ok'):
    batch.plan_batch(inventory([media('v','film.mp4'),media('s','film.srt')]),tmp_path,'2026-08-01','2026-10-15')
    source=tmp_path/'source.srt'
    source.write_text('1\n00:00:01,000 --> 00:00:02,000\n前半\n\n2\n00:00:02,000 --> 00:00:03,000\n後半\n',encoding='utf-8')
    target=tmp_path/'translations/v';batch.native.prepare(source,target)
    batch.save(target/'media_identity.json',dict(source=batch.native.proof(source),plan=batch.native.proof(tmp_path/'plan.json')))
    batch.save(target/'packets/001/output.json',[
        ['1','可靠对白','ok','',''],['2','后半' if status=='ok' else '',status,'','' if status=='ok' else '未决']])
    batch.native.accept(target,'001','/root/local_test')


def test_local_delivery_preserves_name_bytes_and_never_calls_cloud(tmp_path,monkeypatch):
    prepare_local_fixture(tmp_path)
    import cloud_common
    monkeypatch.setattr(cloud_common,'Cloud115',lambda *a:pytest.fail('Local delivery must never access cloud'))
    result=batch.local_delivery(tmp_path,'v')
    path=Path(result['path'])
    assert path.name=='film.srt'
    assert result['sha1']==hashlib.sha1(path.read_bytes()).hexdigest().upper()
    assert result['old_subtitles'][0]['suggested_retained_name']=='film旧.srt'
    assert batch.local_delivery(tmp_path,'v')==result
    path.write_text('user edits',encoding='utf-8')
    with pytest.raises(ValueError,match='overwrite'):
        batch.local_delivery(tmp_path,'v')


@pytest.mark.parametrize('status',['unresolved'])
def test_local_delivery_rejects_incomplete_subtitles(tmp_path,status):
    prepare_local_fixture(tmp_path,status)
    with pytest.raises(ValueError,match='Incomplete'):
        batch.local_delivery(tmp_path,'v')
    assert not (tmp_path/'manual_upload/index.json').exists()


def test_isolated_cues_no_longer_block_local_delivery(tmp_path):
    prepare_local_fixture(tmp_path,'content_isolated')
    result=batch.local_delivery(tmp_path,'v')
    assert Path(result['path']).exists()


def test_user_allowance_delivers_small_pending_and_reports_once(tmp_path,monkeypatch):
    prepare_local_fixture(tmp_path,'unresolved')
    batch.native.build(tmp_path/'translations/v')
    original_manifest=(tmp_path/'translations/v/delivery/manifest.json').read_bytes()
    batch.save(tmp_path/'delivery_policy.json',dict(mode='local_only',max_deferred_cues=2))
    import cloud_common
    monkeypatch.setattr(cloud_common,'Cloud115',lambda *a:pytest.fail('No cloud access'))
    receipt=(tmp_path/'translations/v/packets/001/receipt.json').read_bytes()
    result=batch.finish_local(tmp_path,'v')
    assert result['status']=='ready_with_deferred'
    assert result['deferred_ids']==['2'] and Path(result['local_path']).exists()
    assert batch.read(tmp_path/'manual_upload/index.json')['jobs']['v']['acceptance']=='ready_with_deferred'
    batch.finish_local(tmp_path,'v')
    assert len((tmp_path/'completion_events.jsonl').read_text('utf-8').splitlines())==1
    assert (tmp_path/'translations/v/packets/001/receipt.json').read_bytes()==receipt
    assert (tmp_path/'translations/v/delivery/manifest.json').read_bytes()==original_manifest

def test_frozen_batch_cannot_silently_change(tmp_path):
    first=inventory([media('v','film.mp4'),media('s','film.srt')]);batch.plan_batch(first,tmp_path,'2026-08-01','2026-10-15')
    with pytest.raises(ValueError):batch.plan_batch(inventory([media('v','film.mp4'),media('x','film.srt')]),tmp_path,'2026-08-01','2026-10-15')


def test_retaining_old_subtitle_is_verified_and_idempotent(tmp_path):
    data=b'original subtitle';meta=dict(media('s','film.srt'),s=len(data),sha=hashlib.sha1(data).hexdigest().upper())
    class Cloud:
        def __init__(self):self.row=dict(meta);self.renames=0
        def list_dir(self,cid):return [dict(self.row)]
        def download_verified(self,row,path):path.write_bytes(data)
        def rename(self,fid,name):self.row['n']=name;self.renames+=1;return True
    cloud=Cloud();journal={}
    batch.retain_old(cloud,meta,tmp_path,journal,lambda:None)
    batch.retain_old(cloud,meta,tmp_path,journal,lambda:None)
    assert cloud.renames==1 and cloud.row['n']=='film旧.srt'
    assert journal['s']['sha1']==meta['sha']


def test_bad_backup_never_renames_old_cloud_file(tmp_path):
    data=b'original';meta=dict(media('s','film.srt'),s=len(data),sha=hashlib.sha1(data).hexdigest().upper())
    folder=tmp_path/'originals/s';folder.mkdir(parents=True);(folder/'film.srt').write_bytes(b'corrupt')
    class Cloud:
        def list_dir(self,cid):return [dict(meta)]
        def rename(self,*args):raise AssertionError('Must not rename before backup verification')
    with pytest.raises(ValueError,match='backup size/hash mismatch'):
        batch.retain_old(Cloud(),meta,tmp_path,{},lambda:None)


def test_existing_partial_and_complete_native_results_are_reused(tmp_path):
    source=tmp_path/'source.srt';source.write_text('1\n00:00:01,000 --> 00:00:02,000\n原文\n',encoding='utf-8')
    old=tmp_path/'history/old';batch.native.prepare(source,old)
    batch.native.save(old/'packets/001/output.json',[['1','可靠含义','partial','不清词','保留已确认部分']])
    batch.native.accept(old,'001','/root/actual_previous_worker');batch.native.build(old)
    target=tmp_path/'next';seed=batch.prior_seed(source,tmp_path/'history',target)
    state=batch.native.prepare(source,target,seed=seed)
    assert state['packets']==[] and state['seed'][0][2]=='partial'
