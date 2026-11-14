import hashlib
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import native_subtitles as native
import subtitle_batch as batch
import subtitle_supplement as supplement


def test_each_paid_packet_only_receives_its_own_neighbours():
    payload=dict(rows=[[str(i),'source'] for i in range(5)],context=[['distant','not needed']],
        context_by_id={str(i):[[f'n{i}',f'neighbour {i}']] for i in range(5)})
    requests=list(supplement.plan_requests(payload))
    assert len(requests)==2
    first=json.loads(requests[0][1].split('\n')[-1])
    last=json.loads(requests[1][1].split('\n')[-1])
    assert [r[0] for r in first['context']]==['n0','n1','n2','n3']
    assert last['context']==[['n4','neighbour 4']]


@pytest.fixture
def work(tmp_path):
    def meta(fid,name):return dict(fid=fid,cid='1',n=name,tp='1788278400',s=1,sha='hash'+fid,pc='p')
    batch.plan_batch(dict(directories={'1':dict(path='/films',files=[meta('v','film.mp4'),meta('s','film.srt')])}),
                     tmp_path,'2026-08-01','2026-10-15')
    source=tmp_path/'film.srt'
    source.write_text('1\n00:00:01,000 --> 00:00:02,000\n前半\n\n2\n00:00:02,000 --> 00:00:03,000\n中間\n\n3\n00:00:03,000 --> 00:00:04,000\n後半\n',encoding='utf-8')
    target=tmp_path/'translations/v';native.prepare(source,target)
    native.save(target/'media_identity.json',dict(source=native.proof(source),plan=native.proof(tmp_path/'plan.json')))
    native.save(target/'packets/001/output.json',[
        ['1','可靠部分','ok','',''],['2','','unresolved','中間','未定'],['3','','unresolved','後半','未定']])
    native.accept(target,'001','/root/test_native');native.build(target)
    native.save(tmp_path/'delivery_policy.json',dict(mode='local_only',max_deferred_cues=0))
    return tmp_path


def fake_ask(calls):
    def ask(prompt,path):
        calls.append(prompt)
        content=json.dumps({'rows':[['2','中间句','ok','',''],['3','后半句','ok','','']]},ensure_ascii=False)
        native.save(path,dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),model='deepseek-flash',
            response_format='json',finish_reason='stop',content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),provider_request_id='fake-test-receipt',
            usage=dict(prompt_tokens=100,completion_tokens=30,prompt_cache_hit_tokens=50)))
    return ask


def test_repair_only_unresolved_reuses_api_cache_and_preserves_native_proofs(work):
    target=work/'translations/v';original=(target/'delivery/manifest.json').read_bytes()
    receipt=(target/'packets/001/receipt.json').read_bytes()
    calls=[];gates=[]
    result,usage=supplement.run_job(work,'v',None,ask=fake_ask(calls),gate=lambda timeout:gates.append(timeout))
    assert result['status']=='ready' and Path(result['local_path']).exists()
    assert usage['off_peak_cost_estimate_cny']==pytest.approx(.000171)
    assert (target/'delivery/manifest.json').read_bytes()==original
    assert (target/'packets/001/receipt.json').read_bytes()==receipt
    assert native.read(target/'supplement/repair.json')['rows'][0][0]=='2'
    supplement.run_job(work,'v',None,ask=fake_ask(calls),gate=lambda timeout:gates.append(timeout))
    assert len(calls)==1 and gates==[300]
    assert supplement.candidates(work)==[]


def test_small_deferred_jobs_are_not_paid_again(work):
    native.save(work/'delivery_policy.json',dict(mode='local_only',max_deferred_cues=2))
    assert supplement.candidates(work)==[]


def test_explicit_small_cue_repair_updates_only_an_unchanged_managed_delivery(work):
    native.save(work/'delivery_policy.json',dict(mode='local_only',max_deferred_cues=2,supplement_small_deferred=True))
    initial=batch.finish_local(work,'v')
    old=Path(initial['local_path']).read_bytes()
    assert supplement.candidates(work)==['v']
    result,_=supplement.run_job(work,'v',None,ask=fake_ask([]),gate=lambda timeout:None)
    assert result['status']=='ready' and result['deferred_ids']==[]
    assert Path(result['local_path']).read_bytes()!=old
    assert supplement.candidates(work)==[]


def test_isolated_cues_stay_local_while_ordinary_cues_may_be_supplemented(work):
    target=work/'translations/v';path=target/'delivery/manifest.json'
    data=native.read(path);data['counts']['content_isolated']=1
    data['ledger'][0].update(status='content_isolated',translation='',reason='受限')
    data['ledger'][0]['source']['text']='隔离原文'
    native.save(path,data)
    assert supplement.candidates(work)==['v']
    _,payload=supplement.prepare_input(target)
    assert [row[0] for row in payload['rows']]==['2','3']
    assert all(row[0]!='1' for row in payload['context'])
    assert '隔离原文' not in json.dumps(payload,ensure_ascii=False)
    native.save(target/'content_hold.json',dict(reason='hold'))
    assert supplement.candidates(work)==[]
    with pytest.raises(ValueError,match='hold'):
        supplement.prepare_input(target)


def test_modified_api_receipt_blocks_delivery(work):
    supplement.run_job(work,'v',None,ask=fake_ask([]),gate=lambda timeout:None)
    record=next((work/'translations/v/supplement/requests').glob('*.json'))
    record.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):
        batch.build_for_delivery(work,'v')


def cue_repair_fixture(work):
    target=work/'translations/v';folder=target/'cue_repair'
    base=native.read(target/'delivery/manifest.json')
    checks=[]
    for row in base['ledger']:
        if row['status']!='unresolved':continue
        s=row['source'];a=s['start_ms']/1000;b=s['end_ms']/1000
        for window in ('near','context'):
            clip=folder/f'acoustic/{s["id"]}.{window}.wav'
            clip.parent.mkdir(parents=True,exist_ok=True);clip.write_bytes(b'test-only-clip')
            checks.append(dict(cue_id=s['id'],window=window,clip_start=a,clip_end=b,
                cue_start=a,cue_end=b,clip=native.proof(clip),text=s['text']))
    evidence=dict(binding=dict(base_manifest=native.proof(target/'delivery/manifest.json'),
                               source=base['source']),checks=checks,human_audio_verified=False)
    native.save(folder/'acoustic/large-v3.json',evidence)
    native.save(folder/'input.json',dict(rows=[['2','中間'],['3','後半']],evidence=checks))
    (folder/'rules.txt').write_text('test-only rules',encoding='utf-8')
    rows=[['2','中间句','ok','',''],['3','','unresolved','後半','新证据仍不足']]
    start=dict(state='started',thread_id='test-only-thread',turn_id='test-only-turn',
               model=native.MODEL,effort=native.EFFORT,input=native.proof(folder/'input.json'))
    native.save(folder/'app_server_turn_start.json',start)
    native.save(folder/'app_server_turn.json',dict(thread_id=start['thread_id'],model=native.MODEL,
        effort=native.EFFORT,input=start['input'],turn=dict(id=start['turn_id'],status='completed'),
        final_messages=[json.dumps(dict(content_hold=False,rows=rows),ensure_ascii=False)]))
    native.save(folder/'repair.json',dict(source=base['source'],base_manifest=evidence['binding']['base_manifest'],
        rows=rows,proofs=[native.proof(folder/p) for p in ('input.json','rules.txt','acoustic/large-v3.json',
                                                    'app_server_turn.json','app_server_turn_start.json')]))
    return target,folder


def test_acoustic_repair_preserves_originals_and_rebuilds_identical_delivery(work):
    target,folder=cue_repair_fixture(work)
    original=(target/'delivery/manifest.json').read_bytes()
    receipt=(target/'packets/001/receipt.json').read_bytes()
    one=batch.build_for_delivery(work,'v');two=batch.build_for_delivery(work,'v')
    assert one==two and one['counts']==dict(ok=2,unresolved=1)
    assert one['ledger'][0]['translation']=='可靠部分' and one['deferred_ids']==['3']
    assert one['human_audio_verified'] is False
    assert (target/'delivery/manifest.json').read_bytes()==original
    assert (target/'packets/001/receipt.json').read_bytes()==receipt


@pytest.mark.parametrize('tamper',['input','clip','turn','base','translated_id','sound','hold','isolated'])
def test_acoustic_repair_rejects_changed_evidence_and_forbidden_overrides(work,tamper):
    from subtitle_cue_repair import apply_repair
    target,folder=cue_repair_fixture(work);base=native.read(target/'delivery/manifest.json')
    path=folder/'repair.json';repair=native.read(path)
    if tamper=='input':native.save(folder/'input.json',{})
    elif tamper=='clip':Path(native.read(folder/'acoustic/large-v3.json')['checks'][0]['clip']['path']).write_bytes(b'changed')
    elif tamper=='turn':native.save(folder/'app_server_turn.json',{})
    elif tamper=='base':base['ledger'][0]['translation']='变更'
    elif tamper=='translated_id':repair['rows'][0][0]='1';native.save(path,repair)
    elif tamper=='sound':
        repair['rows'][0]=['2','','sound','','不能凭识别文本判断'];native.save(path,repair)
        turn=native.read(folder/'app_server_turn.json')
        turn['final_messages']=[json.dumps(dict(content_hold=False,rows=repair['rows']))]
        native.save(folder/'app_server_turn.json',turn)
        repair['proofs']=[native.proof(p['path']) for p in repair['proofs']];native.save(path,repair)
    elif tamper=='hold':native.save(target/'content_hold.json',dict(reason='test hold'))
    elif tamper=='isolated':base['counts']['content_isolated']=1
    with pytest.raises(ValueError):apply_repair(target,base)


def test_response_cannot_replace_an_already_accepted_id():
    content=json.dumps({'rows':[['1','不该修改','ok','','']]})
    with pytest.raises(ValueError,match='coverage'):
        supplement.parse_rows(content,['2'])


def test_missing_partial_span_preserves_original_without_retry(work):
    target=work/'translations/v';original=(target/'packets/001/receipt.json').read_bytes()
    calls=[]
    def ask(prompt,path):
        calls.append(prompt)
        content=json.dumps({'rows':[['2','可靠部分','partial','','存疑词未记录'],
                                    ['3','后半句','ok','','']]},ensure_ascii=False)
        native.save(path,dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),model='deepseek-flash',
            response_format='json',finish_reason='stop',content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),provider_request_id='test-missing-span',
            usage=dict(prompt_tokens=100,completion_tokens=30,prompt_cache_hit_tokens=50)))
    result,_=supplement.run_job(work,'v',None,ask=ask,gate=lambda timeout:None)
    assert result['status']=='needs_source_decision'
    repair=native.read(target/'supplement/repair.json')
    assert repair['rows']==[['2','','unresolved','中間','未定'],['3','后半句','ok','','']]
    assert [r['id'] for r in repair['validation_deferred']]==['2']
    assert (target/'packets/001/receipt.json').read_bytes()==original
    supplement.run_job(work,'v',None,ask=ask,gate=lambda timeout:None)
    assert len(calls)==1


def test_missing_span_does_not_hide_coverage_or_other_contract_errors():
    originals={'2':['2','','unresolved','原文','未定']}
    for rows in ([['3','部分','partial','','存疑']],
                 [['2','部分','partial','','存疑'],['2','重复','ok','','']],
                 [['2','含日文あ','partial','','存疑']]):
        with pytest.raises(ValueError):
            supplement.parse_rows_with_deferred_span(json.dumps({'rows':rows}),['2'],originals)
    assert supplement.parse_rows_with_deferred_span('{"content_hold":true,"rows":[]}',[],{})==(None,[])


def test_batches_send_only_neighbouring_context_not_other_pending_cues():
    payload=dict(rows=[[str(i),'原文'] for i in range(1,10)],context=[['0','近邻']])
    plans=list(supplement.plan_requests(payload))
    assert [len(rows) for rows,_ in plans]==[4,4,1]
    assert [row[0] for rows,_ in plans for row in rows]==[str(i) for i in range(1,10)]
    for rows,prompt in plans:
        data=json.loads(prompt[len(supplement.RULES)+1:])
        assert data['context']==[['0','近邻']]


def test_anytime_policy_disables_both_request_waits(work,monkeypatch):
    policy=native.read(work/'delivery_policy.json');policy['deepseek_off_peak_only']=False
    native.save(work/'delivery_policy.json',policy)
    monkeypatch.setenv('DEEPSEEK_OFF_PEAK_ONLY','1')
    gate=supplement.configure_timing(work)
    assert supplement.os.environ['DEEPSEEK_OFF_PEAK_ONLY']=='0'
    assert gate(300) is None


def test_quota_reserve_stops_supplement_before_api_and_rechecks_after_offpeak_wait(work):
    policy=dict(pause_supplement_with_gpt=True,state='running',
        translation_stop_at_remaining_percent=20,last_observed_remaining_percent=20)
    native.save(work/'quota_policy.json',policy)
    def no_request(*args):raise AssertionError('No paid request while extraction only')
    with pytest.raises(supplement.TranslationQuotaPaused):
        supplement.run_job(work,'v',None,ask=no_request,gate=no_request)
    assert not (work/'translations/v/supplement/attempt.json').exists()
    policy['last_observed_remaining_percent']=21;native.save(work/'quota_policy.json',policy)
    def quota_dropped(timeout):
        policy.update(state='paused_waiting',last_observed_remaining_percent=5)
        native.save(work/'quota_policy.json',policy)
    with pytest.raises(supplement.TranslationQuotaPaused):
        supplement.run_job(work,'v',None,ask=no_request,gate=quota_dropped)
    assert native.read(work/'translations/v/supplement/attempt.json')['status']=='waiting_offpeak'
    policy.update(state='running',last_observed_remaining_percent=100)
    native.save(work/'quota_policy.json',policy)
    supplement.require_translation_quota(work)


def test_restricted_index_keeps_references_without_dialogue(work):
    target=work/'translations/v';native.save(target/'content_hold.json',dict(reason='test classification'))
    result=batch.refresh_restricted_index(work)
    index=native.read(work/'restricted_review/index.json')
    assert result['content_hold_jobs']==1
    assert index['jobs']['v']['hold']==native.proof(target/'content_hold.json')
    assert index['jobs']['v']['source']==native.read(target/'media_identity.json')['source']
    assert '前半' not in (work/'restricted_review/index.json').read_text(encoding='utf-8')
    index_time=(work/'restricted_review/index.json').stat().st_mtime_ns
    readme_time=(work/'restricted_review/README.md').stat().st_mtime_ns
    batch.refresh_restricted_index(work)
    assert (work/'restricted_review/index.json').stat().st_mtime_ns==index_time
    assert (work/'restricted_review/README.md').stat().st_mtime_ns==readme_time
