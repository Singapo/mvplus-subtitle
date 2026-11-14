import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import native_subtitles as native


@pytest.fixture
def job(tmp_path):
    source=tmp_path/'movie.srt'
    source.write_text('1\n00:00:01,000 --> 00:00:02,000\n前半\n\n2\n00:00:02,000 --> 00:00:03,000\n後半\n',encoding='utf-8')
    work=tmp_path/'work';native.prepare(source,work)
    return source,work


def put_result(work,rows):
    native.save(work/'packets/001/output.json',rows)
    native.accept(work,'001','/root/actual_native_worker')


def test_single_pass_preserves_partial_text_and_original_time(job):
    _,work=job
    put_result(work,[['1','可靠部分','partial','未辨词','只保留可确认含义'],['2','后半句','ok','','']])
    result=native.build(work)
    assert result['counts']=={'partial':1,'ok':1}
    assert result['automatic_semantic_review'] is False
    assert [(r['start'],r['end']) for r in native.core.parse_srt(Path(result['output']['path']))]==[
        ('00:00:01,000','00:00:02,000'),('00:00:02,000','00:00:03,000')]


def test_unresolved_does_not_trigger_review_or_claim_ready(job):
    _,work=job
    put_result(work,[['1','能确认的句子','ok','',''],['2','','unresolved','後半','没有可靠语义']])
    result=native.build(work)
    assert result['status']=='needs_source_decision'
    assert result['counts']['unresolved']==1
    assert len(read_requests(work))==1
    assert not list(work.glob('*review*'))
    with pytest.raises(ValueError,match='publication not allowed'):
        native.publish(work,work/'nonexistent_plan.json')


def test_explicit_small_deferred_allowance_keeps_receipts_and_records_missing_ids(job):
    _,work=job
    put_result(work,[['1','可靠意思','ok','',''],['2','','unresolved','後半','关键含义未明']])
    receipt=(work/'packets/001/receipt.json').read_bytes()
    result=native.build(work,max_deferred=2)
    assert result['status']=='ready_with_deferred' and result['deferred_ids']==['2']
    assert (work/'packets/001/receipt.json').read_bytes()==receipt
    assert native.build(work)['status']=='needs_source_decision'


def test_minor_content_hold_cannot_be_relaxed_by_deferred_allowance(job):
    _,work=job
    native.save(work/'content_hold.json',{'reason':'content hold'})
    with pytest.raises(ValueError,match='hold'):
        native.build(work,max_deferred=2)


def read_requests(work):
    return list((work/'packets').glob('*/input.json'))


def test_content_isolation_no_longer_blocks_delivery_of_independent_dialogue(job):
    _,work=job
    put_result(work,[['1','独立普通对白','ok','',''],['2','','content_isolated','','受限内容']])
    result=native.build(work)
    assert result['status']=='ready'
    assert result['counts']=={'ok':1,'content_isolated':1}
    assert result['deferred_ids']==[]
    assert [r['text'] for r in native.core.parse_srt(Path(result['output']['path']))]==['独立普通对白']


@pytest.mark.parametrize('status,expected',[
    ('content_isolated','no_reliable_dialogue'),
    ('unresolved','needs_source_decision'),
    ('sound','no_reliable_dialogue')])
def test_no_translated_dialogue_keeps_ledger_and_never_becomes_publishable(job,status,expected):
    _,work=job
    put_result(work,[['1','',status,'','条目记录'],['2','','filler','','无新增信息']])
    result=native.build(work)
    assert result['status']==expected and len(result['ledger'])==2
    assert not Path(result['output']['path']).read_text('utf-8').strip()
    with pytest.raises(ValueError,match='publication not allowed'):
        native.publish(work,work/'nonexistent_plan.json')


def test_update_rules_preserves_inputs_and_refuses_sealed_results(job):
    _,work=job
    before=(work/'packets/001/input.json').read_bytes()
    native.update_unaccepted_rules(work)
    native.checked(work)
    assert (work/'packets/001/input.json').read_bytes()==before
    put_result(work,[['1','前半','ok','',''],['2','后半','ok','','']])
    with pytest.raises(ValueError,match='sealed'):
        native.update_unaccepted_rules(work)


@pytest.mark.parametrize('rows',[
    [['2','后半','ok','',''],['1','前半','ok','','']],
    [['1','前半','ok','',''],['1','重复','ok','','']],
    [['1','','partial','词','不确定'],['2','后半','ok','','']],
    [['1','前半','partial','','不确定'],['2','后半','ok','','']],
    [['1','[原文待核]','ok','',''],['2','后半','ok','','']],
    [['1','日本かな','ok','',''],['2','后半','ok','','']],
    [['1','','filler','',''],['2','后半','ok','','']],
])
def test_bad_outputs_rejected(job,rows):
    _,work=job
    with pytest.raises(ValueError):put_result(work,rows)


def test_source_change_blocks_resume_and_delivery(job):
    source,work=job;source.write_text(source.read_text('utf-8')+'\nchanged',encoding='utf-8')
    with pytest.raises(ValueError):native.checked(work)


def test_packet_and_resealed_control_change_is_detected(job):
    _,work=job;folder=work/'packets/001'
    payload=native.read(folder/'input.json');payload['rows'][0][1]='別の源文'
    native.save(folder/'input.json',payload)
    control=native.read(folder/'control.json');control['input']=native.proof(folder/'input.json')
    native.save(folder/'control.json',control)
    with pytest.raises(ValueError):native.checked(work)


def test_sealed_output_change_blocks_delivery(job):
    _,work=job;put_result(work,[['1','前半','ok','',''],['2','后半','ok','','']])
    native.save(work/'packets/001/output.json',[['1','已篡改','ok','',''],['2','后半','ok','','']])
    with pytest.raises(ValueError):native.build(work)


def test_exact_seed_reuse_excludes_targets_but_retains_neighbor_context(tmp_path):
    source=tmp_path/'movie.srt';source.write_text('1\n00:00:01,000 --> 00:00:02,000\n前半\n\n2\n00:00:02,000 --> 00:00:03,000\n後半\n',encoding='utf-8')
    prior=tmp_path/'actual_prior_result.json';native.save(prior,[['1','前半','ok','','']])
    seed=tmp_path/'seed.json';native.save(seed,dict(source_sha256=native.proof(source)['sha256'],rows=native.read(prior),proofs=[native.proof(prior)]))
    work=tmp_path/'work';state=native.prepare(source,work,seed)
    payload=native.read(work/'packets/001/input.json')
    assert payload['rows']==[['2','後半']]
    assert payload['context']==[['1','前半']]
    assert len(state['seed'])==1
    prior.write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError):native.checked(work)


def test_partial_seed_is_reused_without_erasing_uncertainty(tmp_path):
    source=tmp_path/'movie.srt';source.write_text('1\n00:00:01,000 --> 00:00:02,000\n原文\n',encoding='utf-8')
    prior=tmp_path/'prior.json';native.save(prior,[['1','可靠部分','partial','原文中未确认词','只保留已辨含义']])
    seed=tmp_path/'seed.json';native.save(seed,dict(source_sha256=native.proof(source)['sha256'],rows=native.read(prior),proofs=[native.proof(prior)]))
    work=tmp_path/'work';state=native.prepare(source,work,seed)
    assert state['packets']==[]
    result=native.build(work)
    assert result['counts']=={'partial':1}
    assert result['ledger'][0]['uncertain_span']=='原文中未确认词'


def test_content_hold_cannot_prepare_model_input(tmp_path):
    work=tmp_path/'held';work.mkdir();native.save(work/'content_hold.json',{'reason':'restricted source'})
    with pytest.raises(ValueError,match='Content hold'):
        native.prepare(tmp_path/'source.srt',work)


def test_evidence_focus_excludes_neighboring_sentence_and_deduplicates():
    entries=[dict(id='1',start='00:00:02,000',end='00:00:03,000',text='旧词'),
             dict(id='2',start='00:00:03,000',end='00:00:04,000',text='旧词二')]
    segments=[dict(start=0,end=6,text='前句 候选 候选二 后句',words=[
        dict(start=0,end=1,text='前句'),dict(start=2.2,end=2.8,text='候选'),
        dict(start=3.2,end=3.8,text='候选二'),dict(start=5,end=6,text='后句')])]
    rows,shared,missing=native.evidence_from_segments(entries,segments)
    assert rows=={'1':[['s0','候选']],'2':[['s0','候选二']]}
    assert len(shared)==1 and missing==[]


def test_agreeing_evidence_is_covered_without_duplicate_prompt_text():
    entries=[dict(id='1',start='00:00:01,000',end='00:00:02,000',text='こんにちは。')]
    rows,shared,missing=native.evidence_from_segments(entries,[dict(start=1,end=2,text='こんにちは',
        words=[dict(start=1.1,end=1.9,text='こんにちは')])])
    assert rows=={'1':[]} and shared=={} and missing==[]


def test_repetition_is_not_treated_as_independent_support():
    entries=[dict(id='1',start='00:00:01,000',end='00:00:02,000',text='普通の台詞')]
    rows,shared,missing=native.evidence_from_segments(entries,[dict(start=1,end=2,text='あ'*80,
        words=[dict(start=1,end=2,text='あ'*80)])])
    assert rows=={'1':[]} and not shared and missing==['1']


@pytest.mark.parametrize('short_answer',['不','停','好'])
def test_informative_short_answer_is_kept_while_filler_is_recorded(job,short_answer):
    _,work=job
    put_result(work,[['1',short_answer,'ok','',''],['2','','filler','','上下文中无新增信息的语气尾音']])
    result=native.build(work)
    cues=native.core.parse_srt(Path(result['output']['path']))
    assert result['status']=='ready'
    assert result['counts']=={'ok':1,'filler':1}
    assert [e['text'] for e in cues]==[short_answer]
    assert result['ledger'][1]['reason']=='上下文中无新增信息的语气尾音'
