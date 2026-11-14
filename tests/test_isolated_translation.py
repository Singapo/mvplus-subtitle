import hashlib
import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import native_subtitles as native
import subtitle_batch as batch
import subtitle_isolated_translation as isolated


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
        ['1','可靠部分','ok','',''],['2','','content_isolated','','受限'],['3','','content_isolated','','受限']])
    native.accept(target,'001','/root/test_native');native.build(target)
    native.save(tmp_path/'delivery_policy.json',dict(mode='local_only',max_deferred_cues=2))
    return tmp_path


def fake_ask(rows):
    def ask(prompt,path):
        content=json.dumps({'rows':rows},ensure_ascii=False)
        native.save(path,dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),model='deepseek-flash',
            response_format='json',finish_reason='stop',content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),provider_request_id='fake-isolated',
            usage=dict(prompt_tokens=100,completion_tokens=40,prompt_cache_hit_tokens=20)))
    return ask


def test_isolated_cues_are_translated_and_delivered(work):
    assert isolated.candidates(work)==['v']
    result,_=isolated.run_job(work,'v',None,ask=fake_ask([
        ['2','中间句','ok','',''],['3','后半句','ok','','']]),gate=lambda timeout:None)
    assert result['status']=='ready' and Path(result['local_path']).exists()
    assert result['counts'].get('content_isolated',0)==0
    manifest=native.read(work/'translations/v/isolated/delivery/manifest.json')
    assert [r['translation'] for r in manifest['ledger']]==['可靠部分','中间句','后半句']
    assert isolated.candidates(work)==[]


def test_flagged_line_stays_blank_without_stopping_the_film(work):
    result,_=isolated.run_job(work,'v',None,ask=lambda prompt,path:native.save(
        path,dict(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),model='deepseek-flash',
            response_format='json',finish_reason='stop',content='{"content_hold":true,"rows":[]}',
            content_sha256=hashlib.sha256(b'{"content_hold":true,"rows":[]}').hexdigest(),
            provider_request_id='fake-hold',
            usage=dict(prompt_tokens=10,completion_tokens=5,prompt_cache_hit_tokens=0))),
        gate=lambda timeout:None)
    assert not (work/'translations/v/content_hold.json').exists()
    assert result['status'] in {'ready','ready_with_deferred'}
    manifest=native.read(work/'translations/v/isolated/delivery/manifest.json')
    assert [r['status'] for r in manifest['ledger']]==['ok','unresolved','unresolved']
    assert isolated.candidates(work)==[]
