from pathlib import Path
import sys
import wave

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import native_subtitles as native
import nosub_sources as sources


def test_language_needs_two_confident_agreeing_samples():
    assert sources.choose_language([dict(language='en',probability=.96),dict(language='en',probability=.91),dict(language='ja',probability=.9)])=='en'
    with pytest.raises(ValueError,match='Language uncertain'):
        sources.choose_language([dict(language='ja',probability=.99),dict(language='en',probability=.7),dict(language='en',probability=.6)])


def word_source(folder,end='00:00:01,000'):
    source=folder/'film.srt';source.write_text('1\n00:00:00,000 --> '+end+'\nHello.\n',encoding='utf-8')
    audio=folder/'full_audio.wav'
    with wave.open(str(audio),'wb') as stream:
        stream.setparams((1,2,16000,0,'NONE','not compressed'));stream.writeframes(b'\0\0'*32000)
    config=folder/'model.json';config.write_text('{}',encoding='utf-8')
    native.save(source.with_suffix('.word_clock.json'),dict(policy=sources.WORD_POLICY,quality_passed=True,language='en',
        source=native.proof(source),audio=native.proof(audio),model_config=native.proof(config),audio_seconds=2))
    return source,audio


def test_native_packet_binds_word_clock_evidence_and_rejects_changes(tmp_path):
    source,_=word_source(tmp_path);target=tmp_path/'translation'
    state=native.prepare(source,target)
    assert native.proof(source.with_suffix('.word_clock.json')) in state['inputs']
    source.with_suffix('.word_clock.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):native.checked(target)


def test_changed_audio_is_not_accepted_as_verified_source(tmp_path):
    source,audio=word_source(tmp_path);audio.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):native.prepare(source,tmp_path/'translation')


def test_cues_beyond_audio_are_blocked_before_model_dispatch(tmp_path):
    source,_=word_source(tmp_path,'00:00:03,000')
    with pytest.raises(ValueError,match='Invalid word-clock cue'):native.prepare(source,tmp_path/'translation')


def test_only_unattempted_language_blocks_can_use_a_resolved_local_review(tmp_path):
    target=tmp_path/'target';job=tmp_path/'job';target.mkdir();job.mkdir()
    native.save(job/'language_speech_review.json',dict(status='language_resolved',language='en'))
    native.save(target/'blocked.json',dict(reason='Language uncertain; retain audio without forcing Japanese'))
    assert sources.language_review_available(target,job)
    native.save(target/'blocked.json',dict(reason='Language uncertain; retain audio without forcing Japanese',speech_review_attempted=True))
    assert not sources.language_review_available(target,job)
    native.save(target/'blocked.json',dict(reason='Japanese alignment failed; see extract.log'))
    assert not sources.language_review_available(target,job)


def test_fully_reused_decisions_finish_without_waiting_for_a_model_packet(tmp_path,monkeypatch):
    import subtitle_batch as batch
    source,_=word_source(tmp_path)
    work=tmp_path/'batch';work.mkdir();(work/'extraction').mkdir();(work/'translations').mkdir()
    native.save(work/'plan.json',dict(items=[dict(video=dict(fid='v',n='film.mp4'))]))
    monkeypatch.setattr(batch,'cache_source',lambda *args:source)
    monkeypatch.setattr(batch,'prior_seed',lambda *args:None)
    monkeypatch.setattr(sources,'ORIG_ARCHIVE_DIR',tmp_path/'archive')
    monkeypatch.setattr(sources,'verified_link',lambda *args:None)
    monkeypatch.setattr(native,'prepare',lambda *args,**kwargs:dict(packets=[]))
    finished=[];monkeypatch.setattr(batch,'finish_local',lambda *args:finished.append(args))
    report=sources.prepare_sources(work,1)
    assert report['prepared_video_fids']==['v']
    assert finished==[(work,'v')]


@pytest.mark.parametrize('duration,hold,expected',[(1200.1,False,'peer'),(1190,False,None),(1200.1,True,None)])
def test_completed_peer_is_a_candidate_only_with_matching_catalogue_duration_and_no_hold(tmp_path,duration,hold,expected):
    row=dict(video=dict(fid='target',n='PRED-785.mp4'),duration_seconds=1200)
    peer=dict(video=dict(fid='peer',n='PRED-785(1).mp4'),duration_seconds=duration)
    native.save(tmp_path/'cloud_placement/journals/peer.json',dict(stage='complete'))
    if hold:native.save(tmp_path/'translations/peer/content_hold.json',dict(status='held'))
    native.save(tmp_path/'plan.json',dict(source_preparer='multilingual_word_clock_v1',items=[row,peer]))
    assert sources.versions.decision(tmp_path,row).get('peer_fid')==expected


def test_cross_version_reuse_helper_is_not_shipped(tmp_path,monkeypatch):
    """跨版本复用需要额外的一次性脚本，本发行版不带；缺失时必须明确失败而不是重跑 ASR。"""
    from types import SimpleNamespace
    import subtitle_batch as batch
    work=tmp_path/'batch';work.mkdir();(work/'extraction').mkdir()
    native.save(work/'plan.json',dict(items=[dict(video=dict(fid='v',n='PRED-785.mp4'))]))
    monkeypatch.setattr(sources.versions,'decision',lambda *args:dict(action='reuse',peer_fid='peer',peer_work='peer_work',publication_journal='journal'))
    called=[]
    monkeypatch.setattr(sources.subprocess,'run',lambda *args,**kwargs:called.append(args))
    monkeypatch.setattr(batch,'cache_source',lambda *args:pytest.fail('Missing reuse helper must not fall back to duplicate ASR'))
    report=sources.prepare_sources(work,1)
    assert not called
    assert report['failed_video_fids']==['v']
    reason=native.read(work/'translations/v/blocked.json')['reason']
    assert 'not part of this distribution' in reason
    assert 'before duplicate ASR' in reason


def test_unverified_version_does_not_fall_back_to_duplicate_asr(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import subtitle_batch as batch
    work=tmp_path/'batch';work.mkdir();(work/'extraction').mkdir()
    native.save(work/'plan.json',dict(items=[dict(video=dict(fid='v',n='PRED-785.mp4'))]))
    monkeypatch.setattr(sources.versions,'decision',lambda *args:dict(action='reuse',peer_fid='peer',peer_work='peer_work',publication_journal='journal'))
    monkeypatch.setattr(sources.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(returncode=1))
    monkeypatch.setattr(batch,'cache_source',lambda *args:pytest.fail('Failed reuse requires review'))
    report=sources.prepare_sources(work,1)
    assert report['failed_video_fids']==['v']
    assert not report['prepared_video_fids']
    assert 'before duplicate ASR' in native.read(work/'translations/v/blocked.json')['reason']
