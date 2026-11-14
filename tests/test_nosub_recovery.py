from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import os
import hashlib
import json
from pathlib import Path
import sys
import time
import wave
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from nosub_recovery import WindowAlignmentRepair,gpu_slot


@pytest.mark.parametrize('seconds,stderr',[(1,b'Error submitting packet to decoder'),(.6,b'')])
def test_verified_source_rejects_decoder_errors_and_missing_pcm(tmp_path,monkeypatch,seconds,stderr):
    import nosub_recovery
    job=tmp_path/'job';job.mkdir()
    data=b'verified media';video=dict(fid='123',n='video.mp4',s=len(data),sha=hashlib.sha1(data).hexdigest().upper())
    def download(meta,target):target.write_bytes(data)
    def run(command,**kwargs):
        if command[0]=='ffprobe':
            return NS(returncode=0,stdout=json.dumps(dict(packets=[dict(pts_time='0',duration_time='1')])).encode())
        with wave.open(command[-1],'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(b'\0\0'*round(16000*seconds))
        return NS(returncode=0,stderr=stderr)
    monkeypatch.setattr(nosub_recovery.subprocess,'run',run)
    with pytest.raises(ValueError,match='alternate source needed'):
        nosub_recovery.verified_local_audio(NS(download_media_verified=download),video,job,tmp_path/'downloads')
    failure=json.loads((job/'audio_recovery_failure.json').read_text(encoding='utf-8'))
    assert failure['decoder_error']==bool(stderr)
    assert failure['no_automatic_same_source_retry'] is True
    assert not (job/'full_audio.wav').exists()


def source(tmp_path):
    audio=tmp_path/'full_audio.wav'
    with wave.open(str(audio),'wb') as wav:
        wav.setparams((1,2,16000,0,'NONE','not compressed'));wav.writeframes(b'\0\0'*16000*20)
    return audio


def result(text='Hello',start=1,end=2):
    return NS(start=0,end=12,text=text,words=[NS(start=start,end=end,word=text)],
              no_speech_prob=.1,avg_logprob=-.1)


def test_local_repair_preserves_text_and_checks_two_real_windows(tmp_path):
    calls=[]
    def transcribe(samples,**kwargs):calls.append(len(samples));return iter([result()]),None
    repair=WindowAlignmentRepair(source(tmp_path),'en',tmp_path/'report.json')
    assert repair(NS(transcribe=transcribe),result(end=12))==[(1000,2000,'Hello')]
    assert calls==[192000,208000]


@pytest.mark.parametrize('replacement',[result('Different'),result(start=1.5,end=2.5),result(start=1,end=1)])
def test_local_repair_never_discards_words_or_fabricates_an_agreed_clock(tmp_path,replacement):
    values=iter([result(),replacement])
    def transcribe(*args,**kwargs):return iter([next(values)]),None
    repair=WindowAlignmentRepair(source(tmp_path),'en',tmp_path/'report.json')
    with pytest.raises(ValueError):repair(NS(transcribe=transcribe),result(end=12))
    assert repair.checks[0]['status']=='unresolved'


@pytest.mark.skipif(os.name!='nt',reason='Windows named GPU ownership mutex')
def test_gpu_slot_serializes_workers_from_different_batches(monkeypatch):
    monkeypatch.setattr('nosub_recovery.GPU_MUTEX_NAME','Local\\MVplusSubtitleGpuTest'+uuid.uuid4().hex)
    running=[];observed=[]
    def work(index):
        with gpu_slot():
            running.append(index);observed.append(len(running));time.sleep(.02);running.remove(index)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,range(2)))
    assert observed==[1,1]
