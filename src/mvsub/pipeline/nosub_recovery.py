"""Opt-in recovery evidence; source clocks and normal acceptance limits stay fixed."""
from contextlib import contextmanager
from pathlib import Path
import ctypes
import os
import json
import math
import shutil
import subprocess
import wave

import native_subtitles as native
import config

GPU_MUTEX_NAME='Local\\MVplusSubtitleGpu0'


def verified_local_audio(cloud,video,job,download_root=None):
    """A verified full file resolves remote timeouts without trusting header duration."""
    from audio_archive import digest,pcm_info,archive_audio
    fid=str(video['fid']);folder=(Path(download_root) if download_root else config.work_dir()/'recovery')/fid
    folder.mkdir(parents=True,exist_ok=True);source=folder/('video'+Path(video['n']).suffix)
    if not source.exists():
        if shutil.disk_usage(folder).free<int(video['s'])+5*2**30:raise RuntimeError('Insufficient disk for verified source fallback')
        cloud.download_media_verified(video,source)
    if source.stat().st_size!=int(video['s']) or digest(source)!=video['sha'].upper():
        raise ValueError('Local source identity mismatch')
    run=subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_packets',
        '-show_entries','packet=pts_time,duration_time','-of','json',str(source)],capture_output=True,timeout=180)
    if run.returncode:raise RuntimeError('Verified local audio packet probe failed')
    packets=json.loads(run.stdout).get('packets',[])
    if not packets:raise ValueError('Verified source has no audio packets')
    first=float(packets[0]['pts_time']);last=max(float(p['pts_time'])+float(p.get('duration_time',0)) for p in packets)
    if not math.isfinite(first) or not math.isfinite(last) or last<=0 or abs(first)>.1:
        raise ValueError('Invalid or nonzero packet origin needs review')
    audio=Path(job)/'full_audio.wav';part=audio.with_suffix('.recovery.part.wav')
    result=subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(source),'-map','0:a:0',
        '-vn','-ar','16000','-ac','1','-af','aresample=16000:async=16000:first_pts=0','-c:a','pcm_s16le',str(part)],
        capture_output=True,timeout=900)
    decoder_error=any(value in result.stderr for value in (b'Error submitting packet',b'Invalid data found',b'Error while decoding'))
    if result.returncode:raise RuntimeError('Verified local PCM extraction failed')
    info=pcm_info(part)
    if decoder_error or abs(info['duration']-last)>.2:
        native.save(Path(job)/'audio_recovery_failure.json',dict(video_fid=fid,source=native.proof(source),
            packet_count=len(packets),last_packet_end=last,pcm_seconds=info['duration'],
            decoder_error=decoder_error,reason='Verified source audio cannot be fully decoded',no_automatic_same_source_retry=True))
        raise ValueError('Verified source audio cannot be fully decoded; alternate source needed')
    part.replace(audio)
    native.save(Path(job)/'local_audio_recovery.json',dict(video=dict(fid=fid,sha=video['sha'],size=video['s']),
        local_source=native.proof(source),audio=native.proof(audio),packet_count=len(packets),
        first_packet_pts=first,last_packet_end=last,pcm_seconds=info['duration'],
        limitation='Verified source bytes and packet/PCM clock; not human audio verification'))
    archive_audio(audio,Path(video['n']).stem,fid)
    return audio


@contextmanager
def gpu_slot():
    """One GPU ASR owner across both the original and recovery batches."""
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateMutexW.argtypes=[wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR]
    api.CreateMutexW.restype=wintypes.HANDLE
    api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
    api.ReleaseMutex.argtypes=[wintypes.HANDLE];api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.CreateMutexW(None,False,GPU_MUTEX_NAME)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    acquired=False
    try:
        result=api.WaitForSingleObject(handle,0xffffffff)
        if result not in (0,0x80):raise RuntimeError('GPU ownership wait failed')
        acquired=True;yield
    finally:
        if acquired:api.ReleaseMutex(handle)
        api.CloseHandle(handle)


class WindowAlignmentRepair:
    """Two bounded re-recognitions must preserve text and agree on timing."""
    def __init__(self,audio,language,report):
        self.audio=Path(audio);self.language=language;self.path=Path(report);self.checks=[]
        with wave.open(str(audio)) as wav:self.duration=wav.getnframes()/wav.getframerate()

    def __call__(self,model,segment):
        import numpy as np
        from extract_srt import _word_cues
        compact=lambda text:''.join(text.split())
        if segment.end-segment.start>40:raise ValueError('Recovery span exceeds bounded window')
        check=dict(start=segment.start,end=segment.end,status='unresolved',windows=[])
        self.checks.append(check)
        try:
            choices=[]
            for padding in (0,1):
                left=max(0,segment.start-padding);right=min(self.duration,segment.end+padding)
                with wave.open(str(self.audio)) as wav:
                    wav.setpos(round(left*16000))
                    samples=np.frombuffer(wav.readframes(round((right-left)*16000)),dtype='<i2').astype(np.float32)/32768
                segments,_=model.transcribe(samples,language=self.language,beam_size=5,temperature=0,
                    vad_filter=False,word_timestamps=True,condition_on_previous_text=False,no_speech_threshold=.5)
                segments=list(segments)
                if compact(''.join(s.text for s in segments))!=compact(segment.text):
                    raise ValueError('Local recovery text differs; preserve source for review')
                cues=[]
                for item in segments:
                    if item.no_speech_prob>.5:raise ValueError('Local recovery speech confidence failed')
                    for a,b,text in _word_cues(item):
                        if item.avg_logprob<(-2.5 if b-a<2500 else -1):raise ValueError('Local recovery confidence failed')
                        cues.append((a+round(left*1000),b+round(left*1000),text))
                if not cues or any(not 0<=a<b<=self.duration*1000+1 for a,b,_ in cues):
                    raise ValueError('Invalid local recovery clock')
                check['windows'].append(dict(left=left,right=right,cues=cues));choices.append(cues)
            first,second=choices
            if len(first)!=len(second) or any(compact(a[2])!=compact(b[2]) or
                    max(abs(a[0]-b[0]),abs(a[1]-b[1]))>250 for a,b in zip(first,second)):
                raise ValueError('Local recovery windows disagree on timing')
            check['status']='text_preserved_two_window_timing_agreement'
            return first
        finally:
            native.save(self.path,dict(audio=native.proof(self.audio),language=self.language,
                method='Whisper bounded local re-recognition; no independent aligner',checks=self.checks))


def reuse_language_evidence(row,audio,job):
    """Bind prior language observations to exactly the same verified PCM bytes."""
    origin=row.get('recovery_origin')
    if not origin:return
    previous=Path(origin);old_audio=previous/'full_audio.wav';old_language=previous/'language.json'
    if not old_audio.exists() or not old_language.exists():return
    current=native.proof(audio)
    if native.proof(old_audio)['sha256']!=current['sha256']:raise ValueError('Recovery audio differs')
    observations=native.read(old_language)['observations']
    if row.get('recheck_language'):
        import numpy as np
        from faster_whisper import WhisperModel
        from faster_whisper.vad import get_speech_timestamps,VadOptions
        from asr_config import MODEL_HOME
        with wave.open(str(audio)) as wav:
            samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
        regions=get_speech_timestamps(samples,VadOptions(threshold=.5,min_speech_duration_ms=250,
            min_silence_duration_ms=500,speech_pad_ms=80))
        speech=np.concatenate([samples[r['start']:r['end']] for r in regions]) if regions else np.array([],dtype=np.float32)
        if len(speech)<6*16000:raise ValueError('Insufficient local speech for language recovery; no absence-of-dialogue claim')
        detector=WhisperModel(str(MODEL_HOME/'models/whisper-large-v3'),device='cpu',compute_type='int8',cpu_threads=4)
        observations=[]
        for clip in np.array_split(speech,3 if len(speech)>=9*16000 else 2):
            language,probability,_=detector.detect_language(audio=clip[:30*16000])
            observations.append(dict(language=language,probability=float(probability)))
        native.save(Path(job)/'language_recovery_observations.json',dict(audio=current,observations=observations,
            speech_seconds=len(speech)/16000,original_clock_unchanged=True,paid_requests=0))
    from nosub_sources import choose_language
    language=choose_language(observations)
    native.save(Path(job)/'language_speech_review.json',dict(status='language_resolved',language=language,
        observations=observations,audio=current,prior_language=native.proof(old_language)))
