"""Keep verified full PCM audio beside local Japanese originals."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import wave

import config

ARCHIVE_DIR=config.archive_dir()


def digest(path):
    h=hashlib.sha1()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(4*1024*1024):h.update(chunk)
    return h.hexdigest().upper()


def pcm_info(path):
    with wave.open(str(path),'rb') as stream:
        if (stream.getframerate(),stream.getnchannels(),stream.getsampwidth(),stream.getcomptype())!=(16000,1,2,'NONE'):
            raise ValueError('Expected full mono 16kHz PCM16 audio')
        frames=stream.getnframes();actual=0
        while chunk:=stream.readframes(262144):actual+=len(chunk)
        if frames<=0 or actual!=frames*2:raise ValueError('Incomplete PCM audio')
    return dict(frames=frames,sample_rate=16000,duration=frames/16000)


def atomic_json(path,data):
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.part',delete=False) as stream:
        temp=Path(stream.name);stream.write(json.dumps(data,ensure_ascii=False,indent=2).encode('utf-8'))
    try:temp.replace(path)
    finally:temp.unlink(missing_ok=True)


def verified_link(source,target,sha,size):
    if target.exists():
        if target.stat().st_size!=size or digest(target)!=sha:raise ValueError('Existing archive differs')
        return
    with tempfile.NamedTemporaryFile(dir=target.parent,suffix='.part',delete=False) as stream:temp=Path(stream.name)
    try:
        temp.unlink()
        try:os.link(source,temp)
        except OSError:shutil.copy2(source,temp)
        if temp.stat().st_size!=size or digest(temp)!=sha or digest(source)!=sha:
            raise ValueError('Audio changed during archive')
        temp.replace(target)
    finally:temp.unlink(missing_ok=True)


def archive_audio(source,stem,video_id=None,archive_dir=ARCHIVE_DIR):
    source=Path(source);folder=Path(archive_dir)
    if Path(stem).name!=stem:raise ValueError('Invalid video stem')
    info=pcm_info(source);sha=digest(source);size=source.stat().st_size
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/(stem+'.audio.wav')
    if target.exists() and (target.stat().st_size!=size or digest(target)!=sha):
        # Keep every existing audio version; never overwrite unknown content.
        target=folder/(stem+'.audio.'+sha+'.wav')
    verified_link(source,target,sha,size)
    manifest=dict(file=target.name,sha1=sha,bytes=size,video_id=str(video_id) if video_id is not None else None,
        source=str(source.resolve()),storage='hardlink_or_verified_copy',**info)
    atomic_json(folder/(stem+'.audio.json'),manifest)
    return target,manifest


def restore_audio(target,stem,video_id,archive_dir=ARCHIVE_DIR):
    target=Path(target);folder=Path(archive_dir);meta=folder/(stem+'.audio.json')
    if not meta.exists():return False
    data=json.loads(meta.read_text('utf-8'))
    if data.get('video_id')!=str(video_id):return False
    name=data.get('file','')
    if Path(name).name!=name or not name.startswith(stem+'.audio') or not name.endswith('.wav'):
        raise ValueError('Invalid archived audio path')
    source=folder/name
    if source.stat().st_size!=data['bytes'] or digest(source)!=data['sha1']:
        raise ValueError('Archived audio checksum differs')
    pcm_info(source);target.parent.mkdir(parents=True,exist_ok=True)
    verified_link(source,target,data['sha1'],data['bytes'])
    return True
