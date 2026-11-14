import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
import cloud_common as cloud


def setup(monkeypatch,content,status=200,range_header=None):
    calls=[]
    class Response:
        status_code=status
        headers={'Content-Range':range_header} if range_header else {}
        def raise_for_status(self):pass
        def iter_bytes(self,size):yield content
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def stream(*args,**kwargs):calls.append(kwargs);return Response()
    monkeypatch.setattr(cloud.httpx,'stream',stream)
    client=cloud.Cloud115.__new__(cloud.Cloud115)
    client.download_url=lambda pc:('private-url',{'User-Agent':'test'})
    meta=dict(pc='p',s=6,sha=hashlib.sha1(b'abcdef').hexdigest())
    return client,meta,calls


def test_streamed_media_checks_full_hash_and_reuses_identical_file(tmp_path,monkeypatch):
    client,meta,calls=setup(monkeypatch,b'abcdef');target=tmp_path/'video.mp4'
    client.download_media_verified(meta,target);client.download_media_verified(meta,target)
    assert target.read_bytes()==b'abcdef' and len(calls)==1


def test_checked_range_resume_appends_only_remaining_bytes(tmp_path,monkeypatch):
    client,meta,calls=setup(monkeypatch,b'def',206,'bytes 3-5/6');target=tmp_path/'video.mp4'
    target.with_name(target.name+'.part').write_bytes(b'abc')
    client.download_media_verified(meta,target)
    assert target.read_bytes()==b'abcdef' and calls[0]['headers']['Range']=='bytes=3-'


def test_ignored_range_restarts_partial_without_duplicate_prefix(tmp_path,monkeypatch):
    client,meta,_=setup(monkeypatch,b'abcdef');target=tmp_path/'video.mp4'
    target.with_name(target.name+'.part').write_bytes(b'abc');client.download_media_verified(meta,target)
    assert target.read_bytes()==b'abcdef'


@pytest.mark.parametrize('content,status,header',[(b'def',206,'bytes 2-4/6'),(b'abcxyz',200,None)])
def test_bad_range_or_hash_never_publishes_verified_media(tmp_path,monkeypatch,content,status,header):
    client,meta,_=setup(monkeypatch,content,status,header);target=tmp_path/'video.mp4'
    if status==206:target.with_name(target.name+'.part').write_bytes(b'abc')
    with pytest.raises(RuntimeError):client.download_media_verified(meta,target)
    assert not target.exists()
