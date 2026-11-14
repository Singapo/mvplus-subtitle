#!/usr/bin/env python3
"""115 网盘公共层：**所有 115 相关的补丁与接口知识只在这里实现一次**。

为什么存在这个模块（2026-09-22）：
    subtitle 与 demosaic 两条流水线各自实现了一遍 115 上传，结果同两个上游 bug
    被独立踩了两次、修了两次（`upload_gettoken` 信封、尾片 Content-Length），
    连上传 UA 都写成了两个版本（`115disk/30.5.1` 与 `115disk/37.0.2`）。
    这就是「每次都要重新调教」的结构性原因。**新增 115 逻辑请加在这里，不要各写一份。**

上游 p115sync 的两个 bug（都在本模块里绕过）：
    1. `upload_gettoken` 信封校验：legacy 接口 `uplb.115.com/3.0/gettoken.php`
       返回扁平体（StatusCode/AccessKeyId/... 在顶层，无 state/data），
       p115sync 却用 `{state:True, data:{...}}` 的信封去校验 → 成功响应被判失败
       → `P115APIError: 115 API 请求失败`。
       **只在 status=1（真传字节）时触发**；秒传命中(status=2)时根本不调用它。
    2. 尾片 Content-Length 声明过大：`_UploadWindowReader._declared_length()`
       无条件返回 max_bytes(128MB)，文件大小不是 128MB 整数倍时尾片实际不足，
       OSS 拒收 `Too little data for declared Content-Length`
       → 任何非 128MB 整数倍的大文件 Multipart 必然失败。
    3. 附带：每个分片没有重试 → 任一分片失败即从头上传。

用法：
    from cloud_common import Cloud115          # 下载 / 上传 / 列目录 / 删改名
    cloud = Cloud115(cookie_string)            # 自动打好全部补丁
    url, headers = cloud.download_url(pickcode)
    cloud.upload(local_path, cid, "远端名")
"""
from __future__ import annotations

import time
import json
import subprocess
import wave
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

_AUDIO_READ_LIMIT = threading.BoundedSemaphore(3)
from pathlib import Path
from typing import Any, BinaryIO

import httpx

# ------------------------------------------------- 常量（与 SOP / VR 手册一致）
UA_DISK = "Mozilla/5.0 115disk/30.5.1"      # 下载直链与 uploadinfo 必须带
UA_UPLOAD = "Mozilla/5.0 115disk/30.5.1"    # 上传沿用同一 UA（实测可用；VR 侧曾用 37.0.2，同样可用）
PART_SIZE = 128 * 1024 * 1024
SIMPLE_LIMIT = 5_000_000_000 - 1024 * 1024  # 5GB 以下走 p115sync 原单次 PUT 路径
PART_TRIES = 6


def _log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] [115] {msg}", flush=True)


def _truth(v: Any) -> bool:
    return v is True or v == 1 or (isinstance(v, str) and v.strip().lower() in ("true", "1"))


# ------------------------------------------------- 补丁 1：gettoken 信封
def install_token_compat(client) -> bool:
    """把 STS 扁平响应归一化成 p115sync 期望的 `{state:True, data:{...}}`。

    幂等：重复调用安全。
    """
    if getattr(client, "_token_compat_patched", False):
        return True
    original = client.upload_gettoken

    def upload_gettoken() -> dict:
        resp = original()
        if isinstance(resp, dict) and _truth(resp.get("state")):
            return resp
        fields = {"AccessKeyId", "AccessKeySecret", "SecurityToken", "Expiration"}
        if isinstance(resp, dict) and str(resp.get("StatusCode")) == "200" and fields & set(resp):
            return {"state": True, "data": {k: resp[k] for k in fields if k in resp}}
        return resp

    client.upload_gettoken = upload_gettoken
    client._token_compat_patched = True
    return True


# ------------------------------------------------- 补丁 2+3：OSS multipart 重试与尾片长度
def install_oss_upload_retry() -> bool:
    """把 p115sync 的 OSS Multipart 换成「分片级重试 + 尾片长度修正」版本。

    >4.66 GiB 的文件**必须**启用（缺省就永远传不上去）。
    幂等：装了就直接返回。
    """
    try:
        from p115sync import upload as U
    except Exception as exc:  # noqa: BLE001
        _log(f"无法导入 p115sync.upload: {exc}")
        return False
    if getattr(U, "_mvplus_retry_patched", False):
        return True

    def _request_retry(http_client, **kw):
        last = None
        for attempt in range(1, 4):
            try:
                return U._request_oss(http_client, **kw)
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(2 * attempt)
        raise last

    def multipart_retry(http_client, handle: BinaryIO, *, upload_url, bucket, object_name,
                        token, callback, part_size=PART_SIZE) -> httpx.Response:
        init = _request_retry(
            http_client, upload_url=upload_url, bucket=bucket, object_name=object_name,
            token=token, method="POST", params={"sequential": "1", "uploads": "1"},
            headers={"x-oss-security-token": str(token["SecurityToken"])})
        from xml.etree.ElementTree import fromstring
        upload_id = getattr(fromstring(init.content).find("UploadId"), "text", "")
        if not upload_id:
            raise RuntimeError("OSS 分片上传初始化未返回 UploadId")

        total = None
        try:
            cur = handle.tell()
            handle.seek(0, 2)
            total = handle.tell()
            handle.seek(cur)
        except Exception:  # noqa: BLE001
            pass
        _log(f"分片上传开始 uploadId={upload_id[:18]}… 总大小 "
             f"{total / 2 ** 30:.2f} GiB" if total else "分片上传开始")

        parts: list[dict[str, Any]] = []
        part_number = 1
        t_start = time.time()
        while True:
            if U._is_handle_exhausted(handle):
                break
            start_pos = handle.tell()
            got = 0
            last_exc = None
            for attempt in range(1, PART_TRIES + 1):
                handle.seek(start_pos)
                reader = U._UploadWindowReader(handle, max_bytes=part_size)
                try:
                    resp = U._request_oss(
                        http_client, upload_url=upload_url, bucket=bucket,
                        object_name=object_name, token=token, method="PUT",
                        params={"partNumber": str(part_number), "uploadId": upload_id},
                        headers={"x-oss-security-token": str(token["SecurityToken"])},
                        data=reader)
                    got = reader.bytes_read
                    etag = resp.headers.get("ETag")
                    if not etag:
                        raise RuntimeError("分片返回缺少 ETag")
                    parts.append({"PartNumber": part_number, "ETag": etag})
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    wait = min(30, 3 * attempt)
                    _log(f"⚠ 分片 {part_number} 第{attempt}/{PART_TRIES}次失败: "
                         f"{str(exc)[:120]} — {wait}s 后重发")
                    time.sleep(wait)
            if last_exc is not None:
                raise RuntimeError(f"分片 {part_number} 连续 {PART_TRIES} 次失败: {last_exc}")
            if got <= 0:
                break
            if total and part_number % 5 == 0:
                done = min(start_pos + got, total)
                rate = done / max(1e-6, time.time() - t_start) / 2 ** 20
                _log(f"分片 {part_number} 完成 进度 {done / 2 ** 30:.2f}/{total / 2 ** 30:.2f} GiB "
                     f"({done / total * 100:.0f}%)  {rate:.1f} MiB/s")
            if got < part_size:
                break
            part_number += 1

        if not parts:
            raise RuntimeError("OSS 分片上传未生成任何分片")
        _log(f"分片全部完成，共 {len(parts)} 片，用时 {(time.time() - t_start) / 60:.1f} 分钟，正在合并…")

        complete_body = b"".join((
            b"<CompleteMultipartUpload>",
            *[(f"<Part><PartNumber>{p['PartNumber']}</PartNumber>"
               f"<ETag>{p['ETag']}</ETag></Part>").encode("utf-8") for p in parts],
            b"</CompleteMultipartUpload>"))
        return _request_retry(
            http_client, upload_url=upload_url, bucket=bucket, object_name=object_name,
            token=token, method="POST", params={"uploadId": upload_id},
            headers={"x-oss-security-token": str(token["SecurityToken"]),
                     "x-oss-callback": U._to_base64(str(callback["callback"])),
                     "x-oss-callback-var": U._to_base64(str(callback["callback_var"])),
                     "content-type": "text/xml"},
            data=complete_body)

    U._oss_multipart_upload = multipart_retry

    def _declared_length_fixed(self) -> int | None:
        """尾片按「剩余实际字节」声明长度，否则 OSS 拒收。"""
        declared = self._initial_remaining
        try:
            file_size = int(__import__("os").fstat(self._handle.fileno()).st_size)
            avail = max(file_size - self._start_offset, 0)
            return avail if declared is None else min(declared, avail)
        except (AttributeError, OSError):
            return declared

    U._UploadWindowReader._declared_length = _declared_length_fixed
    U._mvplus_retry_patched = True
    _log("已安装 OSS 补丁：分片级重试 + 尾片 Content-Length 修正（>4.66 GiB 必须）")
    return True


# ------------------------------------------------- 客户端
class Cloud115:
    """115 客户端：打好全部补丁，提供下载 / 上传 / 列目录 / 改名 / 删除。

    所有技能知识集中在此，subtitle 与 demosaic 两条线共用。
    """

    def __init__(self, cookie: str, log=_log):
        from p115sync import P115Client, P115FileSystem, P115UploadHelper

        self.cookie = cookie
        self.log = log
        self.cookies = {k.strip(): v.strip() for k, v in
                        (p.split("=", 1) for p in cookie.split(";") if "=" in p)}
        self.client = P115Client(cookie)
        self.fs = P115FileSystem(self.client)
        install_oss_upload_retry()
        self._init_upload()
        self.uploader = P115UploadHelper(self.client)

    def _init_upload(self) -> None:
        """上传前置：拿 user_id/userkey。缺这步 upload_file_init 会报 errno 99 请重新登录。"""
        r = httpx.get("https://proapi.115.com/app/uploadinfo", cookies=self.cookies,
                      headers={"User-Agent": UA_DISK}, timeout=30).json()
        if not _truth(r.get("state")):
            raise RuntimeError(f"app/uploadinfo 失效（cookie 可能过期）: {str(r)[:200]}")
        self.client.user_id = str(r["user_id"])
        self.client.user_key = r["userkey"]
        install_token_compat(self.client)

    # ---- 列目录
    def list_dir(self, cid) -> list[dict]:
        """列目录（自动分页）。注意：目录条目的 id 字段是 cid，文件才有 fid。"""
        items: list[dict] = []
        offset = 0
        while True:
            data = self.client.fs_files({"cid": str(cid), "show_dir": 1, "offset": offset,
                                         "limit": 1000, "o": "file_name", "asc": 1})
            if not _truth(data.get("state", True)) and data.get("state") is False:
                raise RuntimeError(f"列目录失败 cid={cid}: {str(data)[:200]}")
            batch = data.get("data") or []
            items.extend(batch)
            offset += len(batch)
            if len(batch) < 1000 or offset >= int(data.get("count") or 0):
                break
        return items

    def resolve_path(self, path: str) -> str:
        """网盘路径 -> cid，如 /中文字幕/原文。"""
        cid = "0"
        for part in [p for p in path.split("/") if p]:
            found = next((it for it in self.list_dir(cid) if it.get("n") == part), None)
            if not found or found.get("fid") is not None:
                raise FileNotFoundError(f"路径不存在或不是目录: {part}")
            cid = str(found["cid"])
        return cid

    def mkdir(self, name: str, pid="0") -> str:
        """建目录（幂等：同名会返回 errno 20004，此时直接查回 cid）。"""
        r = httpx.post("https://webapi.115.com/files/add",
                       data={"pid": str(pid), "cname": name},
                       cookies=self.cookies,
                       headers={"User-Agent": UA_DISK}, timeout=30).json()
        if _truth(r.get("state")):
            return str(r["cid"])
        if r.get("errno") == 20004:
            found = next((it for it in self.list_dir(pid) if it.get("n") == name), None)
            if found:
                return str(found["cid"])
        raise RuntimeError(f"建目录失败 {name}: {str(r)[:200]}")

    # ---- 下载
    def download_url(self, pickcode: str) -> tuple[str, dict]:
        """取下载直链。

        只有 `get_download_info_from_pickcode` + `115disk` UA 这一条路能用：
          · `download_url`（app 接口）不带该 UA 报「请重新登录」
          · `download_url_web` 对大文件报 errno 50028「请使用115电脑端下载」
        返回的 headers 里混有非字符串值，必须过滤后交给 httpx。
        """
        info = self.fs.get_download_info_from_pickcode(pickcode, user_agent=UA_DISK)
        if not info.get("url"):
            raise RuntimeError(f"取不到下载直链: {str(info.get('response'))[:200]}")
        headers = {k: v for k, v in (info.get("headers") or {}).items()
                   if isinstance(v, (str, bytes))}
        return info["url"], headers

    # ---- 上传
    def download_verified(self, meta: dict, target: Path) -> None:
        """Bounded SRT download with exact byte/hash checks and private errors."""
        size = int(meta.get('s') or 0)
        if not 0 < size < 10_000_000:raise ValueError('Subtitle size outside allowed bounds')
        try:
            url, headers = self.download_url(meta['pc'])
            response = httpx.get(url, headers=headers, follow_redirects=True, timeout=120)
            response.raise_for_status()
            content = response.content
        except Exception:
            raise RuntimeError('Subtitle download failed; private URL withheld') from None
        if len(content) != size or (meta.get('sha') and hashlib.sha1(content).hexdigest().upper() != meta['sha'].upper()):
            raise RuntimeError('Subtitle download size/hash mismatch')
        target = Path(target);target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name+'.part');part.write_bytes(content);part.replace(target)

    def download_media_verified(self, meta: dict, target: Path) -> None:
        """Stream a media fallback with exact identity and checked Range resume."""
        size=int(meta.get('s') or 0);expected=str(meta.get('sha','')).upper()
        if size<=0 or len(expected)!=40:raise ValueError('Media size/hash identity missing')
        target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
        def valid(path):
            if path.stat().st_size!=size:return False
            sha=hashlib.sha1()
            with path.open('rb') as stream:
                while chunk:=stream.read(4*1024*1024):sha.update(chunk)
            return sha.hexdigest().upper()==expected
        if target.exists():
            if valid(target):return
            raise ValueError('Existing media differs; preserve without overwrite')
        part=target.with_name(target.name+'.part');offset=part.stat().st_size if part.exists() else 0
        if offset>size:raise ValueError('Media partial exceeds frozen size')
        if offset<size:
            try:
                url,headers=self.download_url(meta['pc']);headers=dict(headers)
                if offset:headers['Range']=f'bytes={offset}-'
                with httpx.stream('GET',url,headers=headers,follow_redirects=True,
                                  timeout=httpx.Timeout(30,read=120)) as response:
                    response.raise_for_status()
                    if response.status_code==206:
                        value=response.headers.get('Content-Range','')
                        if value!=f'bytes {offset}-{size-1}/{size}':raise ValueError('Unexpected media Content-Range')
                    elif response.status_code==200:offset=0
                    else:raise ValueError('Unexpected media download status')
                    with part.open('ab' if offset else 'wb') as stream:
                        total=offset
                        for chunk in response.iter_bytes(1024*1024):
                            total+=len(chunk)
                            if total>size:raise ValueError('Media response exceeds frozen size')
                            stream.write(chunk)
            except Exception:
                raise RuntimeError('Media stream failed; partial retained, URL withheld') from None
        if not valid(part):raise RuntimeError('Media download size/hash mismatch; partial retained')
        part.replace(target)

    def publish_missing_subtitle(self, video: dict, local: Path, journal: dict, save) -> dict:
        """Publish a missing SRT once; reconcile unknown uploads from a stable stage."""
        local=Path(local);cid=str(video['cid']);fid=str(video['fid'])
        target=Path(video['n']).stem+'.srt'
        expected=hashlib.sha1(local.read_bytes()).hexdigest().upper();size=local.stat().st_size
        stage=Path(target).stem+'.mvplus-stage-'+fid+'.srt'
        identity=dict(video_fid=fid,target_name=target,sha1=expected,size=size)
        if any(k in journal and journal[k]!=v for k,v in identity.items()):
            raise RuntimeError('Publication identity changed; retain previous journal')
        journal.update(identity);save()
        def fresh():
            rows=self.list_dir(cid)
            current=next((r for r in rows if str(r.get('fid'))==fid),None)
            if not current or current['n']!=video['n'] or (video.get('sha') and current.get('sha')!=video['sha']):
                raise RuntimeError('Source video changed before publication')
            return rows
        def verify(row):
            if int(row.get('s') or -1)!=size or str(row.get('sha') or '').upper()!=expected:
                raise RuntimeError('New subtitle size/SHA1 verification failed')
        rows=fresh();existing=[r for r in rows if r.get('n')==target]
        if existing:
            if len(existing)!=1 or str(existing[0]['fid'])!=journal.get('stage_fid'):
                raise RuntimeError('Canonical subtitle exists; preserve it without overwrite')
            verify(existing[0]);journal['stage']='complete';save();return journal
        hits=[r for r in rows if r.get('n')==stage]
        if not hits:
            if journal.get('upload_attempted'):
                raise RuntimeError('Previous upload result unknown; no automatic duplicate upload')
            journal['upload_attempted']=True;save()
            if not self.upload(local,cid,stage):raise RuntimeError('Staged upload failed; reconcile before retry')
            hits=[r for r in fresh() if r.get('n')==stage]
        if len(hits)!=1:raise RuntimeError('Staged subtitle missing or ambiguous')
        verify(hits[0]);journal.update(stage_fid=str(hits[0]['fid']),stage='staged');save()
        if any(r.get('n')==target for r in fresh()):raise RuntimeError('Canonical subtitle appeared during publication')
        if not self.rename(journal['stage_fid'],target):raise RuntimeError('Rename result unknown; retain journal')
        published=[r for r in fresh() if r.get('n')==target]
        if len(published)!=1 or str(published[0]['fid'])!=journal['stage_fid']:
            raise RuntimeError('Published subtitle identity differs')
        verify(published[0]);journal['stage']='complete';save();return journal

    def replace_subtitle(self, original: dict, local: Path, backup: Path, journal: dict, save,
                         target_name: str | None = None) -> dict:
        """Verified publication retaining old cloud SRTs with a 旧 suffix.

        Existing canonical subtitle is retained too when the selected version
        has a duplicate suffix. Persist every rename for rollback/resume.
        """
        cid, fid, name = str(original['cid']), str(original['fid']), original['n']
        target_name = target_name or name
        local, backup = Path(local), Path(backup)
        expected = hashlib.sha1(local.read_bytes()).hexdigest().upper()
        size = local.stat().st_size
        # 115 batch rename retains the existing extension: stage as .srt too.
        stage_name = Path(target_name).stem+'.mvplus-stage-'+fid+'.srt'
        def fresh():return self.list_dir(cid)
        def verify(row):
            if int(row.get('s') or -1) != size:raise RuntimeError('Cloud subtitle size mismatch')
            if row.get('sha'):
                if row['sha'].upper() != expected:raise RuntimeError('Cloud subtitle SHA1 mismatch')
            else:
                check=local.with_name(local.name+'.cloudcheck')
                self.download_verified(row,check)
                try:
                    if hashlib.sha1(check.read_bytes()).hexdigest().upper()!=expected:raise RuntimeError('Cloud bytes mismatch')
                finally:check.unlink(missing_ok=True)
        items=fresh()
        old=next((x for x in items if str(x.get('fid'))==fid),None)
        if not backup.exists():
            if not old or old['n']!=name:raise RuntimeError('Original changed or missing before backup')
            if original.get('sha') and str(old.get('sha','')).upper()!=original['sha'].upper():raise RuntimeError('Original subtitle changed since selection')
            self.download_verified(old,backup)
        if original.get('sha') and hashlib.sha1(backup.read_bytes()).hexdigest().upper()!=original['sha'].upper():
            raise RuntimeError('Rollback copy hash mismatch')
        stage_fid=journal.get('stage_fid')
        new=next((x for x in items if stage_fid and str(x.get('fid'))==str(stage_fid)),None)
        if new and new['n']==target_name:
            verify(new)
        else:
            hits=[x for x in items if x.get('n')==stage_name]
            if not hits:
                retained_name=next((x['new_name'] for x in journal.get('retained_old',[]) if x['fid']==fid),None)
                if not old or old['n'] not in (name,retained_name):raise RuntimeError('Original missing before staged upload')
                if old['n']==name and len([x for x in items if x.get('n')==name])!=1:raise RuntimeError('Ambiguous existing subtitle filename')
                if not self.upload(local,cid,stage_name):raise RuntimeError('Staged upload failed')
                hits=[x for x in fresh() if x.get('n')==stage_name]
            if len(hits)!=1:raise RuntimeError('Ambiguous staged upload; original retained')
            if new and str(new['fid'])!=str(hits[0]['fid']):
                verify(new)
                if '.mvplus-stage-'+fid in new['n']:
                    obsolete=journal.setdefault('obsolete_stage_fids',[])
                    if str(new['fid']) not in obsolete:obsolete.append(str(new['fid']))
            new=hits[0];verify(new)
            journal.update(stage_fid=str(new['fid']),stage='staged');save()
            if 'retained_old' not in journal:
                retained=[]
                for row in items:
                    if str(row.get('fid'))==fid or (row.get('n')==target_name and str(row.get('fid'))!=str(new['fid'])):
                        path=backup.parent.parent/str(row['fid'])/row['n']
                        if not path.exists():self.download_verified(row,path)
                        if row.get('sha') and hashlib.sha1(path.read_bytes()).hexdigest().upper()!=row['sha'].upper():
                            raise RuntimeError('Old canonical rollback hash mismatch')
                        old_name=row['n'];new_name=Path(old_name).stem+'旧.srt'
                        if any(x.get('n')==new_name for x in items):
                            new_name=Path(old_name).stem+'('+str(row['fid'])+')旧.srt'
                        if any(x.get('n')==new_name and str(x.get('fid'))!=str(row['fid']) for x in items):
                            raise RuntimeError('Retained old filename collision; original unchanged')
                        retained.append(dict(fid=str(row['fid']),old_name=old_name,new_name=new_name,sha=row.get('sha')))
                journal['retained_old']=retained;save()
            for row in journal['retained_old']:
                current=next((x for x in fresh() if str(x.get('fid'))==row['fid']),None)
                if not current or current['n'] not in (row['old_name'],row['new_name']):
                    raise RuntimeError('Old subtitle changed during publication')
                if row.get('sha') and str(current.get('sha','')).upper()!=row['sha'].upper():
                    raise RuntimeError('Old subtitle content changed during publication')
                if current['n']==row['old_name'] and not self.rename(row['fid'],row['new_name']):
                    raise RuntimeError('Old suffix rename failed; new subtitle remains staged')
                journal.update(stage='old_renamed');save()
            if not self.rename(new['fid'],target_name):
                for row in reversed(journal['retained_old']):self.rename(row['fid'],row['old_name'])
                raise RuntimeError('New subtitle rename failed; rollback attempted')
            journal.update(stage='published');save()
            new=next((x for x in fresh() if str(x.get('fid'))==str(new['fid'])),None)
            if not new or new['n']!=target_name:raise RuntimeError('Published subtitle missing')
            verify(new)
        for row in journal.get('retained_old',[]):
            current=next((x for x in fresh() if str(x.get('fid'))==row['fid']),None)
            if not current or current['n']!=row['new_name']:raise RuntimeError('Retained old subtitle missing')
        # Only our malformed older staging copies, never retained old SRTs.
        for obsolete in journal.get('obsolete_stage_fids',[]):
            row=next((x for x in fresh() if str(x.get('fid'))==obsolete),None)
            if row:
                verify(row)
                if '.mvplus-stage-'+fid not in row['n']:raise RuntimeError('Unknown staging name; refusing cleanup')
                if not self.delete(obsolete):raise RuntimeError('Obsolete staging cleanup failed')
        journal.update(stage='complete',stage_fid=str(new['fid']),sha1=expected,target_name=target_name);save()
        return journal

    def extract_audio(self, pickcode: str, dest: Path, chunk_seconds: int = 600) -> float:
        """Read only remote audio in resumable bounded chunks; never log URLs."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        url, headers = self.download_url(pickcode)
        header_text = ''.join(f'{k}: {v}\r\n' for k, v in headers.items())
        try:
            probe = subprocess.run(['ffprobe', '-v', 'error', '-rw_timeout', '30000000',
                '-headers', header_text, '-show_entries', 'stream=codec_type,start_time,duration:format=duration',
                '-of', 'json', url], capture_output=True, timeout=120)
        except subprocess.TimeoutExpired:
            raise RuntimeError('Remote audio probe timed out; signed URL withheld') from None
        if probe.returncode:
            raise RuntimeError('Remote audio probe failed; signed URL withheld')
        metadata = json.loads(probe.stdout)
        stream = next(s for s in metadata['streams'] if s['codec_type'] == 'audio')
        if abs(float(stream.get('start_time') or 0)) > .1:
            raise RuntimeError('Nonzero audio origin requires timeline review')
        duration = float(stream.get('duration') or metadata['format']['duration'])
        chunks = dest.parent / 'audio_chunks'
        chunks.mkdir(exist_ok=True)
        def wav_duration(path):
            try:
                with wave.open(str(path), 'rb') as f:
                    if (f.getframerate(), f.getnchannels(), f.getsampwidth()) != (16000, 1, 2):
                        return -1
                    return f.getnframes() / 16000
            except (OSError, EOFError, wave.Error):
                return -1
        def extract_chunk(offset, auth=None):
            length = min(chunk_seconds, duration - offset)
            if length <= 0:return
            target = chunks / f'{offset:08}.wav'
            if abs(wav_duration(target) - length) <= .15:return
            got = -1
            code = None
            current_url,current_header = auth or (url,header_text)
            part = target.with_suffix('.part.wav')
            try:
                with _AUDIO_READ_LIMIT:
                    result = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y',
                        '-rw_timeout', '30000000', '-headers', current_header, '-ss', str(offset),
                        '-i', current_url, '-t', str(length), '-vn', '-ar', '16000', '-ac', '1',
                        '-af', 'aresample=16000:async=16000:first_pts=0',
                        '-c:a', 'pcm_s16le', str(part)], capture_output=True, timeout=360)
                got = wav_duration(part)
                code = result.returncode
                good = code == 0 and abs(got - length) <= .15
            except subprocess.TimeoutExpired:
                good = False
            if good:
                part.replace(target)
                self.log(f'[audio] {offset:.0f}-{offset+length:.0f}s complete')
                return
            part.unlink(missing_ok=True)
            raise RuntimeError(f'Audio chunk {offset}s incomplete: expected {length:.3f}s, got {got:.3f}s, rc={code}; signed URL withheld')
        # Same three-download ceiling as the SOP; GPU inference remains serial.
        failed=[]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures=[(offset,pool.submit(extract_chunk,offset)) for offset in range(0,int(duration)+1,chunk_seconds)]
            for offset,future in futures:
                try:future.result()
                except RuntimeError:failed.append(offset)
        # Retry only failed blocks, after this file's parallel readers finish.
        # Refresh URL+Cookie together; never repeatedly reuse a rejected token.
        for offset in failed:
            for retry in range(2):
                fresh_url,fresh_headers=self.download_url(pickcode)
                fresh_header=''.join(f'{k}: {v}\r\n' for k,v in fresh_headers.items())
                try:
                    extract_chunk(offset,(fresh_url,fresh_header))
                    break
                except RuntimeError:
                    if retry==1:raise
        part = dest.with_suffix('.part.wav')
        with wave.open(str(part), 'wb') as out:
            out.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            for target in sorted(chunks.glob('[0-9]*.wav')):
                with wave.open(str(target), 'rb') as inp:
                    while frames := inp.readframes(16000 * 10):out.writeframesraw(frames)
        if abs(wav_duration(part) - duration) > .2:
            raise RuntimeError('Assembled audio duration mismatch')
        part.replace(dest)
        # Only our completed chunks, after full audio duration verification.
        for target in chunks.glob('[0-9]*.wav'):target.unlink()
        chunks.rmdir()
        return duration

    def upload(self, local: Path | str, cid, name: str, tries: int = 3) -> bool:
        """上传单个文件到 cid 目录，远端名为 name。

        **同名不会覆盖**：115 会并存两份同名文件，调用方需自行决定是否先处理旧文件。
        """
        local = Path(local)
        last = None
        for attempt in range(1, tries + 1):
            try:
                r = self.uploader.upload_file_sample(str(local), pid=int(cid), filename=name)
                if isinstance(r, dict) and _truth(r.get("state")):
                    return True
                last = r
                self.log(f"  上传未成功(第{attempt}/{tries}次) {str(r)[:150]}")
            except Exception as exc:  # noqa: BLE001
                last = exc
                self.log(f"  上传异常(第{attempt}/{tries}次) {type(exc).__name__}: {str(exc)[:150]}")
            time.sleep(5)
        self.log(f"  上传最终失败: {str(last)[:200]}")
        return False

    # ---- 改名 / 删除
    def rename(self, fid, new_name: str) -> bool:
        try:
            return bool(self.client.fs_rename((int(fid), new_name)).get("state"))
        except Exception as exc:  # noqa: BLE001
            self.log(f"  改名失败 fid={fid}: {str(exc)[:120]}")
            return False

    def delete(self, fid) -> bool:
        try:
            return _truth(self.client.fs_delete(int(fid)).get('state'))
        except Exception as exc:  # noqa: BLE001
            self.log(f"  删除失败 fid={fid}: {str(exc)[:120]}")
            return False

    def find_by_name(self, cid, name: str) -> list[dict]:
        return [it for it in self.list_dir(cid) if it.get("n") == name]

    # ---- 校验
    def verify_uploaded(self, cid, name: str, expect_size: int) -> tuple[bool, list[dict]]:
        """上传后核对：名字存在且大小一致，并返回全部同名条目（用于发现重复）。"""
        hits = self.find_by_name(cid, name)
        ok = any(int(it.get("s") or -1) == int(expect_size) for it in hits)
        return ok, hits
