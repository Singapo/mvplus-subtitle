#!/usr/bin/env python3
"""
115 云盘下载/上传手工入口（单文件用）。
基于 p115sync + 直接 webapi 调用。

⚠ **115 接口知识已统一到 `cloud_common.py`（同目录）**：
   本文件是早期的一次性 CLI 封装，**没有**打 p115sync 的两个必要补丁
   （`upload_gettoken` 信封锁、尾片 Content-Length / 分片重试）。
   新代码请用 `from cloud_common import Cloud115`；批处理流水线
   （`nosub_pipeline.py`）都走那边。
   本文件保留仅为手工单文件下载/上传的便利入口，**不要在这里加新逻辑**。

用法:
    mvsub cloud ls [--remote-dir /]
    mvsub cloud download <pickcode|remote_path|share_url> [--outdir <dir>]
    mvsub cloud upload <local_path> [--remote-dir /]

认证:
    1. 环境变量 P115_COOKIES
    2. --cookies 参数
    3. 仓库根目录或当前目录的 .env 文件（推荐）
"""
import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import httpx
from tqdm import tqdm

import config

# 强制 stdout 使用 UTF-8，避免 Windows 终端中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TOOLS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = config.repo_root()  # 仓库根（.env 在这里）
DEFAULT_OUTDIR = config.download_dir()
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def get_cookies(args) -> dict:
    """从参数、环境变量或 .env 读取 115 cookie，返回 dict。"""
    cookie_str = (args.cookies or config.dotenv_value("P115_COOKIES")).strip()
    if not cookie_str:
        raise EnvironmentError(
            "未找到 115 cookie，请通过以下方式之一提供：\n"
            "  1. 环境变量 P115_COOKIES\n"
            "  2. --cookies 参数\n"
            f"  3. 在 {PROJECT_ROOT / '.env'} 写入 P115_COOKIES=UID=...;CID=...;SEID=...;KID=..."
        )

    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def init_client(cookies: dict):
    """初始化 p115sync 客户端。"""
    from p115sync import P115Client

    return P115Client("; ".join(f"{k}={v}" for k, v in cookies.items()))


def normalize_remote_path(path: str) -> str:
    """处理 MSYS2/Git Bash 自动路径转换问题：/xxx 会被转成 C:/Program Files/Git/xxx。"""
    prefix = "C:/Program Files/Git/"
    if path.startswith(prefix):
        return "/" + path[len(prefix):]
    return path


def api_list_dir(cookies: dict, cid: int | str):
    """直接调用 115 webapi 列目录，返回正确处理中文的条目列表。"""
    url = "https://webapi.115.com/files"
    params = {"cid": cid, "show_dir": 1, "offset": 0, "limit": 1000}
    resp = httpx.get(url, params=params, cookies=cookies, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.json()


def _is_dir_entry(cookies: dict, item: dict) -> bool:
    """判断条目是否为目录。115 根目录 listing 中部分目录 m=0，需要二次校验。"""
    if item.get("m") == 1:
        return True
    cid = item.get("cid")
    if item.get("fid") is not None or not cid:
        return False
    # 二次校验：把 cid 当作目录列一下，能成功说明是目录
    try:
        data = api_list_dir(cookies, cid)
        return bool(data.get("state"))
    except Exception:
        return False


def resolve_path_to_pickcode(cookies: dict, path: str) -> tuple[str, str]:
    """把 115 网盘路径解析为 (pickcode, name)。路径用 / 分隔。"""
    parts = [p for p in path.split("/") if p]
    cid = 0
    for i, part in enumerate(parts):
        data = api_list_dir(cookies, cid)
        found = None
        for item in data.get("data", []):
            name = item.get("n", "")
            if name == part:
                found = item
                break
        if not found:
            raise FileNotFoundError(f"路径不存在: {'/'.join(parts[:i+1])}")
        if i == len(parts) - 1:
            return found.get("pc"), found.get("n")
        if not _is_dir_entry(cookies, found):
            raise NotADirectoryError(f"不是目录: {'/'.join(parts[:i+1])}")
        cid = found.get("cid")
    raise ValueError("空路径")


def api_get_download_info(cookies: dict, pickcode: str) -> dict:
    """通过 webapi 获取 pickcode 的文件名和大小。"""
    url = "https://webapi.115.com/files/download"
    params = {"pickcode": pickcode}
    resp = httpx.get(url, params=params, cookies=cookies, headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    return resp.json()


def cmd_ls(args):
    """列出 115 网盘目录内容。"""
    cookies = get_cookies(args)
    remote_dir = normalize_remote_path(args.remote_dir or "/")
    print(f"[115] 列出目录: {remote_dir}")

    if remote_dir == "/":
        cid = 0
    else:
        cid = resolve_path_to_cid(cookies, remote_dir, create=False)

    data = api_list_dir(cookies, cid)
    for item in data.get("data", []):
        name = item.get("n", "")
        size = item.get("s", 0)
        is_dir = _is_dir_entry(cookies, item)
        kind = "DIR" if is_dir else "FILE"
        pickcode = item.get("pc", "")
        print(f"  [{kind}] {name}\tsize={size}\tpc={pickcode}")


def cmd_download(args):
    """下载 115 网盘文件。"""
    from p115sync import P115FileSystem

    cookies = get_cookies(args)
    client = init_client(cookies)
    fs = P115FileSystem(client)
    target = args.target
    outdir = Path(args.outdir).resolve() if args.outdir else DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    if target.startswith("http://") or target.startswith("https://"):
        _download_share(client, target, outdir)
    elif "/" in target or target.endswith((".mp4", ".mkv", ".avi", ".mov", ".srt")):
        # 先尝试作为路径解析 pickcode
        normalized = normalize_remote_path(target)
        try:
            pickcode, name = resolve_path_to_pickcode(cookies, normalized)
            _download_by_pickcode(cookies, fs, pickcode, outdir / name)
        except FileNotFoundError:
            # 可能 target 就是 pickcode
            _download_by_pickcode(cookies, fs, normalized, outdir)
    else:
        # 当作 pickcode
        _download_by_pickcode(cookies, fs, target, outdir)


def _download_share(client, url: str, outdir: Path):
    """从 115 分享链接下载所有文件。"""
    from p115sync import P115Helper

    helper = P115Helper(client)
    print(f"[115] 解析分享链接: {url}")
    files = helper.get_all_share_files(url)
    if not files:
        print("[error] 分享链接中没有找到文件", file=sys.stderr)
        sys.exit(1)
    for item in files:
        name = item.get("name") or item.get("file_name", "unknown")
        file_id = item.get("file_id") or item.get("id")
        try:
            info = helper.get_share_download_url(url, file_id=file_id)
            _download_one(info, outdir / name)
        except Exception as e:
            print(f"[error] 下载 {name} 失败: {e}", file=sys.stderr)


def _download_by_pickcode(cookies: dict, fs, pickcode: str, dst: Path | None = None):
    """通过 pickcode 下载单个文件。"""
    info = fs.get_download_info_from_pickcode(pickcode)

    # 获取真实文件名（115 对大文件会返回 state=false 但仍有 file_name）
    name = pickcode
    try:
        meta = api_get_download_info(cookies, pickcode)
        if meta.get("file_name"):
            name = meta.get("file_name")
    except Exception:
        pass

    if dst is None:
        dst = DEFAULT_OUTDIR / name
    elif dst.is_dir():
        dst = dst / name
    _download_one(info, dst)


def _download_one(info: dict, dst: Path):
    """根据 download_info 下载单个文件。"""
    url = info.get("url")
    headers = info.get("headers", {})
    if not url:
        print(f"[error] 无法获取下载地址: {dst.name}", file=sys.stderr)
        return

    print(f"[115] 开始下载: {dst.name}")
    timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(dst, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name
        ) as bar:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"[ok] 已下载: {dst}")


def resolve_path_to_cid(cookies: dict, path: str, *, create: bool = True) -> str:
    """把 115 网盘路径解析为最后一级目录的 cid。create=False 时目录不存在则报错。"""
    path = normalize_remote_path(path)
    parts = [p for p in path.split("/") if p]
    cid = "0"
    for i, part in enumerate(parts):
        data = api_list_dir(cookies, cid)
        found = None
        for item in data.get("data", []):
            if item.get("n") == part and _is_dir_entry(cookies, item):
                found = item
                break
        if found:
            cid = found.get("cid")
        elif create:
            url = "https://webapi.115.com/files/add"
            resp = httpx.post(
                url,
                data={"pid": cid, "cname": part},
                cookies=cookies,
                headers={"User-Agent": USER_AGENT},
            )
            result = resp.json()
            if not result.get("state"):
                raise RuntimeError(f"创建目录失败 {'/'.join(parts[:i+1])}: {result}")
            cid = result.get("cid")
        else:
            raise FileNotFoundError(f"目录不存在: {'/'.join(parts[:i+1])}")
    return cid


def get_upload_userkey(cookies: dict) -> tuple[str, str]:
    """通过 app/uploadinfo 获取 user_id 和 userkey（web cookie 也可用）。"""
    url = "https://proapi.115.com/app/uploadinfo"
    resp = httpx.get(url, cookies=cookies, headers={"User-Agent": "Mozilla/5.0 115disk/30.5.1"})
    resp.raise_for_status()
    data = resp.json()
    if not data.get("state"):
        raise EnvironmentError(f"获取 115 upload userkey 失败: {data}")
    return str(data["user_id"]), data["userkey"]


def cmd_upload(args):
    """上传文件到 115 网盘。"""
    from p115sync import P115UploadHelper

    cookies = get_cookies(args)
    client = init_client(cookies)

    local_path = Path(args.local_path).resolve()
    remote_dir = normalize_remote_path(args.remote_dir or "/")

    if not local_path.exists():
        print(f"[error] 本地文件不存在: {local_path}", file=sys.stderr)
        sys.exit(1)

    # 获取上传密钥并注入客户端
    user_id, user_key = get_upload_userkey(cookies)
    client.user_id = user_id
    client.user_key = user_key

    # 确保远程目录存在并获取 cid
    parent_cid = resolve_path_to_cid(cookies, remote_dir)

    upload_helper = P115UploadHelper(client)
    print(f"[115] 上传: {local_path} -> {remote_dir}/{local_path.name}")

    # upload_file_sample 内部会先尝试秒传，失败再走 OSS 单次/分片上传
    result = upload_helper.upload_file_sample(str(local_path), pid=int(parent_cid))
    print(f"[ok] 上传完成: {result}")


def main():
    parser = argparse.ArgumentParser(description="MVplus 115 网盘工具")
    parser.add_argument(
        "--cookies",
        default=None,
        help="115 cookie 字符串（UID=...;CID=...;SEID=...;KID=...）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ls
    ls_parser = subparsers.add_parser("ls", help="列出目录")
    ls_parser.add_argument("--remote-dir", default="/", help="远程目录路径（默认 /，目前仅支持根目录）")
    ls_parser.set_defaults(func=cmd_ls)

    # download
    dl_parser = subparsers.add_parser("download", help="下载文件")
    dl_parser.add_argument("target", help="115 pickcode、网盘路径或分享链接")
    dl_parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help=f"下载目录（默认 {DEFAULT_OUTDIR}）",
    )
    dl_parser.set_defaults(func=cmd_download)

    # upload
    ul_parser = subparsers.add_parser("upload", help="上传文件")
    ul_parser.add_argument("local_path", help="本地文件路径")
    ul_parser.add_argument("--remote-dir", default="/", help="远程目标目录（默认 /）")
    ul_parser.set_defaults(func=cmd_upload)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
