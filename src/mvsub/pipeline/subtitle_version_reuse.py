"""Local scheduling for alternate encodes; audio verification still decides reuse."""
from pathlib import Path
import re
import unicodedata

import native_subtitles as native
import config

ROOT=config.repo_root()


def catalogue(name):
    value=unicodedata.normalize('NFKC',Path(name).stem).casefold()
    value=re.sub(r'^(?:4k688\.com@|hhd800\.com@|hhd800@)','',value)
    match=re.search(r'(?<![a-z])([a-z]{2,8})[-_]0*(\d{2,6})(?!\d)',value)
    if not match:return None
    # Explicit parts are distinct even when their lengths happen to agree.
    parts=tuple(re.findall(r'(?:part|sp|pt)[ _-]*(\d+)',value))
    trailing=re.match(r'[-_](\d+)(?!\d)',value[match.end():])
    return match[1],int(match[2]),parts,(int(trailing[1]) if trailing else None)


def workspaces(work):
    work=Path(work).resolve();path=work/'version_reuse_policy.json'
    paths=[work]
    if path.exists():paths.extend(Path(p).resolve() for p in native.read(path)['workspaces'])
    return sorted(set(paths),key=lambda p:str(p).casefold())


def duration(row):
    value=row.get('duration_seconds')
    if value is None:
        path=ROOT/'subtitle/work/nosub/confirmed_20261104/probes'/(str(row['video']['fid'])+'.json')
        if path.exists():value=native.read(path).get('media',{}).get('format',{}).get('duration')
    return float(value) if value else None


def same_version_candidate(left,right):
    a,b=left['video'],right['video']
    if a.get('sha') and a['sha'].upper()==b.get('sha','').upper():return True
    key=catalogue(a['n'])
    if key is None or key!=catalogue(b['n']):return False
    first,second=duration(left),duration(right)
    return first is not None and second is not None and abs(first-second)<=.3


def decision(work,row):
    """Elect one producer across explicitly linked frozen batches, without models."""
    work=Path(work).resolve();fid=str(row['video']['fid']);paths=workspaces(work)
    plans={p:native.read(p/'plan.json') for p in paths if (p/'plan.json').exists()}
    if plans.get(work,{}).get('source_preparer')!='multilingual_word_clock_v1':
        return dict(action='produce')
    for publication_work in paths:
        path=publication_work/'cloud_placement/journals'/(fid+'.json')
        if path.exists() and native.read(path).get('stage')=='complete':
            return dict(action='complete')  # Never revise an already published result.
    candidates=[]
    for peer_work,plan in plans.items():
        for peer in plan['items']:
            if not same_version_candidate(row,peer):continue
            other=str(peer['video']['fid']);target=peer_work/'translations'/other
            if (target/'content_hold.json').exists():continue
            if (target/'blocked.json').exists():continue
            publication=None
            for publication_work in paths:
                path=publication_work/'cloud_placement/journals'/(other+'.json')
                if path.exists() and native.read(path).get('stage')=='complete':
                    publication=path;break
            prepared=(target/'media_identity.json').exists()
            candidates.append((0 if publication else 1 if prepared else 2,
                str(peer_work).casefold(),other,peer_work,publication))
    if not candidates:return dict(action='produce')
    _,_,other,peer_work,publication=min(candidates)
    if other==fid and peer_work==work:return dict(action='produce')
    result=dict(action='reuse' if publication else 'wait',peer_fid=other,peer_work=str(peer_work))
    if publication:result['publication_journal']=str(publication)
    return result


def for_fid(work,fid):
    work=Path(work);path=work/'plan.json'
    if not path.exists():return dict(action='produce')
    plan=native.read(path)
    if plan.get('source_preparer')!='multilingual_word_clock_v1':return dict(action='produce')
    row=next(r for r in plan['items'] if str(r['video']['fid'])==str(fid))
    return decision(work,row)


def remember_wait(work,fid,result):
    path=Path(work)/'translations'/str(fid)/'version_reuse_wait.json'
    if not path.exists() or native.read(path)!=result:native.save(path,result)
