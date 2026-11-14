"""Retain checked source cues while explicitly deferring one reversed pair."""
from pathlib import Path

from audio_archive import digest
import native_subtitles as native
from aligned_asr.utterance import pipeline_revision, review_digest
from source_review import seconds
from translate_srt_batch import parse_srt

POLICY='explicit_source_gap_for_reversed_pair_v1'


def selection(source, report, audio, deferred_ids):
    if report.get('pipeline_revision')!=pipeline_revision():
        raise ValueError('Original source pipeline changed')
    if report.get('quality_passed') is not False or report.get('source_sha1')!=digest(source):
        raise ValueError('Original failed candidate/report binding differs')
    if report.get('audio_sha1')!=digest(audio):
        raise ValueError('Original exact audio required; do not reuse another PCM clock')
    if report.get('manual_review_digest')!=review_digest(Path(source).parent):
        raise ValueError('Original manual review changed')
    entries=parse_srt(Path(source));required=set();backwards=[];previous=None
    if len({e[0] for e in entries})!=len(entries):raise ValueError('Duplicate source IDs')
    for entry in entries:
        start,end=seconds(entry[1]),seconds(entry[2])
        if not entry[3] or not 0<=start<end<=report['audio_seconds']+.001:
            raise ValueError('Only an otherwise valid reversed pair may be deferred')
        if previous and start<seconds(previous[1]):
            required.update((previous[0],entry[0]));backwards.append(entry)
        previous=entry
    issues=report.get('quality_issues',[])
    if len(backwards)!=1 or len(required)!=2 or set(deferred_ids)!=required:
        raise ValueError('Defer exactly the independently identified reversed pair')
    offending=backwards[0]
    if (len(issues)!=2 or {r.get('type') for r in issues}!={'invalid_cue','backward_cue'}
            or any(abs(r.get('start',-1)-seconds(offending[1]))>.001 for r in issues)
            or str(next(r for r in issues if r['type']=='invalid_cue').get('id'))!=offending[0]
            or next(r for r in issues if r['type']=='backward_cue').get('text')!=offending[3]):
        raise ValueError('Unresolved source issues beyond the exact reversed pair')
    retained=[e for e in entries if e[0] not in required]
    if len(retained)<len(required):raise ValueError('Insufficient retained source')
    if any(seconds(b[1])<seconds(a[1]) for a,b in zip(retained,retained[1:])):
        raise ValueError('Retained subset still has a clock conflict')
    return retained,[e for e in entries if e[0] in required]


def build(source, original_report, audio, output, deferred_ids, reason):
    source,original_report,audio,output=map(Path,(source,original_report,audio,output))
    if not reason.strip():raise ValueError('Explicit source-gap reason required')
    if output.exists() or output.with_suffix('.alignment.json').exists():
        raise ValueError('Preserve existing source; do not overwrite a prepared subset')
    report=native.read(original_report)
    retained,deferred=selection(source,report,audio,deferred_ids)
    output.parent.mkdir(parents=True,exist_ok=True)
    if audio.resolve()!=(output.parent/'full_audio.wav').resolve():
        raise ValueError('Subset must bind its local full audio')
    output.write_text('\n\n'.join(f'{e[0]}\n{e[1]} --> {e[2]}\n{e[3]}' for e in retained)+'\n',encoding='utf-8')
    review=dict(policy=POLICY,quality_scope='retained_cues_only',full_source_complete=False,
        inputs=[native.proof(source),native.proof(original_report),native.proof(audio)],
        source=native.proof(source),original_report=native.proof(original_report),audio=native.proof(audio),
        output=native.proof(output),reason=reason,
        deferred=[dict(id=e[0],start=e[1],end=e[2],text=e[3],status='source_unresolved') for e in deferred])
    path=output.parent/'source_subset_review.json';native.save(path,review)
    metadata=dict(report,quality_passed=True,quality_issues=[],original_quality_issues=report['quality_issues'],
        quality_scope='retained_cues_only',full_source_complete=False,
        manual_review_digest=review_digest(output.parent),cues=len(retained),source_sha1=digest(output),
        source_subset_review_sha1=digest(path),source_deferred_cues=len(deferred))
    native.save(output.with_suffix('.alignment.json'),metadata)
    verify(output.parent,metadata)
    return review


def verify(job, report):
    job=Path(job);path=job/'source_subset_review.json'
    if digest(path)!=report.get('source_subset_review_sha1'):raise ValueError('Source-gap review changed')
    value=native.read(path)
    if (value.get('policy')!=POLICY or value.get('quality_scope')!='retained_cues_only'
            or value.get('full_source_complete') is not False or not value.get('reason','').strip()):
        raise ValueError('Source gap must remain explicit')
    native.check_proofs(value['inputs']+[value['source'],value['original_report'],value['audio'],value['output']])
    source=Path(value['source']['path']);audio=Path(value['audio']['path']);output=Path(value['output']['path'])
    if audio.resolve()!=(job/'full_audio.wav').resolve() or output.parent.resolve()!=job.resolve():
        raise ValueError('Source-gap files escaped their job')
    original=native.read(value['original_report']['path'])
    retained,deferred=selection(source,original,audio,[e['id'] for e in value['deferred']])
    expected=[dict(id=e[0],start=e[1],end=e[2],text=e[3],status='source_unresolved') for e in deferred]
    if value['deferred']!=expected or parse_srt(output)!=retained:
        raise ValueError('Retained text, cue IDs or clocks changed')
    if (report.get('quality_scope')!='retained_cues_only' or report.get('full_source_complete') is not False
            or report.get('source_deferred_cues')!=len(deferred) or report.get('cues')!=len(retained)
            or report.get('original_quality_issues')!=original['quality_issues']
            or report.get('audio_sha1')!=original['audio_sha1'] or report.get('source_sha1')!=digest(output)):
        raise ValueError('Subset was misrepresented as complete or rebound')
    changed={'quality_passed','quality_issues','manual_review_digest','cues','source_sha1'}
    if any(report.get(key)!=item for key,item in original.items() if key not in changed):
        raise ValueError('Original quality evidence was changed')
    return value
