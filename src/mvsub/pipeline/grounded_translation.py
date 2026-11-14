"""Source-grounded, bounded subtitle requests with durable response reuse."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import re
from pathlib import Path
import tempfile

from audio_archive import atomic_json
import zhconv


class SourceContextHold(RuntimeError):
    """A per-film hold must not pause unrelated films' financial ledger."""
    pass


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def parse_reply(raw, target):
    data = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(data, dict) or set(data) != {'items'} or not isinstance(data['items'], list):
        raise ValueError('Expected JSON object containing items array')
    source = {e[0]: e[3] for e in target}
    items = {}
    for item in data['items']:
        if not isinstance(item, dict) or not {'id', 'zh', 'review'} <= set(item):
            raise ValueError('Invalid translation item')
        if set(item)-{'id','zh','review','span'}:raise ValueError('Unexpected item fields')
        idx = item['id']
        if not isinstance(idx, str) or idx not in source or idx in items:
            raise ValueError('Unknown or duplicate subtitle ID')
        text, review = item['zh'], item['review']
        if not isinstance(text, str) or review not in {'none', 'source', 'context'}:
            raise ValueError('Invalid text/review value')
        if review != 'source' and not text.strip():
            raise ValueError('Empty accepted translation')
        if review=='none' and re.search(r'\[(?:原文)?待核\]|【(?:原文)?待核】|译者注[:：]',text):
            raise ValueError('A process marker cannot be published as dialogue')
        if review == 'source':
            span = item.get('span')
            if not isinstance(span, str) or not span or span not in source[idx]:
                raise ValueError('Source review requires an exact source span')
        if any('\u3040' <= c <= '\u30ff' for c in text) and review == 'none':
            raise ValueError('Japanese remains in accepted translation')
        items[idx] = dict(id=idx, zh=zhconv.convert(text.strip(), 'zh-hans'), review=review,
                          **({'span': item['span']} if review == 'source' else {}))
    return items, [e[0] for e in target if e[0] not in items]


def plan_batches(entries, batch_size=16, frame_size=48, margin=8, max_chars=2000, blocked=()):
    if batch_size < 1 or frame_size < batch_size or margin < 0 or max_chars < 1:
        raise ValueError('Invalid batch/reference limits')
    if len({e[0] for e in entries}) != len(entries):
        raise ValueError('Duplicate input IDs')
    blocked = set(blocked)
    plans = []
    for start in range(0, len(entries), frame_size):
        frame_end = min(len(entries), start + frame_size)
        reference = [e for e in entries[max(0, start-margin):min(len(entries), frame_end+margin)]
                     if e[0] not in blocked]
        targets = [e for e in entries[start:frame_end] if e[0] not in blocked]
        current = []; characters = 0
        for entry in targets:
            if len(entry[3]) > max_chars:
                raise ValueError(f'Single cue {entry[0]} exceeds text limit; review source before paying')
            if current and (len(current) >= batch_size or characters + len(entry[3]) > max_chars):
                plans.append((reference, current)); current = []; characters = 0
            current.append(entry); characters += len(entry[3])
        if current:
            plans.append((reference, current))
    return plans


def build_prompt(reference, target, system_prompt, names=None, omitted=()):
    # Fixed reference precedes the changing target, for native prefix caching.
    prefix = dict(reference=[dict(id=e[0], ja=e[3]) for e in reference],
                  confirmed_names=names or {}, omitted_source_ids=list(omitted))
    request = [dict(id=e[0], ja=e[3]) for e in target]
    return (system_prompt.strip() + '\n\n只供理解的相邻原文与已确认人名：\n'
            + json.dumps(prefix, ensure_ascii=False, separators=(',', ':'))
            + '\n\n仅翻译以下ID，不输出参考中的其他ID：\n'
            + json.dumps(request, ensure_ascii=False, separators=(',', ':'))
            + '\n返回JSON对象：{"items":[{"id":"原ID","zh":"译文","review":"none|source|context"}]}。'
              'source表示原文识别疑点，另给span且必须是该条原文的原样子串；不确定时zh可为空。'
              'context表示语义指向待核。不得把标记或说明塞进zh。')


def semantic_risks(source, translated):
    """Cheap review cues, not a claim that every alternative wording is wrong."""
    risks=[]
    if re.search(r'だけ(?!ど|れど)',source) and not re.search(r'只|仅|就|光|不止',translated):
        risks.append('source_limiter_needs_review')
    return risks


def translate(entries, output, system_prompt, ask, settings, batch_size=16, frame_size=48,
              margin=8, max_workers=1, blocked=(), reference_entries=None, names=None,failure_callback=None):
    output = Path(output)
    work = output.with_suffix(output.suffix+'.translation')
    cache = work/'requests'; cache.mkdir(parents=True, exist_ok=True)
    plans = plan_batches(entries, batch_size, frame_size, margin, blocked=blocked)
    if reference_entries is not None:
        plans = [([e for e in reference_entries if e[0] not in set(blocked)], target)
                 for _, target in plans]
    settings = dict(settings,system_prompt_sha256=sha(system_prompt.encode('utf-8')))
    results = {}; failures = []; cache_hits = 0

    def run(plan):
        reference, target = plan
        accepted = {}; hits = 0
        # At most one missing-ID repair, never repeat successful IDs.
        for _ in range(2):
            pending = [e for e in target if e[0] not in accepted]
            if not pending:
                break
            prompt = build_prompt(reference, pending, system_prompt, names, blocked)
            identity = sha(json.dumps(dict(settings=settings, prompt=prompt),
                                      ensure_ascii=False, sort_keys=True).encode('utf-8'))
            record = cache/(identity+'.json')
            if record.exists():
                saved = json.loads(record.read_text(encoding='utf-8'))
                if saved.get('prompt_sha256') != sha(prompt.encode('utf-8')):
                    raise ValueError('Response cache identity mismatch')
                if saved.get('content_sha256') != sha(saved['content'].encode('utf-8')):
                    raise ValueError('Response cache content changed')
                for key in ('model', 'reasoning_effort', 'max_output_tokens'):
                    if saved.get(key) != settings.get(key):
                        raise ValueError('Response cache settings mismatch')
                if saved.get('response_format') != 'json' or saved.get('finish_reason') != 'stop':
                    raise ValueError('Cached response did not complete; do not pay to blindly repeat it')
                raw = saved['content']; hits += 1
            else:
                raw = ask(prompt, record)
                # ask must durably capture the response before returning.
                if not record.exists():
                    raise RuntimeError('Response was not persisted; refusing volatile paid results')
            items, missing = parse_reply(raw, pending)
            source_by_id={e[0]:e[3] for e in pending}
            for idx,item in items.items():
                risks=semantic_risks(source_by_id[idx],item['zh'])
                if risks and item['review']=='none':item.update(review='context',host_review=risks)
            accepted.update(items)
            if not missing or not items:
                break
        return accepted, hits, [e[0] for e in target if e[0] not in accepted]

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, 2))) as pool:
        futures = {pool.submit(run, plan): plan for plan in plans}
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                items, hits, missing = future.result()
                results.update(items); cache_hits += hits
                if missing:
                    failures.append(dict(ids=missing, error='missing_ids_after_bounded_repair'))
            except Exception as error:
                failures.append(dict(ids=[e[0] for e in futures[future][1]],
                                     error=f'{type(error).__name__}: {error}'))
                if failure_callback and not isinstance(error,SourceContextHold):
                    failure_callback('Translation response/cache failed validation; inspect before new paid requests')
                for pending in futures:
                    pending.cancel()
            atomic_json(work/'progress.json', dict(items=list(results.values()), failures=failures))

    missing = [e[0] for e in entries if e[0] not in results]
    review = [i for i, item in results.items() if item['review'] != 'none']
    target_blocked=[e[0] for e in entries if e[0] in set(blocked)]
    passed = not missing and not review and not failures and not target_blocked
    report = dict(protocol='grounded-json-v1', source_sha256=sha(json.dumps(entries, ensure_ascii=False).encode('utf-8')),
                  settings=settings, source_hold_ids=target_blocked, omitted_reference_ids=list(blocked), review_ids=review,
                  items=list(results.values()), missing_ids=missing, failures=failures,
                  cache_hits=cache_hits, quality_passed=passed, output_updated=False)
    report['plan']=dict(batch_size=batch_size,reference_frame_size=frame_size,margin=margin,
                        max_target_characters=2000,external_reference=reference_entries is not None)
    if passed:
        from translate_srt_batch import write_srt
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            write_srt(temporary, [(e[0], e[1], e[2], results[e[0]]['zh']) for e in entries])
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        report.update(output_updated=True, output_sha256=sha(output.read_bytes()))
    atomic_json(work/'report.json', report)
    return report


def require_audit(source, output, review_path):
    """Publication gate: an old SRT file alone is not a completed new workflow."""
    from source_review import load_report,sha256
    from translate_srt_batch import parse_srt
    output=Path(output)
    audit=json.loads((output.with_suffix(output.suffix+'.translation')/'certificate.json').read_text(encoding='utf-8'))
    if (audit.get('protocol')!='grounded-json-v1' or audit.get('quality_passed') is not True
            or audit.get('output_updated') is not True or audit.get('missing_ids') or audit.get('review_ids')
            or audit.get('source_hold_ids') or audit.get('failures')
            or audit.get('source_file_sha256')!=sha256(source) or audit.get('output_sha256')!=sha256(output)):
        raise ValueError('Translation audit is incomplete or source/output changed')
    if audit.get('source_review_available') is not True or audit.get('source_review_sha256')!=sha256(review_path):
        raise ValueError('Current acoustic source review is required for publication')
    review=load_report(review_path,source)
    if review['hold_ids']:raise ValueError('Unresolved acoustic source review')
    if audit.get('reference_file'):
        from source_review import reference_source
        from translate_srt_batch import format_reference_context
        if sha256(audit['reference_file'])!=audit.get('reference_file_sha256'):
            raise ValueError('Reference context changed')
        format_reference_context(parse_srt(Path(audit['reference_file'])),parse_srt(reference_source(review)))
    a,b=parse_srt(Path(source)),parse_srt(output)
    if [e[:3] for e in a]!=[e[:3] for e in b]:raise ValueError('Rendered subtitle identity/timing changed')
    return audit


def write_certificate(report,folder):
    """Immutable acceptance content excludes mutable cache/latency diagnostics."""
    fields=('protocol','settings','plan','quality_passed','output_updated','source_file_sha256','output_sha256',
            'source_review_available','source_review_sha256','source_hold_ids','review_ids','missing_ids',
            'reference_file','reference_file_sha256','failures','human_audio_verified')
    atomic_json(Path(folder)/'certificate.json',{key:report.get(key) for key in fields})
