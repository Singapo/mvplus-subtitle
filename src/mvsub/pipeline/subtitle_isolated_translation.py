"""Translate cues a previous native pass had isolated.

Policy change 2026-10-31: adult fictional dialogue is translated on its own
merits and isolation is no longer a terminal status. Minor-related content
still stops the whole film through content_hold, and is never sent onwards.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import native_subtitles as native
from translation_budget import TranslationBudget, BudgetExceeded

RULES = (native.RULES + '\n下面每一行都要给出中文译文，不能留空、不能跳过，'
         '不要输出 content_isolated，也不要输出 content_hold。'
         '按原意自然翻译，保留语气、情绪、称谓、人物关系和节奏；原文的情色、挑逗、亲密表达要如实、'
         '生动地译出，不要弱化成中性说明，也不要添加原文没有的动作或情节。'
         'status 只用 ok/partial；只有确实听不清的才用 unresolved。'
         '只输出JSON对象 {"rows":[["id","中文","status","存疑原文词段","简短原因"]]}。')
BATCH_SIZE = 4
MAX_OUTPUT_TOKENS = 16384


def candidates(work):
    work = Path(work)
    result = []
    for path in (work / 'translations').glob('*/delivery/manifest.json'):
        target = path.parent.parent
        if (target / 'content_hold.json').exists():
            continue
        if (target / 'isolated/repair.json').exists():
            continue
        manifest = native.read(path)
        if not manifest['counts'].get('content_isolated'):
            continue
        result.append((manifest['counts']['content_isolated'], target.name))
    return [fid for _, fid in sorted(result)]


def prepare_input(target):
    target = Path(target)
    base = native.read(target / 'delivery/manifest.json')
    native.check_proofs(base['inputs'] + [base['output'], base['source']])
    rows = base['ledger']
    positions = [i for i, r in enumerate(rows) if r['status'] == 'content_isolated']
    contexts = {}
    for i in positions:
        near = [j for j in range(max(0, i - 2), min(len(rows), i + 3))
                if j != i and rows[j]['status'] != 'content_isolated']
        contexts[rows[i]['source']['id']] = [[rows[j]['source']['id'], rows[j]['source']['text']]
                                             for j in near]
    payload = dict(rows=[[rows[i]['source']['id'], rows[i]['source']['text']] for i in positions],
                   contexts=contexts)
    folder = target / 'isolated'
    folder.mkdir(exist_ok=True)
    path = folder / 'input.json'
    if path.exists() and native.read(path) != payload:
        old = native.read(path)
        if 'contexts' in old:
            raise ValueError('Isolated input changed')
        # Old format carried the whole film's targets as context; replace it with
        # per-cue neighbourhoods so prompt size no longer scales with film length.
        path.replace(folder / 'input.old-format.json')
    if not path.exists():
        native.save(path, payload)
    rules = folder / 'rules.txt'
    if rules.exists() and rules.read_text('utf-8') != RULES:
        # Policy revision: keep the old wording as evidence, then continue with the new one.
        backup = folder / 'rules.prev-20261031.txt'
        backup.unlink(missing_ok=True)
        rules.replace(backup)
    if not rules.exists():
        rules.write_text(RULES, encoding='utf-8')
    return base, payload


def plan_requests(payload):
    for offset in range(0, len(payload['rows']), BATCH_SIZE):
        rows = payload['rows'][offset:offset + BATCH_SIZE]
        if 'contexts' in payload:
            seen = {}
            for row in rows:
                for pair in payload['contexts'].get(row[0], []):
                    seen[pair[0]] = pair
            context = list(seen.values())
        else:
            context = payload['context']
        # Only the neighbouring source lines travel with each batch, so prompt size
        # stays flat instead of repeating every other target row.
        yield rows, RULES + '\n' + json.dumps(dict(context=context, rows=rows),
                                              ensure_ascii=False, separators=(',', ':'))


def run_job(work, fid, budget, ask=None, gate=None):
    work = Path(work)
    target = work / 'translations' / fid
    base, payload = prepare_input(target)
    folder = target / 'isolated'
    decisions = []
    proofs = [native.proof(target / 'delivery/manifest.json'), native.proof(folder / 'input.json'),
              native.proof(folder / 'rules.txt'), base['source']]
    usages = []
    requests = []
    for rows, prompt in plan_requests(payload):
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        record = folder / 'requests' / (digest + '.json')
        try:
            if not record.exists():
                record.parent.mkdir(exist_ok=True)
                native.save(folder / 'attempt.json', dict(status='calling', video_fid=fid))
                gate(300) if gate else None
                started = datetime.now(timezone.utc).isoformat()
                if ask is None:
                    import translate_srt_batch as api
                    api.call_ollama(prompt, 'deepseek-flash', 'https://api.deepseek.com', timeout=300,
                                    response_format='json', response_path=record, thinking=True)
                else:
                    ask(prompt, record)
                native.save(folder / 'last_request.json',
                            dict(started_at_utc=started, receipt=native.proof(record)))
            receipt = native.read(record)
            if receipt['prompt_sha256'] != digest or receipt['content_sha256'] != \
                    hashlib.sha256(receipt['content'].encode()).hexdigest():
                raise ValueError('Isolated response cache changed')
            if receipt['model'] != 'deepseek-flash' or receipt['finish_reason'] != 'stop':
                raise ValueError('Isolated response did not finish normally')
            value = json.loads(receipt['content'])
            if value.get('content_hold') is True:
                # One flagged line no longer stops the film: keep only that batch blank.
                decisions.extend([row[0], '', 'unresolved', '', '单条内容保留'] for row in rows)
                proofs.append(native.proof(record))
                usages.append(receipt['usage'])
                requests.append(receipt.get('provider_request_id'))
                continue
            if set(value) != {'rows'}:
                raise ValueError('Unexpected isolated response fields')
            parsed = native.validate(value['rows'], [r[0] for r in rows])
        except BudgetExceeded:
            raise
        except Exception as exc:  # one bad batch must not stop the film
            decisions.extend([[row[0], '', 'unresolved', '', '批次未译'] for row in rows])
            native.save(folder / 'failed_batches.json',
                        dict(last_error=type(exc).__name__ + ': ' + str(exc)[:200],
                             at=datetime.now(timezone.utc).isoformat()))
            continue
        decisions.extend(parsed)
        proofs.append(native.proof(record))
        usages.append(receipt['usage'])
        requests.append(receipt.get('provider_request_id'))
    expected = [r[0] for r in payload['rows']]
    if [d[0] for d in decisions] != expected:
        raise ValueError('Isolated coverage changed')
    usage = dict(prompt_tokens=sum(x['prompt_tokens'] for x in usages),
                 completion_tokens=sum(x['completion_tokens'] for x in usages),
                 off_peak_cost_estimate_cny=sum(
                     (int(x.get('prompt_cache_hit_tokens', 0)) * .02
                      + (x['prompt_tokens'] - int(x.get('prompt_cache_hit_tokens', 0)))
                      + x['completion_tokens'] * 4) / 1e6 for x in usages),
                 request_count=len(usages), provider_request_ids=requests)
    native.save(folder / 'repair.json', dict(
        source=base['source'], ids=expected, rows=decisions, proofs=proofs,
        provider='deepseek-flash', api_usage=usage))
    native.save(folder / 'attempt.json', dict(status='completed', video_fid=fid, usage=usage))
    import subtitle_batch as batch
    result = batch.finish_local(work, fid)
    print(json.dumps(dict(event='isolated_completed', video_fid=fid, result=result, usage=usage),
                     ensure_ascii=True), flush=True)
    return result, usage


def apply_repair(target, base, max_deferred=0):
    target = Path(target)
    state = native.checked(target)
    path = target / 'isolated/repair.json'
    repair = native.read(path)
    if repair['source'] != state['source'] or repair['source'] != base['source']:
        raise ValueError('Isolated source differs')
    native.check_proofs(repair['proofs'])
    expected = [r['source']['id'] for r in base['ledger'] if r['status'] == 'content_isolated']
    if not expected and repair['ids']:
        return base  # already applied on a previous acceptance
    if repair['ids'] != expected:
        raise ValueError('Isolated cue set changed')
    native.validate(repair['rows'], expected)
    values = {r['source']['id']: [r['source']['id'], r['translation'], r['status'],
                                  r['uncertain_span'], r['reason']] for r in base['ledger']}
    for row in repair['rows']:
        values[row[0]] = row
    result = native.render_decisions(
        state, values, base['inputs'] + repair['proofs'] + [native.proof(path)],
        target / 'isolated/delivery', max_deferred=max_deferred,
        model=base['model'] + '+deepseek-isolated', effort=base['effort'])
    native.save(target / 'isolated/delivery/manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--budget-file', type=Path, required=True)
    parser.add_argument('--budget-cny', required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    from reprocess_subtitles import runner_lock
    with runner_lock(work / 'isolated.plan.json'):
        from nosub_pipeline import load_env
        import translate_srt_batch as api
        env = load_env()
        os.environ['DEEPSEEK_API_KEY'] = env['DEEPSEEK_API_KEY']
        budget = TranslationBudget(args.budget_file, args.budget_cny)
        api.API_BUDGET = budget
        api.API_MAX_TOKENS = MAX_OUTPUT_TOKENS
        api.API_REASONING_EFFORT = 'low'
        for fid in candidates(work):
            try:
                run_job(work, fid, budget)
            except Exception as exc:  # noqa: BLE001
                native.save(work / 'translations' / fid / 'isolated/attempt.json',
                            dict(status='failed', error_type=type(exc).__name__))
                print(json.dumps(dict(event='isolated_failed', video_fid=fid,
                                      error_type=type(exc).__name__)), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
