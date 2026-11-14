"""Translate previously unresolved cues using fresh two-window Whisper evidence.

The evidence is local (large-v3, near/context windows); the translation call uses
the ordinary budgeted DeepSeek route. Cues whose evidence is missing, conflicting
or off-target stay unresolved rather than being guessed.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import native_subtitles as native
from translation_budget import TranslationBudget, BudgetExceeded

RULES = (native.RULES + '\n下面每条都是此前听不清的对白。evidence 是同一段音频用 Whisper large-v3 '
         '两个窗口的识别结果，可能有幻觉、重复或邻句，不是人工听验。'
         '只依据与目标时间对应的词还原原文，再译成自然中文；证据不足、互相冲突或指向目标时间以外时，'
         '保持 unresolved，不要猜。status 只用 ok/partial/unresolved；partial 要指出仍存疑的原文片段。'
         '露骨或亲密表达如实、生动地翻译，不弱化也不添加。'
         '只输出JSON对象 {"rows":[["id","中文","status","存疑原文词段","简短原因"]]}。')
BATCH_SIZE = 4
MAX_OUTPUT_TOKENS = 16384


def candidates(work):
    work = Path(work)
    result = []
    for target in (work / 'translations').iterdir():
        evidence = target / 'acoustic_review/large-v3.json'
        if not evidence.exists() or (target / 'acoustic_review/repair.json').exists():
            continue
        if (target / 'content_hold.json').exists():
            continue
        base_path = base_manifest(target)
        try:
            base = native.read(base_path)
        except Exception:  # noqa: BLE001
            continue
        unresolved = [r for r in base['ledger'] if r['status'] == 'unresolved']
        if not unresolved:
            continue
        data = native.read(evidence)
        checked = {(r['cue_id'], r['window']) for r in data.get('checks', [])}
        if not all((r['source']['id'], w) in checked for r in unresolved for w in ('near', 'context')):
            continue
        result.append((len(unresolved), target.name))
    return [fid for _, fid in sorted(result)]


def base_manifest(target):
    for name in ('isolated/delivery/manifest.json', 'supplement/delivery/manifest.json',
                 'cue_repair/delivery/manifest.json', 'delivery/manifest.json'):
        path = Path(target) / name
        if path.exists():
            return path
    raise ValueError('No manifest for ' + Path(target).name)


def prepare_input(target):
    target = Path(target)
    base_path = base_manifest(target)
    base = native.read(base_path)
    native.check_proofs(base['inputs'] + [base['source'], base['output']])
    data = native.read(target / 'acoustic_review/large-v3.json')
    if data['binding']['base_manifest'] != native.proof(base_path):
        raise ValueError('Acoustic evidence belongs to a different base manifest')
    native.check_proofs([r['clip'] for r in data['checks']])
    evidence = {}
    for row in data['checks']:
        evidence.setdefault(row['cue_id'], {})[row['window']] = row['text']
    rows = [r for r in base['ledger'] if r['status'] == 'unresolved']
    ids = {r['source']['id'] for r in rows}
    positions = [i for i, r in enumerate(base['ledger']) if r['source']['id'] in ids]
    near = {j for i in positions for j in range(max(0, i - 2), min(len(base['ledger']), i + 3))}
    context = [[base['ledger'][i]['source']['id'], base['ledger'][i]['source']['text']]
               for i in sorted(near) if base['ledger'][i]['source']['id'] not in ids]
    payload = dict(rows=[[r['source']['id'], r['source']['text']] for r in rows],
                   evidence={r['source']['id']: evidence.get(r['source']['id'], {}) for r in rows},
                   context=context)
    folder = target / 'acoustic_review'
    path = folder / 'input.json'
    if path.exists() and native.read(path) != payload:
        raise ValueError('Acoustic input changed')
    if not path.exists():
        native.save(path, payload)
    rules = folder / 'rules.txt'
    if rules.exists() and rules.read_text('utf-8') != RULES:
        raise ValueError('Acoustic rules changed')
    if not rules.exists():
        rules.write_text(RULES, encoding='utf-8')
    return base, payload


def plan_requests(payload):
    for offset in range(0, len(payload['rows']), BATCH_SIZE):
        rows = payload['rows'][offset:offset + BATCH_SIZE]
        ids = [r[0] for r in rows]
        yield rows, RULES + '\n' + json.dumps(dict(
            context=payload['context'], evidence={i: payload['evidence'][i] for i in ids}, rows=rows),
            ensure_ascii=False, separators=(',', ':'))


def run_job(work, fid, budget, ask=None, gate=None):
    work = Path(work)
    target = work / 'translations' / fid
    base, payload = prepare_input(target)
    folder = target / 'acoustic_review'
    base_path = base_manifest(target)
    decisions, proofs = [], [native.proof(base_path), native.proof(folder / 'input.json'),
                             native.proof(folder / 'rules.txt'), base['source']]
    usages, requests = [], []
    for rows, prompt in plan_requests(payload):
        digest = hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        record = folder / 'requests' / (digest + '.json')
        try:
            if not record.exists():
                record.parent.mkdir(exist_ok=True)
                if ask is None:
                    import translate_srt_batch as api
                    api.call_ollama(prompt, 'deepseek-flash', 'https://api.deepseek.com', timeout=300,
                                    response_format='json', response_path=record, thinking=True)
                else:
                    ask(prompt, record)
            receipt = native.read(record)
            if receipt['prompt_sha256'] != digest:
                raise ValueError('Acoustic response cache changed')
            value = json.loads(receipt['content'])
            if set(value) != {'rows'}:
                raise ValueError('Unexpected acoustic response fields')
            parsed = native.validate(value['rows'], [r[0] for r in rows])
        except BudgetExceeded:
            raise
        except Exception:  # one bad batch keeps its cues unresolved
            decisions.extend([[r[0], '', 'unresolved', '', '证据未译'] for r in rows])
            continue
        decisions.extend(parsed)
        proofs.append(native.proof(record))
        usages.append(receipt['usage'])
        requests.append(receipt.get('provider_request_id'))
    expected = [r[0] for r in payload['rows']]
    if [d[0] for d in decisions] != expected:
        raise ValueError('Acoustic coverage changed')
    usage = dict(prompt_tokens=sum(x['prompt_tokens'] for x in usages),
                 completion_tokens=sum(x['completion_tokens'] for x in usages),
                 request_count=len(usages), provider_request_ids=requests)
    native.save(folder / 'repair.json', dict(
        source=base['source'], base_manifest=native.proof(base_path), ids=expected,
        rows=decisions, proofs=proofs, provider='deepseek-flash', api_usage=usage))
    import subtitle_batch as batch
    result = batch.finish_local(work, fid)
    print(json.dumps(dict(event='acoustic_completed', video_fid=fid, result=result,
                          usage=usage), ensure_ascii=True), flush=True)
    return result, usage


def apply_repair(target, base, max_deferred=0):
    target = Path(target)
    state = native.checked(target)
    path = target / 'acoustic_review/repair.json'
    repair = native.read(path)
    if repair['source'] != state['source'] or repair['source'] != base['source']:
        raise ValueError('Acoustic source differs')
    native.check_proofs(repair['proofs'])
    expected = [r['source']['id'] for r in base['ledger'] if r['status'] == 'unresolved']
    if not expected and repair['ids']:
        return base
    if repair['ids'] != expected:
        raise ValueError('Acoustic cue set changed')
    native.validate(repair['rows'], expected)
    values = {r['source']['id']: [r['source']['id'], r['translation'], r['status'],
                                  r['uncertain_span'], r['reason']] for r in base['ledger']}
    for row in repair['rows']:
        values[row[0]] = row
    result = native.render_decisions(
        state, values, base['inputs'] + repair['proofs'] + [native.proof(path)],
        target / 'acoustic_review/delivery', max_deferred=max_deferred,
        model=base['model'] + '+acoustic-evidence', effort=base['effort'])
    native.save(target / 'acoustic_review/delivery/manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--budget-file', type=Path, required=True)
    parser.add_argument('--budget-cny', required=True)
    args = parser.parse_args()
    work = args.work.resolve()
    from reprocess_subtitles import runner_lock
    with runner_lock(work / 'acoustic.plan.json'):
        from nosub_pipeline import load_env
        import translate_srt_batch as api
        os.environ['DEEPSEEK_API_KEY'] = load_env()['DEEPSEEK_API_KEY']
        budget = TranslationBudget(args.budget_file, args.budget_cny)
        api.API_BUDGET = budget
        api.API_MAX_TOKENS = MAX_OUTPUT_TOKENS
        api.API_REASONING_EFFORT = 'low'
        for fid in candidates(work):
            try:
                run_job(work, fid, budget)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps(dict(event='acoustic_failed', video_fid=fid,
                                      error_type=type(exc).__name__)), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
