"""Apply an explicit native content review bound to new original-scene context.

Original source, packet decisions and receipts remain immutable. This module
never dispatches a model or sends isolated material to a fallback provider.
"""
import json
from pathlib import Path

import native_subtitles as native


def review_input(base):
    ledger=base['ledger']
    positions=[i for i,r in enumerate(ledger) if r['status']=='content_isolated']
    nearby={j for i in positions for j in range(max(0,i-2),min(len(ledger),i+3))}
    return dict(rows=[[ledger[i]['source']['id'],ledger[i]['source']['text']] for i in positions],
        context=[[ledger[i]['source']['id'],ledger[i]['source']['text']] for i in sorted(nearby-set(positions))],
        scene_context=[[r['source']['id'],r['source']['text']] for r in ledger[:25]])


def apply_review(target,base,max_deferred=0):
    target=Path(target);state=native.checked(target);folder=target/'content_review'
    path=folder/'review.json';review=native.read(path)
    native.check_proofs([review['scope']]+review['proofs'])
    if review['scope']!=native.proof(folder/'scope.json'):
        raise ValueError('Content review scope differs')
    scope=native.read(folder/'scope.json');original_path=target/'delivery/manifest.json'
    if scope['base_manifest']!=native.proof(original_path):
        raise ValueError('Content review base manifest changed')
    original=native.read(original_path)
    if (scope['source']!=state['source'] or scope['source']!=base['source'] or
        any(base[k]!=original[k] for k in ('source','ledger','inputs','model','effort'))):
        raise ValueError('Content review can only amend the original native decisions')
    native.check_proofs(base['inputs']+[base['source']])
    expected=[r['source']['id'] for r in base['ledger'] if r['status']=='content_isolated']
    if (not expected or scope['target_ids']!=expected or not scope['new_evidence'] or
        scope['no_other_provider'] is not True or scope['no_automatic_retry'] is not True):
        raise ValueError('Content review requires explicit isolated IDs and new evidence')
    required=[folder/name for name in ('input.json','rules.txt','app_server_turn.json','app_server_turn_start.json')]
    if any(native.proof(p) not in review['proofs'] for p in required):
        raise ValueError('Content review requires complete native receipts')
    if native.read(folder/'input.json')!=review_input(base):
        raise ValueError('Content review scene and neighboring context differ from original source')
    turn=native.read(folder/'app_server_turn.json');start=native.read(folder/'app_server_turn_start.json')
    if (turn['turn']['status']!='completed' or turn['model']!=native.MODEL or turn['effort']!=native.EFFORT or
        turn['input']!=native.proof(folder/'input.json') or len(turn['final_messages'])!=1 or
        start['state']!='started' or start['thread_id']!=turn['thread_id'] or
        start['turn_id']!=turn['turn']['id'] or start['input']!=turn['input'] or
        start['model']!=native.MODEL or start['effort']!=native.EFFORT):
        raise ValueError('Content review native turn is incomplete or has different input')
    raw=json.loads(turn['final_messages'][0])
    if raw.get('content_hold') is not False or raw.get('rows')!=review['rows']:
        raise ValueError('Content review must match the original native response exactly')
    native.validate(review['rows'],expected)
    values={r['source']['id']:[r['source']['id'],r['translation'],r['status'],r['uncertain_span'],r['reason']]
            for r in base['ledger']}
    # Uncertain or text-only sound/filler decisions cannot erase an isolation.
    retained=[]
    for row in review['rows']:
        if row[2] in {'ok','partial','content_isolated'}:values[row[0]]=row
        else:retained.append(row[0])
    result=native.render_decisions(state,values,
        base['inputs']+review['proofs']+[review['scope'],native.proof(path)],folder/'delivery',
        max_deferred=max_deferred,model=base['model']+'+native-content-context-review',effort=base['effort'])
    result['content_review']=dict(review=native.proof(path),original_manifest=scope['base_manifest'],
                                  retained_isolation_ids=retained)
    native.save(folder/'delivery/manifest.json',result)
    return result
