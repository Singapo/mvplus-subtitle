"""Apply one bounded native repair backed by new, immutable acoustic evidence.

This layer never changes the original source, native packets or API receipts.
It is opt-in through cue_repair/repair.json; it does not dispatch any model.
"""
import json
from pathlib import Path

import native_subtitles as native


def apply_repair(target, base, max_deferred=0):
    target=Path(target);state=native.checked(target)
    path=target/'cue_repair/repair.json';repair=native.read(path)
    base_path=target/('supplement/delivery/manifest.json' if (target/'supplement/repair.json').exists()
                      else 'delivery/manifest.json')
    if repair['source']!=state['source'] or repair['source']!=base['source']:
        raise ValueError('Cue repair source differs')
    if repair['base_manifest']!=native.proof(base_path) or native.read(base_path)!=base:
        raise ValueError('Cue repair base manifest differs')
    native.check_proofs(repair['proofs'])
    folder=target/'cue_repair'
    payload=native.read(folder/'input.json')
    evidence=native.read(folder/'acoustic/large-v3.json')
    if evidence['binding']['base_manifest']!=repair['base_manifest'] or evidence['binding']['source']!=base['source']:
        raise ValueError('Cue repair acoustic evidence differs')
    native.check_proofs([r['clip'] for r in evidence['checks']])
    expected=[r['source']['id'] for r in base['ledger'] if r['status']=='unresolved']
    originals=[[r['source']['id'],r['source']['text']] for r in base['ledger'] if r['status']=='unresolved']
    if not expected or payload['rows']!=originals or payload.get('evidence')!=evidence['checks']:
        raise ValueError('Cue repair can only replace current unresolved IDs')
    if {(r['cue_id'],r['window']) for r in evidence['checks']}!={(i,w) for i in expected for w in ('near','context')}:
        raise ValueError('Cue repair requires both acoustic windows for every unresolved cue')
    clocks={r['source']['id']:(r['source']['start_ms']/1000,r['source']['end_ms']/1000) for r in base['ledger']}
    if any((r['cue_start'],r['cue_end'])!=clocks[r['cue_id']] or
           not r['clip_start']<=r['cue_start']<r['cue_end']<=r['clip_end'] for r in evidence['checks']):
        raise ValueError('Cue repair acoustic timing differs from the original source')
    required=[folder/name for name in ('input.json','rules.txt','acoustic/large-v3.json',
                                      'app_server_turn.json','app_server_turn_start.json')]
    if any(native.proof(p) not in repair['proofs'] for p in required):
        raise ValueError('Cue repair requires complete native and acoustic receipts')
    turn=native.read(folder/'app_server_turn.json');start=native.read(folder/'app_server_turn_start.json')
    if (turn['turn']['status']!='completed' or turn['model']!=native.MODEL or turn['effort']!=native.EFFORT or
        turn['input']!=native.proof(folder/'input.json') or len(turn['final_messages'])!=1 or
        start['state']!='started' or start['thread_id']!=turn['thread_id'] or
        start['turn_id']!=turn['turn']['id'] or start['input']!=turn['input'] or
        start['model']!=native.MODEL or start['effort']!=native.EFFORT):
        raise ValueError('Cue repair native turn was not completed with the expected input')
    raw=json.loads(turn['final_messages'][0])
    if raw.get('content_hold') is not False or raw.get('rows')!=repair['rows']:
        raise ValueError('Cue repair must match the original native response exactly')
    native.validate(repair['rows'],expected)
    # A text-only repair cannot establish that an uncertain sound has no meaning.
    if any(r[2] in {'sound','filler'} for r in repair['rows']):
        raise ValueError('Acoustic text candidates cannot establish sound or filler')
    values={r['source']['id']:[r['source']['id'],r['translation'],r['status'],r['uncertain_span'],r['reason']]
            for r in base['ledger']}
    for row in repair['rows']:values[row[0]]=row
    result=native.render_decisions(state,values,base['inputs']+repair['proofs']+[native.proof(path)],
        folder/'delivery',max_deferred=max_deferred,model=base['model']+'+native-acoustic-cue-repair',
        effort=base['effort']+'+low')
    if 'api_usage' in base:result['api_usage']=base['api_usage']
    native.save(folder/'delivery/manifest.json',result)
    return result
