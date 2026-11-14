"""One-pass ordinary unresolved-cue supplements; no native/GPT polling.

Isolated cues never leave the machine: they are excluded from both the target
rows and the neighbouring context, so a film that isolates some cues can still
have its ordinary unresolved cues supplemented by the fallback provider."""
import argparse
from datetime import datetime,timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path

import native_subtitles as native
from deepseek_offpeak import wait_offpeak
from translation_budget import TranslationBudget,BudgetExceeded

RULES=(native.RULES+'\n你只补译上次未确定的普通对白，不能重译其他ID。context仅为相邻原文。'
       '结合上下文准确翻译，不臆造或补写未确认内容；内容边界沿用上述规则。'
       '只有确实听不清时该条才留空，不要编造。只输出JSON对象'
       '{"rows":[["id","中文","status","存疑原文词段","简短原因"]]}。')
BATCH_SIZE=4
MAX_OUTPUT_TOKENS=16384
PILOT_SETTINGS=dict(batch_size=BATCH_SIZE,max_output_tokens=MAX_OUTPUT_TOKENS,thinking='enabled')


class TranslationQuotaPaused(RuntimeError):pass


def require_translation_quota(work):
    path=Path(work)/'quota_policy.json'
    policy=native.read(path) if path.exists() else {}
    if not policy.get('pause_supplement_with_gpt',False):return
    pause=Path(work)/'user_pause.json'
    user_paused=pause.exists() and native.read(pause).get('do_not_resume_automatically',False)
    cutoff=policy.get('translation_stop_at_remaining_percent',policy.get('stop_at_remaining_percent',5))
    if (user_paused or policy.get('state')!='running' or
            policy.get('last_observed_remaining_percent',0)<=cutoff):
        raise TranslationQuotaPaused('Extraction only; do not start supplement requests')


def candidates(work):
    """Never transfer isolated content or minor holds to another provider."""
    work=Path(work);policy= native.read(work/'delivery_policy.json')
    limit=0 if policy.get('supplement_small_deferred') else policy.get('max_deferred_cues',0)
    result=[]
    for path in (work/'translations').glob('*/delivery/manifest.json'):
        target=path.parent.parent
        if (target/'content_hold.json').exists():continue
        from subtitle_version_reuse import for_fid
        if for_fid(work,target.name)['action']!='produce':continue
        manifest=native.read(path)
        if manifest['counts'].get('unresolved',0)<=limit:continue
        if (target/'supplement/repair.json').exists():continue
        attempt=target/'supplement/attempt.json'
        if attempt.exists() and native.read(attempt).get('status')=='failed_no_retry':continue
        result.append((manifest['counts']['unresolved'],target.name))
    return [fid for _,fid in sorted(result)]


def prepare_input(target):
    target=Path(target);native.checked(target)
    manifest_path=target/'delivery/manifest.json';manifest=native.read(manifest_path)
    native.check_proofs(manifest['inputs']+[manifest['output']])
    rows=manifest['ledger'];positions=[i for i,r in enumerate(rows) if r['status']=='unresolved']
    near={j for i in positions for j in range(max(0,i-2),min(len(rows),i+3))}
    # Isolated cues never leave the machine, not even as neighbouring context.
    context=[i for i in sorted(near-set(positions)) if rows[i]['status']!='content_isolated']
    payload=dict(rows=[[rows[i]['source']['id'],rows[i]['source']['text']] for i in positions],
                 context=[[rows[i]['source']['id'],rows[i]['source']['text']] for i in context])
    policy=native.read(target.parent.parent/'delivery_policy.json')
    if policy.get('translation_context')=='fresh_packet':
        payload['context_by_id']={rows[i]['source']['id']:[
            [rows[j]['source']['id'],rows[j]['source']['text']]
            for j in range(max(0,i-2),min(len(rows),i+3))
            if j not in positions and rows[j]['status']!='content_isolated'] for i in positions}
    folder=target/'supplement';folder.mkdir(exist_ok=True)
    path=folder/'input.json'
    if path.exists() and native.read(path)!=payload:raise ValueError('Supplement input changed')
    if not path.exists():native.save(path,payload)
    rules=folder/'rules.txt'
    if rules.exists() and rules.read_text('utf-8')!=RULES:raise ValueError('Supplement rules changed')
    if not rules.exists():rules.write_text(RULES,encoding='utf-8')
    return manifest,payload


def parse_rows(content,ids):
    value=json.loads(content)
    if value.get('content_hold') is True:return None
    if set(value)!={'rows'}:raise ValueError('Unexpected supplement response fields')
    return native.validate(value['rows'],ids)


def parse_rows_with_deferred_span(content,ids,originals):
    """Keep an original unresolved cue when a partial response lacks evidence.

    Do not invent an uncertain span or pay to retry the response. All other
    contract failures, including coverage changes, remain blocking errors.
    """
    missing_span='Partial translation must identify its uncertain source span'
    try:
        return parse_rows(content,ids),[]
    except ValueError as exc:
        if str(exc)!=missing_span:raise
    rows=json.loads(content)['rows'];parsed=[];deferred=[]
    for row in rows:
        try:parsed.extend(native.validate([row],[row[0]]))
        except ValueError as exc:
            if str(exc)!=missing_span:raise
            original=originals[row[0]]
            if original[2]!='unresolved':raise ValueError('Only unresolved cues may be deferred')
            parsed.extend(native.validate([original],[row[0]]))
            deferred.append(dict(id=row[0],reason=missing_span))
    return native.validate(parsed,ids),deferred


def apply_repair(target,base,max_deferred=0):
    target=Path(target);state=native.checked(target)
    path=target/'supplement/repair.json';repair=native.read(path)
    if repair['source']!=state['source']:raise ValueError('Supplement source differs')
    if repair['native_manifest']!=native.proof(target/'delivery/manifest.json'):
        raise ValueError('Supplement native manifest differs')
    native.check_proofs(repair['proofs'])
    expected=[r['source']['id'] for r in base['ledger'] if r['status']=='unresolved']
    native.validate(repair['rows'],expected)
    values={r['source']['id']:[r['source']['id'],r['translation'],r['status'],r['uncertain_span'],r['reason']]
            for r in base['ledger']}
    for row in repair['rows']:values[row[0]]=row
    result=native.render_decisions(state,values,base['inputs']+repair['proofs']+[native.proof(path)],
        target/'supplement/delivery',max_deferred=max_deferred,
        model='gpt-6.1-sol+deepseek-flash',effort='native_low+api_thinking')
    result['api_usage']=repair['api_usage']
    native.save(target/'supplement/delivery/manifest.json',result)
    return result


def cost_estimate(usage):
    prompt=usage['prompt_tokens'];output=usage['completion_tokens']
    hit=usage.get('prompt_cache_hit_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens',0))
    if type(hit) is not int or not 0<=hit<=prompt:raise ValueError('Invalid cache usage')
    return float((Decimal(hit)*Decimal('.02')+Decimal(prompt-hit)+Decimal(output)*4)/Decimal(1_000_000))


def configure_timing(work):
    off_peak_only=native.read(Path(work)/'delivery_policy.json').get('deepseek_off_peak_only',True)
    os.environ['DEEPSEEK_OFF_PEAK_ONLY']='1' if off_peak_only else '0'
    return wait_offpeak if off_peak_only else lambda timeout:None


def plan_requests(payload):
    for offset in range(0,len(payload['rows']),BATCH_SIZE):
        rows=payload['rows'][offset:offset+BATCH_SIZE]
        # Only neighbouring context: repeating every other target row made cost scale with film size.
        context=payload['context']
        if 'context_by_id' in payload:
            nearby={r[0]:r for target in rows for r in payload['context_by_id'][target[0]]}
            context=list(nearby.values())
        yield rows,RULES+'\n'+json.dumps(dict(context=context,rows=rows),ensure_ascii=False,separators=(',',':'))


def run_job(work,fid,budget,ask=None,gate=wait_offpeak):
    work=Path(work);target=work/'translations'/fid
    from subtitle_version_reuse import for_fid
    if for_fid(work,fid)['action']!='produce':
        raise TranslationQuotaPaused('Alternate version waits for verified reuse; no supplement request')
    base,payload=prepare_input(target);folder=target/'supplement'
    decisions=[];proofs=[native.proof(target/'delivery/manifest.json'),native.proof(folder/'input.json'),
                         native.proof(folder/'rules.txt'),base['source']]
    usages=[];requests=[];validation_deferred=[]
    originals={r['source']['id']:[r['source']['id'],r['translation'],r['status'],r['uncertain_span'],r['reason']]
               for r in base['ledger'] if r['status']=='unresolved'}
    for rows,prompt in plan_requests(payload):
        digest=hashlib.sha256(prompt.encode('utf-8')).hexdigest()
        record=folder/'requests'/(digest+'.json')
        if not record.exists():
            require_translation_quota(work)
            native.save(folder/'attempt.json',dict(status='waiting_offpeak',video_fid=fid))
            gate(300)
            require_translation_quota(work)
            if for_fid(work,fid)['action']!='produce':
                raise TranslationQuotaPaused('Alternate version now waits for verified reuse')
            started=datetime.now(timezone.utc).isoformat()
            if ask is None:
                import translate_srt_batch as api
                api.call_ollama(prompt,'deepseek-flash','https://api.deepseek.com',timeout=300,
                                response_format='json',response_path=record,thinking=True)
            else:ask(prompt,record)
            receipt=native.read(record)
            native.save(folder/'last_request.json',dict(started_at_utc=started,receipt=native.proof(record)))
        receipt=native.read(record)
        if receipt['prompt_sha256']!=digest or receipt['content_sha256']!=hashlib.sha256(receipt['content'].encode()).hexdigest():
            raise ValueError('Supplement response cache changed')
        if receipt['model']!='deepseek-flash' or receipt['finish_reason']!='stop' or receipt.get('response_format')!='json':
            raise ValueError('Supplement response did not finish normally')
        if not isinstance(receipt.get('provider_request_id'),str) or not receipt['provider_request_id']:
            raise ValueError('Supplement requires a real provider request ID')
        parsed,deferred=parse_rows_with_deferred_span(receipt['content'],[r[0] for r in rows],originals)
        if parsed is None:
            # One flagged line no longer stops the film: keep only that batch blank.
            decisions.extend([[row[0],'','unresolved','','单条内容保留'] for row in rows])
            proofs.append(native.proof(record));usages.append(receipt['usage'])
            requests.append(receipt.get('provider_request_id'))
            continue
        decisions.extend(parsed);proofs.append(native.proof(record));usages.append(receipt['usage'])
        requests.append(receipt.get('provider_request_id'))
        validation_deferred.extend(dict(item,receipt=native.proof(record)) for item in deferred)
    usage=dict(prompt_tokens=sum(x['prompt_tokens'] for x in usages),
               completion_tokens=sum(x['completion_tokens'] for x in usages),
               off_peak_cost_estimate_cny=sum(cost_estimate(x) for x in usages),
               provider_request_ids=requests,request_count=len(usages),
               peak_cost_upper_estimate_cny=sum((x['prompt_tokens']*2+x['completion_tokens']*8)/1_000_000 for x in usages))
    native.save(folder/'repair.json',dict(source=base['source'],native_manifest=native.proof(target/'delivery/manifest.json'),
        rows=decisions,proofs=proofs,provider='deepseek-flash',api_usage=usage,
        validation_deferred=validation_deferred))
    native.save(folder/'attempt.json',dict(status='completed',video_fid=fid,usage=usage,
                                         validation_deferred=validation_deferred))
    import subtitle_batch as batch
    result=batch.finish_local(work,fid)
    print(json.dumps(dict(event='supplement_completed',video_fid=fid,result=result,usage=usage),ensure_ascii=True),flush=True)
    return result,usage


class ChangeEvent:
    """Windows directory notifications: no timer and no model polling."""
    def __init__(self,work):
        import ctypes
        from ctypes import wintypes
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.FindFirstChangeNotificationW.argtypes=[wintypes.LPCWSTR,wintypes.BOOL,wintypes.DWORD]
        self.api.FindFirstChangeNotificationW.restype=wintypes.HANDLE
        self.api.FindNextChangeNotification.argtypes=[wintypes.HANDLE]
        self.api.FindCloseChangeNotification.argtypes=[wintypes.HANDLE]
        self.api.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        self.handle=self.api.FindFirstChangeNotificationW(str(Path(work).resolve()),False,0x1|0x8|0x10)
        if self.handle in (None,ctypes.c_void_p(-1).value):raise OSError('Could not register completion event watcher')
    def wait(self):
        if self.api.WaitForSingleObject(self.handle,0xFFFFFFFF)!=0:raise OSError('Completion event wait failed')
        if not self.api.FindNextChangeNotification(self.handle):raise OSError('Completion event rearm failed')
    def close(self):self.api.FindCloseChangeNotification(self.handle)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--budget-file',type=Path,required=True)
    parser.add_argument('--budget-cny',required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--scope')
    parser.add_argument('--scope-budget-cny')
    parser.add_argument('--pilot-max-cny',type=float,default=.25)
    args=parser.parse_args();work=args.work.resolve()
    from reprocess_subtitles import runner_lock
    with runner_lock(work/'supplement.plan.json'):
        from nosub_pipeline import load_env
        import translate_srt_batch as api
        env=load_env();os.environ['DEEPSEEK_API_KEY']=env['DEEPSEEK_API_KEY']
        gate=configure_timing(work)
        budget=TranslationBudget(args.budget_file,args.budget_cny,scope=args.scope,scope_limit_cny=args.scope_budget_cny)
        api.API_BUDGET=budget;api.API_MAX_TOKENS=MAX_OUTPUT_TOKENS;api.API_REASONING_EFFORT='low'
        watcher=ChangeEvent(work) if args.watch else None
        pilot=work/'deepseek_pilot.json'
        previous=native.read(pilot) if pilot.exists() else None
        pilot_passed=bool(previous and previous.get('settings')==PILOT_SETTINGS and previous.get('passed'))
        try:
            while True:
                import subtitle_batch as batch
                batch.refresh_restricted_index(work)
                for fid in candidates(work):
                    try:
                        require_translation_quota(work)
                        result,usage=run_job(work,fid,budget,gate=gate)
                        if not pilot_passed:
                            passed=usage['off_peak_cost_estimate_cny']<=args.pilot_max_cny
                            history=(previous.get('history',[])+[{k:v for k,v in previous.items() if k!='history'}]) if previous else []
                            native.save(pilot,dict(video_fid=fid,usage=usage,budget=budget.summary(),passed=passed,
                                                  settings=PILOT_SETTINGS,history=history))
                            pilot_passed=passed
                            if not passed:budget.halt('Pilot exceeded cost allowance; no batch requests')
                        if budget.summary().get('halt_reason'):return 2
                    except TranslationQuotaPaused:
                        break  # Quota/file events resume the watcher; no failed-attempt marker.
                    except BudgetExceeded:
                        print(json.dumps(dict(event='supplement_budget_stopped',budget=budget.summary())),flush=True);return 2
                    except Exception as exc:
                        native.save(work/'translations'/fid/'supplement/attempt.json',dict(status='failed_no_retry',error_type=type(exc).__name__))
                        print(json.dumps(dict(event='supplement_failed',video_fid=fid,error_type=type(exc).__name__)),flush=True)
                        if not pilot_passed or budget.summary().get('halt_reason'):return 2
                if watcher is None:return 0
                watcher.wait()
        finally:
            if watcher:watcher.close()


if __name__=='__main__':raise SystemExit(main())
