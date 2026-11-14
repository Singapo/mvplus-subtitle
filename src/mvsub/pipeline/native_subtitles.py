"""Single-pass native translation: scripts own identity, packets and delivery.

No model, API or external agent runner is invoked by this module.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import subtitle_pipeline_core as core
import config

POLICY = 'native-single-pass-v1'
MODEL = 'gpt-6.1-sol'
EFFORT = 'low'
RULES = ('这是观看用对白字幕，目标是保留剧情、人物关系、意图、动作要求、转折和条件，不是逐音转写。'
         '原文对白译为自然简体中文，结合相邻句理解半句和语尾，不补剧情或未确认的人名。'
         'rows是目标[id,原文]，context仅供理解；evidence是局部识别候选，可能有错，不覆盖原文。'
         '候选若使用引用键，对应文本在shared中；不把相邻时段的话塞进目标字幕。'
         '每条只判断一次，输出JSON数组，每行[id,可靠中文,status,未确认原文词段,短原因]。'
         'status=ok/partial/sound/filler/unresolved/content_isolated。ok完整可译；partial保留可靠意思且指出存疑词段，'
         '不因一个名字或修饰词存疑就丢弃核心意思；sound仅明确纯声音；'
         'filler表示结合上下文确认没有新增对白信息的语气、重复强调或零碎尾音，可省略并留记录。'
         '短不等于无意义：否定、拒绝、制止和关键回答必须保留；不、别、停、作为回答的嗯/好不能因短而省。'
         '同一句内部可自然压缩重复音，不单独猜解每个孤立音节；不影响意思的语尾不用逐字对应。'
         'unresolved仅用于可能含重要对白但关键含义无法可靠辨认的情况；普通碎音不因此升级为剧情缺失。'
         '存疑说明只写记录列，字幕不放待核标记或占位内容。'
         'content_isolated仅标记明确无法提供详细翻译的受限条目，中文与原文词段留空，只写简短类别原因，不复述细节。'
         '成年人自愿情节的既有字幕按原意翻译，不能仅因露骨而隔离。'
         '其余可独立处理的普通对白继续翻译；不要因单个条目受限而停止整片，也不能把受限内容拆句续译。'
         '拒绝、否定或制止本身不等于内容受限，需结合语境判断；不确定的声音不据此隔离。'
         'ok/partial须非空单行中文，译文字符串不得包含换行；sound/filler/unresolved/content_isolated中文留空。除ok外须说明原因。'
         '不输出时码、hash或其他ID，不做第二遍审校。露骨或性描写用克制、非露骨的中文概括，保留剧情，不跳过、不隔离、不整片停止。')


def read(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def proof(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def save(path, data):
    core.save_json(Path(path), data)


def check_proofs(items):
    for item in items:
        if proof(item['path']) != item:
            raise ValueError('Input or receipt changed: ' + Path(item['path']).name)


def checked(work):
    work = Path(work)
    if (work/'content_hold.json').exists():
        raise ValueError('Content hold; no translation or publication')
    state = read(work/'job.json')
    check_proofs(state['inputs'])
    return state


def seconds(stamp):
    return core._stamp_ms(stamp)/1000


def invalid_candidate(text):
    clean=''.join(c for c in text if c.isalnum())
    return len(text)>500 or bool(len(clean)>35 and max(Counter(clean).values(),default=0)/len(clean)>.65)


def evidence_from_segments(entries, segments):
    """Each cue receives target-time words; shared text is never repeated per cue."""
    shared={};rows={};missing=[]
    for entry in entries:
        left,right=seconds(entry['start']),seconds(entry['end'])
        refs=[];observed=False
        for index,segment in enumerate(segments):
            if segment['start']>=right+.15 or segment['end']<=left-.15:
                continue
            text=segment['text'].strip()
            if invalid_candidate(text):
                continue
            focus=''.join(w['text'] for w in segment.get('words',[]) if w['start']<right+.15 and w['end']>left-.15)
            if not focus:
                continue
            observed=True
            clean=lambda text: ''.join(c for c in text if c.isalnum())
            if SequenceMatcher(None,clean(entry['text']),clean(focus)).ratio()>=.9:
                continue  # Matching independent words need no duplicate model input.
            key=f's{index}'
            shared[key]=text
            refs.append([key,focus])
        rows[entry['id']]=refs
        if not observed:missing.append(entry['id'])
    return rows,shared,missing


def recognize_evidence(source,work):
    """One independent local ASR pass before translation, covering the whole audio."""
    from asr_config import MODEL_HOME
    from audio_archive import digest
    from aligned_asr.utterance import model_identity
    from reprocess_subtitles import require_current_timing
    source,work=Path(source).resolve(),Path(work).resolve()
    report=require_current_timing(source)
    audio=source.parent/'full_audio.wav'
    if not audio.exists() or digest(audio)!=report['audio_sha1']:
        raise ValueError('Current full source audio required')
    model_path=MODEL_HOME/'models/whisper-large-v3'
    if not (model_path/'model.bin').exists():
        raise FileNotFoundError('Configured local independent ASR missing; no automatic download')
    folder=work/'source_evidence';folder.mkdir(parents=True,exist_ok=True)
    cache=folder/'local_asr.json'
    identity=dict(audio_sha1=report['audio_sha1'],model_identity=model_identity(model_path),
                  settings='ja-beam5-temperature0-no-context-no-vad-words-v1')
    if cache.exists():
        decoded=read(cache)
        if decoded.get('identity')!=identity:
            raise ValueError('Cached independent ASR belongs to different audio/model')
    else:
        handles=[];bins=[]
        root=config.repo_root()
        for base in (Path(sys.prefix)/'Lib/site-packages/nvidia',root/'lib/cuda/nvidia'):
            if base.is_dir():
                for package in base.iterdir():
                    target=package/'bin'
                    if target.is_dir():
                        bins.append(str(target))
                        if hasattr(os,'add_dll_directory'):handles.append(os.add_dll_directory(str(target)))
        os.environ['PATH']=os.pathsep.join(bins+[os.environ.get('PATH','')])
        from faster_whisper import WhisperModel
        model=WhisperModel(str(model_path),device='cuda',compute_type='float16')
        segments,_=model.transcribe(str(audio),language='ja',beam_size=5,temperature=0,
            word_timestamps=True,vad_filter=False,condition_on_previous_text=False,no_speech_threshold=.95)
        collected=[]
        for segment in segments:
            collected.append(dict(start=segment.start,end=segment.end,text=segment.text,
                words=[dict(text=w.word,start=w.start,end=w.end) for w in segment.words or []]))
            if len(collected)%200==0:
                print(json.dumps(dict(local_asr_segments=len(collected),position_seconds=round(segment.end))),flush=True)
        decoded=dict(identity=identity,segments=collected,human_audio_verified=False)
        save(cache,decoded)
    entries=core.parse_srt(source)
    rows,shared,missing=evidence_from_segments(entries,decoded['segments'])
    output=folder/'evidence.json'
    save(output,dict(source_sha256=proof(source)['sha256'],proofs=[proof(cache)],rows=rows,shared=shared,
        coverage=dict(cues=len(entries),with_independent_words=len(entries)-len(missing),missing_ids=missing),
        human_audio_verified=False,model_calls=0))
    return dict(output=str(output),cues=len(entries),with_independent_words=len(entries)-len(missing),
                missing_independent_words=len(missing),model_calls=0)


def validate(rows, ids):
    if not isinstance(rows, list) or [r[0] for r in rows if isinstance(r,list) and r] != ids:
        raise ValueError('Exact target ID order and coverage required')
    for row in rows:
        if len(row) != 5 or not all(isinstance(x,str) for x in row):
            raise ValueError('Expected five string columns')
        _, text, status, span, reason = row
        if status not in {'ok','partial','sound','filler','unresolved','content_isolated'}:
            raise ValueError('Unknown translation status')
        if status in {'ok','partial'}:
            if not text.strip() or '\n' in text or re.search(r'[\u3040-\u30ff\uac00-\ud7af]|\[原文待核\]|\[待核\]',text):
                raise ValueError('Invalid Chinese translation')
        elif text:
            raise ValueError('Do not mix uncertain draft text with an omission decision')
        if status != 'ok' and not reason.strip():
            raise ValueError('Missing reason')
        if status == 'partial' and not span.strip():
            raise ValueError('Partial translation must identify its uncertain source span')
    return rows


def prepare(source, work, seed=None, evidence=None):
    source, work = Path(source).resolve(), Path(work).resolve()
    if (work/'content_hold.json').exists():
        raise ValueError('Content hold; no model packets may be prepared')
    if (work/'job.json').exists():
        state = checked(work)
        if state['source'] != proof(source):
            raise ValueError('Resume source differs')
        if [proof(p) for p in (seed,evidence) if p] != state['optional_inputs']:
            raise ValueError('Resume seed/evidence differs')
        return state
    entries = core.parse_srt(source)
    if not entries:
        raise ValueError('Empty source')
    previous = -1
    for entry in entries:
        start,end = core._stamp_ms(entry['start']),core._stamp_ms(entry['end'])
        if not 0 <= start < end or start < previous:
            raise ValueError('Invalid or backward source time')
        previous = start
    source_proof = proof(source)
    inputs = [source_proof]
    timing = source.with_suffix('.alignment.json')
    if timing.exists():
        from reprocess_subtitles import require_current_timing
        report = require_current_timing(source)
        if any(core._stamp_ms(e['end']) > report['audio_seconds']*1000+1 for e in entries):
            raise ValueError('Source cue exceeds audio duration')
        inputs.append(proof(timing))
        if report.get('source_subset_review_sha1'):
            from source_subset import verify
            subset=verify(source.parent,report)
            inputs.extend([proof(source.parent/'source_subset_review.json'),*subset['inputs']])
    elif source.with_suffix('.word_clock.json').exists():
        from nosub_sources import validate_word_clock
        validate_word_clock(source)
        inputs.append(proof(source.with_suffix('.word_clock.json')))
    seeded = []
    if seed:
        data = read(seed)
        if data['source_sha256'] != source_proof['sha256']:
            raise ValueError('Seed source differs')
        if not data['proofs']:
            raise ValueError('Seed requires actual prior evidence')
        check_proofs(data['proofs'])
        seeded = validate(data['rows'], [r[0] for r in data['rows']])
        if len({r[0] for r in seeded}) != len(seeded):
            raise ValueError('Duplicate seed ID')
        if any(r[2] not in {'ok','partial','sound','filler'} for r in seeded):
            raise ValueError('Only previously accepted decisions may be reused')
        inputs.extend([proof(seed), *data['proofs']])
    known = {e['id'] for e in entries}
    if any(r[0] not in known for r in seeded):
        raise ValueError('Unknown seed ID')
    evidence_rows = {};shared = {}
    if evidence:
        data = read(evidence)
        if data['source_sha256'] != source_proof['sha256']:
            raise ValueError('Evidence source differs')
        check_proofs(data['proofs'])
        evidence_rows = data['rows']
        shared = data.get('shared',{})
        if set(evidence_rows)-known:
            raise ValueError('Evidence contains unknown source IDs')
        inputs.extend([proof(evidence), *data['proofs']])
    pending = [e for e in entries if e['id'] not in {r[0] for r in seeded}]
    # Existing local candidates may help meaningful source conflicts. A second
    # whole-film ASR pass is optional, never a mandatory translation gate.
    work.mkdir(parents=True,exist_ok=True)
    packets = []
    positions = {e['id']:i for i,e in enumerate(entries)}
    for offset in range(0,len(pending),320):
        targets = pending[offset:offset+320]
        ids = {e['id'] for e in targets}
        nearby = {i for e in targets for i in range(max(0,positions[e['id']]-1),min(len(entries),positions[e['id']]+2))}
        payload = dict(rows=[[e['id'],e['text']] for e in targets],
            context=[[entries[i]['id'],entries[i]['text']] for i in sorted(nearby) if entries[i]['id'] not in ids],
            evidence={e['id']:evidence_rows[e['id']] for e in targets if e['id'] in evidence_rows})
        if shared:
            keys={r[0] for e in targets for r in evidence_rows.get(e['id'],[])}
            payload['shared']={k:v for k,v in shared.items() if k in keys}
        folder = work/'packets'/f'{len(packets)+1:03}'
        folder.mkdir(parents=True,exist_ok=True)
        # Compact input: metadata, times and hashes remain in local control files.
        (folder/'input.json').write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
        control = dict(policy=POLICY,model=MODEL,effort=EFFORT,ids=[e['id'] for e in targets],input=proof(folder/'input.json'))
        save(folder/'control.json',control)
        inputs.extend([proof(folder/'input.json'),proof(folder/'control.json')])
        packets.append(str(folder))
    (work/'rules.txt').write_text(RULES,encoding='utf-8')
    state = dict(policy=POLICY,source=source_proof,entries=entries,seed=seeded,packets=packets,
        inputs=inputs+[proof(work/'rules.txt')],optional_inputs=[proof(p) for p in (seed,evidence) if p],
        model=MODEL,effort=EFFORT,content_priority='plot_and_dialogue_information',
        automatic_semantic_review=False,actual_token_usage=None)
    save(work/'job.json',state)
    return state


def accept(work, packet, agent):
    state = checked(work)
    folder = Path(work)/'packets'/packet
    if str(folder.resolve()) not in state['packets']:
        raise ValueError('Unknown packet')
    control = read(folder/'control.json')
    check_proofs([control['input']])
    validate(read(folder/'output.json'),control['ids'])
    if not agent or control['model'] != MODEL or control['effort'] != EFFORT:
        raise ValueError('Actual native agent and configured model required')
    receipt = dict(agent=agent,model=MODEL,effort=EFFORT,control=proof(folder/'control.json'),output=proof(folder/'output.json'))
    path = folder/'receipt.json'
    if path.exists() and read(path) != receipt:
        raise ValueError('Cannot overwrite a sealed result')
    save(path,receipt)
    return receipt


def update_unaccepted_rules(work):
    work=Path(work).resolve();state=checked(work)
    if any((Path(p)/'receipt.json').exists() for p in state['packets']):
        raise ValueError('Cannot change rules for a sealed result')
    path=work/'rules.txt';old=proof(path)
    path.write_text(RULES,encoding='utf-8')
    state['inputs']=[proof(path) if item==old else item for item in state['inputs']]
    save(work/'job.json',state)
    return state


def build(work, max_deferred=0, persist_manifest=True):
    if type(max_deferred) is not int or max_deferred < 0:
        raise ValueError('Invalid deferred cue allowance')
    work = Path(work)
    state = checked(work)
    values = {r[0]:r for r in state['seed']}
    receipts = []
    for name in state['packets']:
        folder = Path(name)
        receipt = read(folder/'receipt.json')
        check_proofs([receipt['control'],receipt['output']])
        control = read(folder/'control.json')
        if receipt['model'] != MODEL or receipt['effort'] != EFFORT:
            raise ValueError('Wrong model receipt')
        check_proofs([control['input']])
        for row in validate(read(folder/'output.json'),control['ids']):
            if row[0] in values:
                raise ValueError('Duplicate completed ID')
            values[row[0]] = row
        receipts.append(proof(folder/'receipt.json'))
    if set(values) != {e['id'] for e in state['entries']}:
        raise ValueError('Incomplete source coverage')
    return render_decisions(state,values,state['inputs']+receipts,work/'delivery',
                            max_deferred=max_deferred,persist_manifest=persist_manifest)


def render_decisions(state,values,inputs,delivery,max_deferred=0,persist_manifest=True,
                     model=MODEL,effort=EFFORT):
    """Render verified decisions; shared by native results and API supplements."""
    if type(max_deferred) is not int or max_deferred < 0:
        raise ValueError('Invalid deferred cue allowance')
    ids=[e['id'] for e in state['entries']]
    if set(values)!=set(ids):raise ValueError('Incomplete source coverage')
    validate([values[idx] for idx in ids],ids)
    mapping = []; ledger = []; output = []
    for entry in state['entries']:
        idx,text,status,span,reason = values[entry['id']]
        row = dict(source=entry,translation=text,status=status,uncertain_span=span,reason=reason)
        ledger.append(row)
        if status in {'ok','partial'}:
            mapping.append(row)
            output.append(f"{len(output)+1}\n{entry['start']} --> {entry['end']}\n{text}")
    counts = dict(Counter(r['status'] for r in ledger))
    delivery=Path(delivery);delivery.mkdir(parents=True,exist_ok=True)
    path = delivery/Path(state['source']['path']).name
    path.write_text('\n\n'.join(output)+'\n',encoding='utf-8')
    rendered = core.parse_srt(path) if output else []
    if len(rendered) != len(mapping) or any((a['start'],a['end'],a['text']) !=
        (b['source']['start'],b['source']['end'],b['translation']) for a,b in zip(rendered,mapping)):
        raise ValueError('Rendered subtitles differ from accepted source mapping')
    deferred_ids=[r['source']['id'] for r in ledger if r['status']=='unresolved']
    status=('needs_source_decision' if counts.get('unresolved') else
            'no_reliable_dialogue' if not output else 'ready')
    if output and deferred_ids and len(deferred_ids)<=max_deferred:
        status='ready_with_deferred'
    manifest = dict(policy=POLICY,counts=counts,source=state['source'],output=proof(path),ledger=ledger,
        inputs=inputs,model=model,effort=effort,
        status=status,deferred_ids=deferred_ids,max_deferred=max_deferred,
        human_audio_verified=False,automatic_semantic_review=False,actual_token_usage=None)
    if persist_manifest:save(delivery/'manifest.json',manifest)
    return manifest


def publish(work, plan):
    """Publish only a ready result tied to one frozen source/video identity."""
    work,plan=Path(work).resolve(),Path(plan).resolve()
    manifest=build(work)
    if manifest['status']!='ready':
        raise ValueError('Important source meaning unresolved; publication not allowed')
    output=Path(manifest['output']['path'])
    rows=[r for r in read(plan)['items'] if Path(r['video']['n']).stem==output.stem]
    if len(rows)!=1:
        raise ValueError('Subtitle must uniquely match the frozen video plan')
    row=rows[0];video=row['video'];original=row['subtitle']
    if Path(manifest['source']['path']).parent.name not in {str(video['fid']),str(original['fid'])}:
        raise ValueError('Source extraction identity differs from publication plan')
    journal_path=work/'publication.json'
    journal=read(journal_path) if journal_path.exists() else {}
    identity=dict(plan=proof(plan),source=manifest['source'])
    if journal.get('identity',identity)!=identity:
        raise ValueError('Publication identity changed')
    from cloud_common import Cloud115
    from nosub_pipeline import load_env
    cloud=Cloud115(load_env()['P115_COOKIES'])
    actual=next((r for r in cloud.list_dir(row['cid']) if str(r.get('fid'))==str(video['fid'])),None)
    if actual is None or actual['n']!=video['n'] or not video.get('sha') or actual.get('sha')!=video['sha']:
        raise ValueError('Cloud source video changed')
    journal['identity']=identity
    if original.get('placeholder'):
        cloud.publish_missing_subtitle(video,output,journal,lambda:save(journal_path,journal))
    else:
        cloud.replace_subtitle(original,output,work/'rollback'/original['n'],journal,
                               lambda:save(journal_path,journal),target_name=output.name)
    if journal.get('stage')!='complete':
        raise ValueError('Publication transaction incomplete; keep all local files')
    return dict(stage=journal['stage'],cloud_fid=journal['stage_fid'],sha1=journal['sha1'])


def main():
    p=argparse.ArgumentParser(__doc__);sub=p.add_subparsers(dest='action',required=True)
    a=sub.add_parser('prepare');a.add_argument('--source',type=Path,required=True);a.add_argument('--work',type=Path,required=True);a.add_argument('--seed',type=Path);a.add_argument('--evidence',type=Path)
    a=sub.add_parser('source');a.add_argument('--source',type=Path,required=True);a.add_argument('--work',type=Path,required=True)
    a=sub.add_parser('accept');a.add_argument('--work',type=Path,required=True);a.add_argument('--packet',required=True);a.add_argument('--agent',required=True)
    a=sub.add_parser('publish');a.add_argument('--work',type=Path,required=True);a.add_argument('--plan',type=Path,required=True)
    for name in ('build','status'):
        a=sub.add_parser(name);a.add_argument('--work',type=Path,required=True)
    a=p.parse_args()
    if a.action=='source':
        out=recognize_evidence(a.source,a.work)
    elif a.action=='prepare':
        s=prepare(a.source,a.work,a.seed,a.evidence);out=dict(total=len(s['entries']),reused=len(s['seed']),packets=s['packets'])
    elif a.action=='accept':
        accept(a.work,a.packet,a.agent);out=dict(accepted=a.packet)
    elif a.action=='build':
        s=build(a.work);out=dict(status=s['status'],counts=s['counts'],output=s['output']['path'])
    elif a.action=='publish':
        out=publish(a.work,a.plan)
    else:
        s=checked(a.work);out=dict(total=len(s['entries']),reused=len(s['seed']),packets=len(s['packets']),sealed=sum((Path(n)/'receipt.json').exists() for n in s['packets']))
    print(json.dumps(out,ensure_ascii=True))


if __name__=='__main__':
    main()
