"""Text-preserving utterance clocks and explicit quality failures."""
import math
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from .timing import normalize

POLICY='utterance_clock_v2'
CONTENT_POLICY='preserve_dialogue_report_disagreement_omit_clear_sounds_v1'


def pipeline_revision():
    here=Path(__file__).resolve().parent
    paths=[here/name for name in ('recognize.py','align.py','caption.py','repair.py','utterance.py','verify_content.py','vad_evidence.py','acoustic.py','repair_recognition.py','recovery.py','recover_words.py','verify_events.py','clock_order.py')]
    paths.append(here.parent/'extract_aligned_srt.py')
    paths.extend([here.parent/'asr_config.py',here/'timing.py'])
    digest=hashlib.sha1()
    for path in paths:digest.update(path.name.encode());digest.update(path.read_bytes())
    from asr_config import ASR_MODEL, ALIGN_MODEL, MODEL_HOME
    for model in (ASR_MODEL,ALIGN_MODEL,MODEL_HOME/'models/qwen-asr-verifier',MODEL_HOME/'models/whisper-large-v3',MODEL_HOME/'models/audio-event-verifier'):
        digest.update(model_identity(model).encode())
    return digest.hexdigest().upper()


def model_identity(model):
    digest=hashlib.sha1(str(model.resolve()).encode('utf-8'))
    for path in sorted(model.glob('*')):
        if path.is_file() and path.suffix in {'.json','.bin','.safetensors','.jinja'}:
            digest.update(path.name.encode());digest.update(str((path.stat().st_size,path.stat().st_mtime_ns)).encode())
            if path.suffix in {'.json','.jinja'}:digest.update(path.read_bytes())
    return digest.hexdigest().upper()


def recognizer_revision():
    from asr_config import ASR_MODEL
    here=Path(__file__).resolve().parent
    digest=hashlib.sha1((here/'recognize.py').read_bytes())
    digest.update((here.parent/'asr_config.py').read_bytes());digest.update(model_identity(ASR_MODEL).encode())
    return digest.hexdigest().upper()


def review_digest(job):
    path=Path(job)/'manual_audio_review.json'
    data={'non_speech_regions':[],'utterance_confirmations':[]}
    if path.exists():data.update(json.loads(path.read_text('utf-8')))
    return hashlib.sha1(json.dumps(data,sort_keys=True,ensure_ascii=False).encode('utf-8')).hexdigest().upper()
# Do not consume the leading ん of the next sound stem. Possessive suffixes
# otherwise turn e.g. んぐっんぐっ into a false lexical repetition.
SOUND_STEM=r'(?:ちゅぱ|ちゅぽ|ちゅぷ|んちゅ|んじゅ|ちゅ|じゅぽ|じゅる|ぢゅぽ|ぢゅる|じゅ|ぢゅ|ぐちゅ|くちゅ|れろ|ぺろ|ごきゅ|ごく|ひっく|ひゃ|ひぅ|にゃ|ふ{2,}+|く{2,}+|[はふくぷ]っ|くん|むちゅ|むふ|あふ|はふ|あむ|はむ|んむ|んぐ|すん|んふ)'
NEXT_SOUND=r'(?!ん(?:ぐ|ふ|む|ちゅ|じゅ))'
SOUND_PATTERN=r'(?:'+SOUND_STEM+r'(?:'+NEXT_SOUND+r'[るろぷぱぽっんぁあぅうー])*+|[はふすくぷむ][ぁあぅうー]++|(?:'+NEXT_SOUND+r'[あぁいぃうぅえぇおぉんっー])++)++'
NONVERBAL=re.compile(SOUND_PATTERN+'$')
SOUND_RUN=re.compile(SOUND_PATTERN)
GRAMMAR={'接頭辞','接尾辞','助詞','助動詞','動詞','形容詞'}

# Short replies and lexical readings must not disappear merely because their
# kana also match a phonetic sound pattern. Ambiguous cases stay for review.
DIALOGUE_INTERJECTIONS={'うん','うんうん','ううん','はい','いいえ','いい','いう',
    'あ','ああ','ん','え','ええ','お','おお','う','うう','ふん','ふうん','ふーん',
    'は','はあ','ふう','すう','くう','くん'}


def non_dialogue_sound(unit):
    """Filter only explicit sound units, never lexical or ambiguous replies."""
    text=unit.get('text','');normalized=normalize(text)
    if unit.get('kind')!='sound' or not normalized or normalized in DIALOGUE_INTERJECTIONS:
        return False
    if len(normalized)<=3 and re.search('[?？]',text):return False
    return bool(NONVERBAL.fullmatch(normalized))


def sound_omission_indices(units,review=None):
    """Preserve unit indexes used by alignment/recovery evidence and clocks."""
    omitted=set()
    for index,unit in enumerate(units):
        if not non_dialogue_sound(unit):continue
        confirmed=human_confirmation(unit,review or {})
        if confirmed and normalize(confirmed['text'])!=normalize(unit['text']):continue
        omitted.add(index)
    return omitted


def sound_token(text,pos=None):
    n=normalize(text)
    if not n or not NONVERBAL.fullmatch(n) or pos in GRAMMAR:return False
    return len(n)>1 or n in {'っ','ぁ','ぅ'} or pos=='感動詞'


def segment_words(block):
    """Repair decoder segment ownership only with exact whole-word text proof."""
    segments=block.get('segments',[])
    if all(normalize(''.join(w['text'] for w in s.get('words',[])))==normalize(s['text']) for s in segments):
        return segments
    words=[w for s in segments for w in s.get('words',[]) if normalize(w['text'])]
    expected=''.join(normalize(s['text']) for s in segments)
    if normalize(''.join(w['text'] for w in words))!=expected:return segments
    ends=[];pos=0
    for w in words:
        pos+=len(normalize(w['text']));ends.append(pos)
    boundaries=[];pos=0
    for s in segments:
        pos+=len(normalize(s['text']));boundaries.append(pos)
    if any(p and p not in ends for p in boundaries):return segments
    result=[];begin=0;wi=0
    for s,stop in zip(segments,boundaries):
        owned=[]
        while wi<len(words) and ends[wi]<=stop:
            owned.append(dict(words[wi]));wi+=1
        result.append(dict(s,words=owned));begin=stop
    return result


def make_units(block,tagger):
    units=[];issues=[]
    for si,seg in enumerate(segment_words(block)):
        native=[dict(start=block['start']+w['start'],end=block['start']+w['end'],text=w['text'],probability=w.get('probability',0))
                for w in seg.get('words',[]) if normalize(w['text'])]
        expected=normalize(seg['text'])
        if normalize(''.join(w['text'] for w in native))!=expected:
            issues.append(dict(type='native_text_mismatch',segment=si));continue
        if not native:continue
        tags=[];pos=0
        if NONVERBAL.fullmatch(expected):
            tags=[(0,len(expected),True)];pos=len(expected)
        else:
            tagged=tagger(seg['text'])
            runs=[(m.start(),m.end()) for m in SOUND_RUN.finditer(expected)
                  if len(m.group())>=2 and any(c in m.group() for c in 'ぁぃぅぇぉっー')]
            for t,p in zip(tagged.words,tagged.postags):
                n=normalize(t)
                if n:
                    explicit=any(a<=pos and pos+len(n)<=b for a,b in runs)
                    if p=='接頭辞':explicit=explicit and any(a<=pos and pos+len(n)<b for a,b in runs)
                    if p=='接尾辞':explicit=explicit and any(a<pos and pos+len(n)<=b for a,b in runs)
                    if p=='形容詞':explicit=False
                    if p=='動詞' and n not in {'は','ふ','す','く','ぷ'}:explicit=False
                    tags.append((pos,pos+len(n),explicit or sound_token(n,p)));pos+=len(n)
        if pos!=len(expected):
            issues.append(dict(type='tagger_text_mismatch',segment=si));continue
        current=[];kind=None;pos=0
        def flush():
            if current:
                units.append(dict(segment=si,kind=kind,words=list(current),
                    text=''.join(w['text'] for w in current),
                    start=min(w['start'] for w in current),end=max(w['end'] for w in current),
                    confidence=sum(w['probability'] for w in current)/len(current),
                    avg_logprob=seg.get('avg_logprob',0),no_speech_prob=seg.get('no_speech_prob',0)))
        for w in native:
            stop=pos+len(normalize(w['text']))
            support=[t for t in tags if t[0]<stop and t[1]>pos]
            new_kind='sound' if support and all(t[2] for t in support) else 'dialogue'
            # Never move words across an utterance or a long native gap.
            if current and (new_kind!=kind or w['start']-current[-1]['end']>=1):flush();current=[]
            if not current:kind=new_kind
            current.append(w);pos=stop
        flush()
    return units,issues


def map_aligned_words(aligned,unit,left,right):
    """One local aligner clock, mapped onto original words without text sorting."""
    expected=normalize(unit['text'])
    if normalize(''.join(w['text'] for w in aligned))!=expected:
        return None,'aligned_text_mismatch'
    slots=[];pos=0;previous=-1
    for w in aligned:
        n=normalize(w['text']);a=float(w['start_time'])+left;b=float(w['end_time'])+left
        if not n:continue
        if not math.isfinite(a+b) or not left<=a<=b<=right+.001 or a<previous-.001:
            return None,'invalid_local_alignment'
        if b-a>4:return None,'long_local_word'
        slots.append(dict(lo=pos,hi=pos+len(n),start=a,end=b));pos+=len(n);previous=a
    if not slots or max(s['end'] for s in slots)-min(s['start'] for s in slots)<=.02:
        return None,'unlocated_local_utterance'
    mapped=[];pos=0
    for w in unit['words']:
        stop=pos+len(normalize(w['text']))
        support=[s for s in slots if s['lo']<stop and s['hi']>pos]
        if not support:return None,'unmapped_native_word'
        mapped.append(dict(w,start=min(s['start'] for s in support),end=max(s['end'] for s in support)))
        pos=stop
    return mapped,None


def block_content_risks(block):
    lexical=[normalize(s['text']) for s in block.get('segments',[]) if len(normalize(s['text']))>=4
             and not NONVERBAL.fullmatch(normalize(s['text']))]
    risks=[]
    # Repeated dialogue is investigated, never silently removed. Vocal repeats
    # are intentionally excluded from this lexical hallucination signal.
    if any(n>=3 for n in Counter(lexical).values()):risks.append('repeated_lexical_segments')
    if any(not NONVERBAL.fullmatch(m.group(1)) for t in lexical
           for m in re.finditer(r'(.{2,20}?)(?:\1){3,}',t)):
        risks.append('repeated_lexical_loop')
    for s in block.get('segments',[]):
        text=normalize(s['text'])
        if len(text)<4 or NONVERBAL.fullmatch(text):continue
        if s.get('avg_logprob',0)<-1 and s.get('no_speech_prob',0)>.4:risks.append('low_confidence_speech')
        words=s.get('words',[])
        for i,w in enumerate(words):
            n=normalize(w['text'])
            if n in {'は','ふ','す','く','ぷ','ち','ゅ','ぢ','じ','れ','ろ'} and i+1<len(words):n+=normalize(words[i+1]['text'])
            if w['end']-w['start']>4 and not NONVERBAL.fullmatch(n):risks.append('long_native_word')
    return sorted(set(risks))


def human_confirmation(unit, review):
    return next((r for r in review.get('utterance_confirmations',[])
        if r.get('human_verified') is True and r.get('source_text')==unit['text']
        and r.get('source_start')==unit['start'] and r.get('source_end')==unit['end']
        and r.get('text')),None)


def unconfirmed_content_risks(block, units, review, resolved_units=()):
    """Resolve only risks whose affected native units were all reviewed."""
    segments=block.get('segments',[])
    lexical=[normalize(s['text']) for s in segments]
    repeated={t for t,n in Counter(t for t in lexical if len(t)>=4
        and not NONVERBAL.fullmatch(t)).items() if n>=3}
    affected={}
    for si,seg in enumerate(segments):
        risks=block_content_risks({'segments':[seg]})
        if lexical[si] in repeated:risks.append('repeated_lexical_segments')
        for risk in risks:
            if risk!='long_native_word':affected.setdefault(risk,set()).add(si)
    result=[]
    for risk,owners in affected.items():
        pending=[ui for ui,u in enumerate(units) if ui not in resolved_units and u.get('kind','dialogue')!='sound' and u['segment'] in owners
                 and not human_confirmation(u,review)]
        if pending:result.append(dict(reason=risk,units=pending))
    return result


def verified_native_interval(unit,check,speech_regions):
    if (check.get('native_text')!=unit['text'] or check.get('unit_start')!=unit['start']
        or check.get('unit_end')!=unit['end'] or check.get('hit_token_limit') is not False
        or check.get('independent',{}).get('requested_language')!='Japanese'):
        return None
    if (check.get('clip_start',-1)<unit['start']-.351 or
        check.get('clip_end',float('inf'))>unit['end']+.351):return None
    independent_text=check['independent']['transcription']
    text=normalize(unit['text'])
    if len(text)<2 or normalize(independent_text).count(text)!=1:return None
    a,b=unit['start'],unit['end']
    if not .02<b-a<=8 or any(w['end']-w['start']>4 for w in unit['words']):return None
    support=[(max(a,s['start']),min(b,s['end'])) for s in speech_regions if s['start']<b and s['end']>a]
    if not support:return None
    # Native text is independently supported. Use only measured speech inside
    # this same utterance, never another region or a neighboring sentence.
    lo=max(a,min(x for x,y in support)-.1);hi=min(b,max(y for x,y in support)+.1)
    if hi-lo<=.02:return None
    return [dict(start=lo,end=hi,text=unit['text'])]
