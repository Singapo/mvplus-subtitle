"""Keep one coherent clock per block; never patch isolated words across clocks."""
import math
import unicodedata


def normalize(text):
    text=unicodedata.normalize('NFKC',text)
    text=''.join(chr(ord(c)-0x60) if '\u30a1'<=c<='\u30f6' else c for c in text)
    return ''.join(c for c in text if not c.isspace() and c not in {'\u3099','\u309a'}
                   and not unicodedata.category(c).startswith(('P', 'S')))


def timing_problems(words, left, right):
    problems=[]
    for i,w in enumerate(words):
        a,b=w['start'],w['end']
        if not math.isfinite(a+b) or a<left-.1 or b>right+.1 or b<a:
            problems.append({'type':'invalid_time','word':i})
        if b-a<=.02:problems.append({'type':'unlocated_word','word':i})
        if b-a>8:problems.append({'type':'long_word','word':i})
        if i and a<words[i-1]['start']-.08:
            problems.append({'type':'backward_word','word':i})
    return problems


def coherent_words(aligned, block):
    """Prefer independent alignment; use the entire native clock on failure.

    Native zero-duration words stay at their native position for local repair.
    No sorting, clipping, guessed duration, or text removal is performed here.
    """
    left,right=block['start'],block['end']
    qwen=[dict(start=float(w['start_time'])+left,
               end=float(w['end_time'])+left,text=w['text']) for w in aligned]
    # Keep ASR utterance boundaries through the aligner's punctuation removal.
    edges=set();edge=0
    for s in block.get('segments',[]):
        edge+=sum(len(normalize(w['text'])) for w in s.get('words',[]))
        edges.add(edge)
    pos=0
    for w in qwen:
        before=pos
        pos+=len(normalize(w['text']))
        if pos in edges:w['utterance_end']=True
        # One aligner token can span two ASR utterances. Do not claim their
        # boundary was preserved; keep this explicit for the quality gate.
        if any(before<e<pos for e in edges):w['crosses_utterance']=True
    problems=timing_problems(qwen,left,right)
    if not problems:return qwen, {'clock':'qwen','trigger':[]}
    native=[];pos=0
    for s in block.get('segments',[]):
        for w in s.get('words',[]):
            n=normalize(w['text'])
            if n:
                native.append(dict(lo=pos,hi=pos+len(n),start=w['start']+left,end=w['end']+left,text=w['text']))
                pos+=len(n)
    if normalize(''.join(w['text'] for w in native))!=normalize(''.join(w['text'] for w in qwen)):
        return qwen, {'clock':'qwen_unresolved','trigger':problems,'reason':'native_text_mismatch'}
    invalid=[p for p in timing_problems(native,left,right)
             if p['type'] in {'invalid_time','backward_word'}]
    if invalid:return qwen, {'clock':'qwen_unresolved','trigger':problems,'reason':'invalid_native_clock'}
    mapped=[];pos=0
    for w in qwen:
        stop=pos+len(normalize(w['text']))
        support=[n for n in native if n['lo']<stop and n['hi']>pos]
        if not support:
            return qwen, {'clock':'qwen_unresolved','trigger':problems,'reason':'unmapped_text'}
        mapped.append(dict(w,start=min(n['start'] for n in support),end=max(n['end'] for n in support)))
        pos=stop
    return mapped, {'clock':'native_block','trigger':problems,
                    'remaining':timing_problems(mapped,left,right)}
