"""Bounded evidence checks for automatic word-clock recovery."""
import math
from .timing import normalize


def matching_words(segments,text,anchor=None):
    words=[w for s in segments for w in s.get('words',[])]
    if anchor is not None:
        words=[w for w in words if anchor[0]-.15<=(w['start']+w['end'])/2<=anchor[1]+.15]
    expected=normalize(text);joined=normalize(''.join(w['text'] for w in words))
    if not expected or joined.count(expected)!=1:return None
    left=joined.index(expected);right=left+len(expected);pos=0;chosen=[]
    for word in words:
        stop=pos+len(normalize(word['text']))
        if pos<right and stop>left:
            if pos<left or stop>right:return None
            chosen.append(word)
        pos=stop
    previous=-1
    for w in chosen:
        a,b=w['start'],w['end']
        if not math.isfinite(a+b) or a<previous or not 0<=a<=b or b-a>4:return None
        previous=a
    if not chosen or max(w['end'] for w in chosen)-min(w['start'] for w in chosen)<=.02:return None
    return chosen


def consistent_recovery(unit,first,second):
    anchor=(unit['start'],unit['end'])
    one=matching_words(first,unit['text'],anchor);two=matching_words(second,unit['text'],anchor)
    if not one or not two:return None
    a,b=min(w['start'] for w in one),max(w['end'] for w in one)
    c,d=min(w['start'] for w in two),max(w['end'] for w in two)
    if max(abs(a-c),abs(b-d))>.6:
        overlap=max(0,min(b,d)-max(a,c))
        # If one model stretches a word into preceding breath, retain the
        # complete narrower model clock only when the two intervals overlap.
        if min(b-a,d-c)<.15 or overlap/min(b-a,d-c)<.8:return None
        if d-c<b-a:one=two;a,b=c,d
    if b-a>8:return None
    # Both decodes must place this sentence near the original utterance.
    if a<unit['start']-2 or b>unit['end']+2:return None
    return [dict(w) for w in one]


def native_clock_supported(unit,duration):
    words=unit['words'];previous=-1
    if unit.get('confidence',0)<.5 or unit.get('no_speech_prob',1)>.8:return False
    a,b=unit['start'],unit['end']
    if not max(.02,min(.2,.04*len(normalize(unit['text']))))<b-a<=12:return False
    for w in words:
        x,y=w['start'],w['end']
        if not math.isfinite(x+y) or not 0<=x<=y<=duration or x<previous or y-x>4:return False
        previous=x
    return True


def very_low_content_confidence(unit,reason,independent_text):
    text=normalize(unit['text']);independent=normalize(independent_text)
    if text and text in independent:return False
    if reason in {'low_confidence_speech','high_no_speech_probability'}:return True
    # Natural repetitions are not rejected just for their count. A collapsed
    # word clock plus no independent lexical support is a very low confidence
    # loop; retain ordinary timed repeats as report-only uncertainty.
    words=[w for w in unit['words'] if normalize(w['text'])]
    positive=sum(w['end']-w['start']>.02 for w in words)/max(1,len(words))
    return positive<.5 or unit.get('confidence',0)<.35


def vocal_event_supported(check):
    scores=check.get('scores',{})
    voice=max((scores.get(k,0) for k in ('Breathing','Gasp','Groan','Sigh','Wheeze','Snort','Pant','Whimper')),default=0)
    background=max(scores.get('Music',0),scores.get('Silence',0))
    return voice>=.15 and voice>background*1.5


def nonvocal_background_supported(check):
    scores=check.get('scores',{})
    if not {'Music','Speech','Singing','Breathing'}.issubset(scores):return False
    vocal_labels=('Speech','Conversation','Narration, monologue','Singing','Humming','Choir',
        'Chant','Whispering','Laughter','Crying, sobbing','Whimper','Groan','Grunt','Shout',
        'Screaming','Breathing','Gasp','Sigh','Wheeze','Pant','Snort','Cough','Sneeze')
    voice=max(scores.get(k,0) for k in vocal_labels)
    return voice<.08 and (scores['Music']>=.7 or scores.get('Silence',0)>=.9)
