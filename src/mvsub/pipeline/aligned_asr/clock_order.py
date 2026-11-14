"""Resolve conflicting utterance clocks with evidence, never sorting cues."""
from .timing import normalize


def zero_phonetic_fragment(unit):
    text=normalize(unit['text'])
    duration=unit['end']-unit['start']
    if duration>.02 or not text or len(text)>4:return False
    # A zero-length slot carries no usable time at all, so it must never split a
    # block. Low-confidence short particles (は/ぢ/す …) fall here; confident
    # dialogue keeps the original closed-set gate below.
    if unit.get('confidence',1)<.6:return True
    phonetic=set(text)<={'っ','ぁ','ぅ','ち','ゅ','ぷ','じ','あ','う','ん'}
    weak_vowel=(unit.get('kind')=='sound' and unit.get('confidence',1)<.35
        and set(text)<={'あ','い','う','え','お','ぁ','ぃ','ぅ','ぇ','ぉ','ん','っ'})
    return bool(phonetic or weak_vowel)


def reconcile(words,source_units,realign):
    groups=[];omissions=[];repairs=[]
    for word in words:
        if groups and groups[-1][0]['utterance']==word['utterance']:groups[-1].append(word)
        else:groups.append([word])
    i=1
    while i<len(groups):
        previous,current=groups[i-1],groups[i]
        a=min(w['start'] for w in previous);b=min(w['start'] for w in current)
        if b>=a-.001:i+=1;continue
        pu=source_units[previous[0]['utterance']];cu=source_units[current[0]['utterance']]
        if (pu['kind']=='sound' and cu['kind']=='dialogue' and pu['end']-pu['start']<=.02
            and len(normalize(pu['text']))<=4 and pu['segment']==cu['segment']
            and previous[0]['clock']=='independently_classified_native_sound_event'):
            # Event classification proves a voice in this area, not a separate
            # phoneme preceding speech. Preserve the independently aligned line.
            omissions.append(dict(reason='ambiguous_zero_sound_before_dialogue',source_unit=previous[0]['utterance'],
                conflicting_unit=current[0]['utterance'],**pu))
            groups.pop(i-1);i=max(1,i-1);continue
        if (pu['kind']==cu['kind']=='dialogue'
            and previous[0]['clock'] in {'local_qwen','joint_local_qwen'}
            and current[0]['clock']=='local_qwen'):
            candidate=realign(pu,cu)
            if candidate:
                first,second=candidate
                before=min(w['start'] for w in groups[i-2]) if i>1 else float('-inf')
                if (first and second and min(w['start'] for w in first)>=before-.001
                    and min(w['start'] for w in second)>=min(w['start'] for w in first)-.001
                    and normalize(''.join(w['text'] for w in first))==normalize(pu['text'])
                    and normalize(''.join(w['text'] for w in second))==normalize(cu['text'])):
                    groups[i-1]=[dict(w,kind='dialogue',utterance=previous[0]['utterance'],
                        utterance_end=j==len(first)-1,clock='joint_local_qwen') for j,w in enumerate(first)]
                    groups[i]=[dict(w,kind='dialogue',utterance=current[0]['utterance'],
                        utterance_end=j==len(second)-1,clock='joint_local_qwen') for j,w in enumerate(second)]
                    repairs.append(dict(type='joint_neighbor_alignment',units=[previous[0]['utterance'],current[0]['utterance']]))
        i+=1
    return [w for group in groups for w in group],omissions,repairs
