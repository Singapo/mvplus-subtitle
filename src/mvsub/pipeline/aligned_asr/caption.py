"""Separate nonverbal sounds from dialogue so sounds do not advance a later line."""
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

WORK=Path(sys.argv[1]).resolve().parent
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import config
sys.path.insert(0,str(config.align_libs()))
import nagisa
sys.stdout.reconfigure(encoding='utf-8')
NONVERBAL=re.compile(r'(?:[あぁうぅえぇおぉんっ]+|は[ぁあ]+|ふ[ぅう]+|[ちゅぱれろむ]+っ?)$')
def norm(t):return ''.join(c for c in t if not c.isspace() and not unicodedata.category(c).startswith(('P','S')))
def stamp(s):
    n=round(s*1000);h,n=divmod(n,3600000);m,n=divmod(n,60000);sec,n=divmod(n,1000)
    return f'{h:02}:{m:02}:{sec:02},{n:03}'
for label in [Path(sys.argv[1]).name]:
    dest=WORK/label;data=json.loads((dest/'alignment.json').read_text(encoding='utf-8'))
    asr=json.loads((dest/'asr.json').read_text(encoding='utf-8'))
    durations={s['sample']:s['duration'] for s in asr['samples']}
    cues={};issues=[]
    for block in data['blocks']:
        sample=block['sample'];group=[];a=b=None;kind=None;utterance=None
        cues.setdefault(sample,[])
        if data.get('policy')=='utterance_clock_v2':
            # The local aligner provides complete utterance units, but one unit
            # often spans several sentences (sometimes tens of seconds apart).
            # Split inside the unit on sentence-final punctuation so every line
            # keeps its own word-level clock instead of stretching one cue.
            bucket=[];identity=None
            def emit_segment(seg):
                if not seg:return
                text=''.join(w['text'] for w in seg)
                if not text.strip():return
                start=min(w['start'] for w in seg);end=max(w['end'] for w in seg)
                cues.setdefault(sample,[]).append([start,end,text,seg[0]['kind'],identity])
            def emit_unit():
                if not bucket:return
                segs=[];cur=[]
                for w in bucket:
                    if cur:
                        tail=cur[-1]['text'].rstrip()
                        punct=tail.endswith(('。','．','！','？','?','!','．'))
                        gap=w['start']-cur[-1]['end']
                        span=cur[-1]['end']-cur[0]['start']
                        # 句末标点，或超过 3 秒的段落里出现 0.6 秒以上停顿
                        # （喘息/漏标点的长句），都在此处断开，各自保留词级时间。
                        if punct or (gap>.6 and span>3):
                            segs.append(cur);cur=[]
                    cur.append(w)
                if cur:segs.append(cur)
                # 零时长碎段并入前一段，避免产生无法显示的独立字幕
                merged=[]
                for piece in segs:
                    a=min(w['start'] for w in piece);b=max(w['end'] for w in piece)
                    if b-a<=.02 and merged:merged[-1].extend(piece)
                    else:merged.append(piece)
                for piece in merged:emit_segment(piece)
            for word in block['words']:
                if not math.isfinite(word['start']+word['end']) or not 0<=word['start']<=word['end']<=durations[sample]+.001:
                    issues.append(dict(sample=sample,type='invalid_word_time',word=word));continue
                if bucket and word['utterance']!=identity:emit_unit();bucket=[]
                identity=word['utterance'];bucket.append(word)
            emit_unit()
            continue
        tagged=nagisa.tagging(block['original_text'])
        tags=[(norm(t),p) for t,p in zip(tagged.words,tagged.postags) if norm(t)]
        tag_pos=0
        def flush():
            if group:cues.setdefault(sample,[]).append([a,b,''.join(group).strip(),kind,utterance])
        for word in block['words']:
            start,end,text=word['start'],word['end'],word['text']
            if not math.isfinite(start+end) or end<start or start<0 or start>=durations[sample] or end>durations[sample]+.1:
                issues.append(dict(sample=sample,type='invalid_word_time',word=word));continue
            end=min(end,durations[sample])
            normalized=norm(text);pos=None
            for j in range(tag_pos,min(len(tags),tag_pos+5)):
                if tags[j][0]==normalized:
                    pos=tags[j][1];tag_pos=j+1;break
            sound=bool(NONVERBAL.fullmatch(normalized))
            # Honorific prefixes and grammatical particles are not vocal noises.
            # E.g. お + 兄さん must remain お兄さん, never "oh" + "brother".
            if pos in {'接頭辞','接尾辞','助詞','助動詞','動詞'}:sound=False
            if len(normalized)==1 and normalized not in {'っ','ぁ','ぅ'} and pos!='感動詞':sound=False
            new_kind=word.get('kind','sound' if sound else 'dialogue')
            current_utterance=word.get('utterance')
            if group and current_utterance!=utterance:flush();group=[]
            # A failed zero timestamp must not pull later dialogue back to the
            # block origin; retain it as an isolated review event.
            if end-start<=.02:
                if group and new_kind==kind and start>=a and abs(start-b)<=.8:
                    group.append(text)
                    if word.get('utterance_end'):flush();group=[]
                    continue
                flush();group=[]
                cues.setdefault(sample,[]).append([start,end,text,new_kind,current_utterance])
                continue
            if group and (new_kind!=kind or start-b>=1 or end-a>8 or start<a-.08):
                flush();group=[]
            if not group:a,b,kind,utterance=start,end,new_kind,current_utterance
            else:a,b=min(a,start),max(b,end)
            group.append(text)
            if word.get('utterance_end'):flush();group=[]
        flush()
    counts={};repairs=[]
    for sample,rows in cues.items():
        # A zero-time phonetic fragment (especially small tsu) belongs to its
        # neighboring utterance. Preserve text by merging; never invent a time.
        for i in range(len(rows)-1,-1,-1):
            a,b,text,kind=rows[i][:4]
            if b-a>.02:continue
            candidates=[]
            fragment=bool(norm(text)) and set(norm(text))<={'っ','ぁ','ぅ'}
            for j in (i-1,i+1):
                if 0<=j<len(rows):
                    x,y,t,k=rows[j][:4]
                    same_unit=rows[j][4]==rows[i][4]
                    same_sentence=(rows[j][4] is not None and rows[i][4] is not None and
                        rows[j][4].split(':')[:2]==rows[i][4].split(':')[:2])
                    native_sound_near=(data.get('policy')=='utterance_clock_v2' and kind==k=='sound'
                        and max(x-a,a-y,0)<=.2)
                    if (same_unit or (fragment and same_sentence) or native_sound_near) and y>x and (k==kind or fragment):
                        distance=max(x-a,a-y,0)
                        if distance<=.8:candidates.append((distance,j))
            if candidates:
                _,j=min(candidates)
                rows[j][2]=rows[j][2]+text if j<i else text+rows[j][2]
                # Local v2 slots are bounded to the same utterance's audio.
                # Preserve a zero prefix's location in the utterance envelope;
                # do not invent a separate dwell time for that prefix.
                if rows[i][4] is not None:
                    rows[j][0]=min(rows[j][0],a);rows[j][1]=max(rows[j][1],b)
                repairs.append(dict(sample=sample,type='merge_phonetic_fragment',text=text,
                                    fragment_start=a,target_start=rows[j][0],target_end=rows[j][1]))
                rows.pop(i)
        # Neighbouring utterance slots can nest: a short line may sit entirely
        # inside the previous long one, which makes a player show both at once.
        # Sort by onset, then merge a nested short line into its host so the
        # text is preserved and only one line is ever on screen.
        rows.sort(key=lambda r:(r[0],r[1]))
        merged=[]
        for row in rows:
            if merged:
                prev=merged[-1]
                # 后一条完全含住前一条：前一条并入后一条，避免 0.02s 碎句残留
                if prev[0]>=row[0]-.001 and prev[1]<=row[1]+.001:
                    repairs.append(dict(sample=sample,type='merge_preceding_into_next',text=prev[2],
                        nested_start=prev[0],nested_end=prev[1],target_start=row[0],target_end=row[1]))
                    row[2]=prev[2]+row[2]
                    row[0]=min(prev[0],row[0])
                    merged[-1]=row
                    continue
                if row[1]<=prev[1]+.001 and prev[1]-row[0]>.3:
                    repairs.append(dict(sample=sample,type='merge_nested_utterance',text=row[2],
                        nested_start=row[0],nested_end=row[1],target_start=prev[0],target_end=prev[1]))
                    prev[2]=prev[2]+row[2]
                    prev[1]=max(prev[1],row[1])
                    continue
                if row[0]<prev[1]-.001:
                    # Keep both lines, but never let their intervals cross.
                    if row[1]-prev[1]>.3:row[0]=prev[1]
                    else:prev[1]=max(prev[0]+.02,row[0])
                    if row[0]<prev[1]-.001:row[0]=prev[1]+.005
            merged.append(row)
        rows=merged
        for i,row in enumerate(rows):
            a,b,text,kind=row[:4]
            if b-a<=.02:issues.append(dict(sample=sample,type='unresolved_zero_or_tiny',start=a,end=b,text=text))
            # Brief sounds need readable dwell time, capped by the next cue.
            if kind=='sound' and b-a<.4 and b>a:
                cap=rows[i+1][0] if i+1<len(rows) and rows[i+1][0]>=b else b
                row[1]=min(max(b,a+.4),cap)
            if i and a<rows[i-1][0]-.001:issues.append(dict(sample=sample,type='backward_cue',start=a,text=text))
        (dest/f'{sample}.events.draft.srt').write_text('\n\n'.join(
            f'{i}\n{stamp(row[0])} --> {stamp(row[1])}\n{row[2]}' for i,row in enumerate(rows,1))+'\n',encoding='utf-8')
        counts[sample]=dict(cues=len(rows),sound_cues=sum(x[3]=='sound' for x in rows))
    (dest/'event_captioning.json').write_text(json.dumps(dict(counts=counts,issues=issues,repairs=repairs,
        rule='Separate sound/dialogue; brief valid events hold up to 0.4s capped by next onset',
        status='review_required' if issues or data['issues'] else 'automatic_checks_only'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('EVENTS',label,counts,'issues',len(issues),flush=True)
