"""Retain unresolved events; forced text alignment is never content evidence."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from nosub_pipeline import parse_srt_cues, probe_duration


def run(dest,audio):
    started=time.perf_counter()
    source=dest/'full_audio.events.draft.srt';cues=parse_srt_cues(source)
    duration=probe_duration(audio)
    if duration is None:raise ValueError('Missing audio duration')
    unresolved=[dict(id=i,start=a,end=b,text=t) for i,a,b,t in cues if not 0<=a<b<=duration+.001]
    # Review retains every event including failed locations. The candidate is
    # evidence only; extractor refuses publication while any event is unresolved.
    (dest/'full_audio.review.ja.srt').write_bytes(source.read_bytes())
    (dest/'full_audio.candidate.ja.srt').write_bytes(source.read_bytes())
    (dest/'zero_event_repair.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,
        repairs=[],unresolved=unresolved,candidate_cues=len(cues),review_cues=len(cues),
        note='No events silently removed. Unresolved times block downstream publication.'),ensure_ascii=False,indent=2),encoding='utf-8')
    print('RETAIN_EVENTS',len(cues),'unresolved',len(unresolved),flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
