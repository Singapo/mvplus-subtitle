"""Independent sound classification at failed vocal-event anchors."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from asr_config import MODEL_HOME,ALIGN_LIBS
sys.path.insert(0,str(ALIGN_LIBS));os.environ['HF_HUB_OFFLINE']='1'
from aligned_asr.utterance import make_units
import nagisa
import numpy as np
from scipy.io import wavfile
import torch
from transformers import AutoFeatureExtractor,AutoModelForAudioClassification


def run(dest,audio):
    started=time.perf_counter();alignment=json.loads((dest/'alignment.json').read_text('utf-8'))
    targets=[dict(i,kind='vocal_event') for i in alignment['issues'] if i['type']=='unverified_acoustic_sound_event']
    asr=json.loads((dest/'asr.json').read_text('utf-8'))
    review_path=dest/'manual_audio_review.json'
    review=json.loads(review_path.read_text('utf-8')) if review_path.exists() else {}
    for issue in alignment['issues']:
        if issue['type']!='unverified_content' or issue['reason']!='low_confidence_speech':continue
        units,_=make_units(asr['blocks'][issue['block']],nagisa.tagging)
        regions=review.get('non_speech_regions',[])
        units=[u for u in units if not any(r.get('human_verified') is True and
            r.get('classification')=='background_music_only' and r['start']<=u['start'] and u['end']<=r['end'] for r in regions)]
        for ui in issue['units']:
            if ui<len(units) and .02<units[ui]['end']-units[ui]['start']<=8:
                targets.append(dict(units[ui],block=issue['block'],unit=ui,kind='background'))
    checks=[]
    if targets:
        path=MODEL_HOME/'models/audio-event-verifier'
        if not (path/'config.json').exists():raise RuntimeError('Local event verifier missing; no download')
        sr,wav=wavfile.read(audio)
        if sr!=16000 or wav.ndim!=1:raise ValueError('Expected mono 16kHz PCM')
        wav=wav.astype(np.float32)/32768
        processor=AutoFeatureExtractor.from_pretrained(path,local_files_only=True)
        model=AutoModelForAudioClassification.from_pretrained(path,local_files_only=True).to('cuda').eval()
        for issue in targets:
            left=max(0,issue['start']-.6);right=min(len(wav)/sr,issue['end']+.6)
            inputs=processor(wav[round(left*sr):round(right*sr)],sampling_rate=sr,return_tensors='pt').to('cuda')
            with torch.inference_mode():scores=model(**inputs).logits.sigmoid()[0].cpu().numpy()
            checks.append(dict(block=issue['block'],unit=issue['unit'],kind=issue['kind'],source_text=issue['text'],start=issue['start'],end=issue['end'],
                clip_start=left,clip_end=right,scores={model.config.id2label[i]:float(s) for i,s in enumerate(scores)}))
    (dest/'event_verification.json').write_text(json.dumps(dict(
        asr_sha1=hashlib.sha1((dest/'asr.json').read_bytes()).hexdigest().upper(),
        audio_sha1=hashlib.sha1(audio.read_bytes()).hexdigest().upper(),checks=checks,
        seconds=time.perf_counter()-started),ensure_ascii=False,indent=2),encoding='utf-8')
    print('EVENT_VERIFICATION',len(checks),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
