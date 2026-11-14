"""Speech regions only; VAD never removes breathing or other sound captions."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.io import wavfile
from faster_whisper.vad import get_speech_timestamps,VadOptions

dest=Path(sys.argv[1]);audio=Path(sys.argv[2]);started=time.perf_counter()
sr,wav=wavfile.read(audio)
if sr!=16000 or wav.ndim!=1:raise ValueError('Expected mono 16kHz PCM')
regions=get_speech_timestamps(wav.astype(np.float32)/32768,VadOptions(threshold=.35,min_speech_duration_ms=100))
report=dict(audio_sha1=hashlib.sha1(audio.read_bytes()).hexdigest().upper(),
    speech_regions=[dict(start=r['start']/sr,end=r['end']/sr) for r in regions],
    seconds=time.perf_counter()-started,note='Speech evidence; never a filter for nonverbal events.')
(dest/'vad_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
