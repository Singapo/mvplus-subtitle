"""Synthetic inter-stage contracts for queue tests; no acoustic/LLM inference."""
import json
from pathlib import Path
from source_review import make_report,sha256
from translate_srt_batch import parse_srt
from audio_archive import atomic_json


def review_fixture(source,audio,work,asr_path,*args,**kwargs):
    work=Path(work)
    # Deliberately separate from the runner's timing-bound full_audio.wav.
    fake=work/'synthetic_review_audio.wav';fake.write_bytes(b'synthetic audio, not a real decoder fixture')
    path=work/'source_review.json'
    make_report(source,fake,path)
    return path


def audit_fixture(source,translated):
    source,translated=Path(source),Path(translated)
    rows=parse_srt(translated)
    path=translated.with_suffix(translated.suffix+'.translation')/'report.json'
    path.parent.mkdir(exist_ok=True)
    report=dict(protocol='grounded-json-v1',quality_passed=True,output_updated=True,
        source_file_sha256=sha256(source),output_sha256=sha256(translated),
        source_review_available=True,source_review_sha256=sha256(source.parent/'source_review.json'),
        items=[dict(id=e[0],zh=e[3],review='none') for e in rows],missing_ids=[],review_ids=[],
        source_hold_ids=[],failures=[])
    atomic_json(path,report)
    from grounded_translation import write_certificate
    write_certificate(report,path.parent)
