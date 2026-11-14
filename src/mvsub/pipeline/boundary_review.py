"""Bounded, traceable review of unsupported zero-time boundary units."""
import json
from pathlib import Path
from audio_archive import digest
from asr_config import ASR_MODEL,MODEL_HOME
from aligned_asr.utterance import NONVERBAL,model_identity
from aligned_asr.timing import normalize


def decision(issue,segment_text,witnesses):
    if issue.get('type')!='cross_block_source_conflict':raise ValueError('Unexpected issue')
    if issue['start']!=issue['end'] or issue['start']!=issue['boundary']:
        raise ValueError('Only exact zero-time boundary units can be reviewed')
    if not issue.get('words') or any(w['start']!=issue['start'] or w['end']!=issue['end'] for w in issue['words']):
        raise ValueError('Nonzero word cannot be omitted')
    if set(witnesses)!={'760M_near','large-v3_near','760M_owner','large-v3_owner'}:
        raise ValueError('Four contextual decodes required')
    text=normalize(issue['text']);source=normalize(segment_text)
    if not text or not source.endswith(text):raise ValueError('Source tail differs')
    prefix=source[:-len(text)]
    fragments={'は','ふ','す','ご','じ','ぢ','ち','ゅ','ぱ','ぷ','ぽ'}
    tail=any(text.endswith(f) and (not text[:-len(f)] or NONVERBAL.fullmatch(text[:-len(f)])) for f in fragments)
    if prefix and NONVERBAL.fullmatch(prefix) and tail:
        return 'unlocated_truncated_nonverbal_tail_after_context_review'
    # A normal word requires absence of even a spelling-equivalent lexical
    # occurrence in all four decodes; never treat repetition as absence.
    lexical=text.replace('っ','')
    for witness in witnesses.values():
        transcript=normalize(''.join(s['text'] for s in witness['segments'])).replace('っ','')
        if lexical in transcript:raise ValueError('Independent lexical support exists; keep blocked')
    return 'unsupported_zero_time_lexical_unit_after_context_review'


def verify_manifest(job,report):
    job=Path(job);path=job/'boundary_review.json'
    if digest(path)!=report.get('boundary_review_sha1'):raise ValueError('Boundary review changed')
    review=json.loads(path.read_text('utf-8'))
    if (review.get('policy')!='four_decode_zero_boundary_review' or review['audio_sha1']!=report['audio_sha1']
        or review['asr_sha1']!=digest(job/'asr.json') or review['pipeline_revision']!=report['pipeline_revision']):
        raise ValueError('Boundary review inputs differ')
    asr=json.loads((job/'asr.json').read_text('utf-8'))
    for item in review['resolved']:
        witnesses={}
        for key,data in item['witnesses'].items():
            file=(job/data['file']).resolve()
            if not file.is_relative_to((job/'boundary_review').resolve()) or digest(file)!=data['sha1']:
                raise ValueError('Boundary witness changed')
            witness=json.loads(file.read_text('utf-8'))
            model=ASR_MODEL if key.startswith('760M_') else MODEL_HOME/'models/whisper-large-v3'
            if witness['audio_sha1']!=review['audio_sha1'] or witness['model_identity']!=model_identity(model):
                raise ValueError('Boundary witness model/audio differs')
            if witness['target']!=item['issue']:raise ValueError('Boundary witness target differs')
            block=asr['blocks'][item['issue']['block']]
            if key.endswith('_owner'):
                if witness['owner_start']!=block['start'] or witness['owner_end']!=block['end']:
                    raise ValueError('Owner interval differs')
            else:
                issue=item['issue']
                if abs(witness['clip_start']-max(0,issue['start']-4))>.001 or abs(witness['clip_end']-min(report['audio_seconds'],issue['end']+4))>.001:
                    raise ValueError('Near context interval differs')
            witnesses[key]=witness
        issue=item['issue'];segment=asr['blocks'][issue['block']]['segments'][issue['segment']]
        if decision(issue,segment['text'],witnesses)!=item['reason']:raise ValueError('Boundary decision differs')
    return review
