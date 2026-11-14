import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from aligned_asr.utterance import make_units, map_aligned_words, block_content_risks, NONVERBAL, verified_native_interval, unconfirmed_content_risks
import extract_aligned_srt as extractor


class UtteranceTests(unittest.TestCase):
    def test_shifted_segment_words_reassigned_only_by_exact_boundaries(self):
        from aligned_asr.utterance import segment_words
        w=lambda t,a,b:dict(text=t,start=a,end=b,probability=.9)
        block=dict(start=0.,segments=[dict(text='はい',words=[w('はい',0.,1.),w('そう',1.,2.)]),
            dict(text='そうです',words=[w('です',2.,3.)])])
        repaired=segment_words(block)
        self.assertEqual([''.join(w['text'] for w in s['words']) for s in repaired],['はい','そうです'])
        self.assertEqual(repaired[1]['words'][0]['start'],1.)
        self.assertEqual(len(block['segments'][0]['words']),2)
        bad=dict(block,segments=[dict(text='はい',words=[w('違う',0.,1.)])])
        self.assertIs(segment_words(bad),bad['segments'])
        split=dict(block,segments=[dict(text='は',words=[w('はい',0.,1.)]),dict(text='い',words=[])])
        self.assertIs(segment_words(split),split['segments'])

    def test_manual_confirmation_resolves_only_owned_content_risks(self):
        segments=[dict(text='おはようございます',avg_logprob=-2,no_speech_prob=.9,words=[]) for _ in range(3)]
        block=dict(segments=segments)
        units=[dict(segment=i,text=s['text'],start=float(i),end=float(i+1)) for i,s in enumerate(segments)]
        reviews=[dict(human_verified=True,source_text=u['text'],source_start=u['start'],
            source_end=u['end'],text=u['text'],start=u['start'],end=u['end']) for u in units]
        partial=unconfirmed_content_risks(block,units,{'utterance_confirmations':reviews[:2]})
        self.assertEqual({r['reason'] for r in partial},{'repeated_lexical_segments','low_confidence_speech'})
        self.assertTrue(all(r['units']==[2] for r in partial))
        self.assertEqual(unconfirmed_content_risks(block,units,{'utterance_confirmations':reviews}),[])
        for key,value in [('source_text','別文'),('source_start',99.),('source_end',99.),('human_verified',False)]:
            altered=[dict(r) for r in reviews];altered[-1][key]=value
            self.assertTrue(unconfirmed_content_risks(block,units,{'utterance_confirmations':altered}))

    def test_extractor_first_alignment_refreshes_evidence_on_every_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audio=root/'full_audio.wav';audio.write_bytes(b'audio')
            (root/'model.bin').touch();(root/'config.json').touch()
            (root/'full_audio.candidate.ja.srt').write_text('1\n00:00:01,000 --> 00:00:02,000\nはい\n',encoding='utf-8')
            for name,data in [('alignment.json',{'issues':[]}),('event_captioning.json',{'issues':[]}),
                              ('zero_event_repair.json',{'unresolved':[]})]:
                (root/name).write_text(json.dumps(data))
            commands=[]
            def execute(cmd,**kwargs):
                commands.append(cmd)
                return SimpleNamespace(returncode=0,stdout='',stderr='')
            with patch.object(extractor,'ASR_MODEL',root),patch.object(extractor,'ALIGN_MODEL',root),\
                 patch.object(extractor,'ALIGN_PYTHON',audio),patch.object(extractor,'ALIGN_LIBS',root),\
                 patch.object(extractor,'probe_duration',return_value=10.),\
                 patch.object(extractor.subprocess,'run',side_effect=execute):
                for _ in range(2):self.assertTrue(extractor.extract(audio,root/'movie.srt',root)['quality_passed'])
            aligns=[c for c in commands if Path(c[1]).name=='align.py']
            self.assertEqual(['--fresh-verification' in c for c in aligns],[True,False,True,False])

    def test_honorific_and_noise_keep_original_words_and_order(self):
        seg={'text':'はぁお兄さん','words':[
            {'text':'はぁ','start':0.,'end':1.},
            {'text':'お','start':1.2,'end':1.4},
            {'text':'兄さん','start':1.4,'end':2.}]}
        tag=lambda text:SimpleNamespace(words=['はぁ','お','兄さん'],postags=['感動詞','接頭辞','名詞'])
        units,issues=make_units({'start':10.,'segments':[seg]},tag)
        self.assertFalse(issues)
        self.assertEqual([(u['kind'],u['text']) for u in units],[('sound','はぁ'),('dialogue','お兄さん')])
        self.assertEqual(''.join(u['text'] for u in units),seg['text'])

    def test_empty_or_wrong_alignment_rejected_even_with_valid_clock(self):
        unit={'text':'舐めてほしい','words':[{'text':'舐めてほしい','start':1.,'end':2.}]}
        for aligned in ([],[{'text':'別の文','start_time':1.,'end_time':2.}]):
            self.assertEqual(map_aligned_words(aligned,unit,0.,3.),(None,'aligned_text_mismatch'))

    def test_zero_suffix_keeps_text_inside_same_positive_utterance(self):
        unit={'text':'はいです','words':[{'text':'はい','start':1.,'end':2.},{'text':'です','start':2.,'end':2.}]}
        mapped,error=map_aligned_words([{'text':'はい','start_time':1.,'end_time':2.},
            {'text':'です','start_time':2.,'end_time':2.}],unit,0.,3.)
        self.assertIsNone(error)
        self.assertEqual(''.join(w['text'] for w in mapped),'はいです')
        self.assertEqual(mapped[-1]['start'],2.)

    def test_local_alignment_cannot_escape_real_audio_window(self):
        unit={'text':'はい','words':[{'text':'はい','start':10.,'end':11.}]}
        self.assertIsNotNone(map_aligned_words([{'text':'はい','start_time':0.,'end_time':4.}],unit,10.,12.)[1])

    def test_repeated_sounds_are_not_lexical_hallucination(self):
        block={'segments':[{'text':'はぁんん','words':[]} for _ in range(8)]}
        self.assertFalse(block_content_risks(block))
        block['segments']=[{'text':'おはようございます','words':[]} for _ in range(8)]
        self.assertIn('repeated_lexical_segments',block_content_risks(block))

    def test_sound_stem_suffix_does_not_swallow_the_next_stem(self):
        from aligned_asr.utterance import normalize
        for text in ['ごくっんぐっんぐっ','んむぅっじゅぽっじゅぽっ','ぢゅぽっ'*20,
                     'ちゅっちゅぱっちゅぽっ','むふぅっむふぅっ','クッ、クッ、アァッ',
                     'あ゛ぁあ゛っ、ん゛ん゛ぅ','ごきゅっひっくにゃぁひゃっ']:
            self.assertIsNotNone(NONVERBAL.fullmatch(normalize(text)),text)
            self.assertFalse(block_content_risks({'segments':[{'text':text,'words':[]}]}))
        for text in ['お兄さん','すごい','だめだめだめだめ','ごくっ日本語んぐっ']:
            self.assertIsNone(NONVERBAL.fullmatch(normalize(text)),text)

    def test_phonetic_suffix_inside_a_sound_run_is_not_a_name_suffix(self):
        seg=dict(text='ごくんっやばい',words=[dict(text='ご',start=0.,end=.2),
            dict(text='くんっ',start=.2,end=.5),dict(text='やばい',start=.5,end=1.)])
        tag=lambda _:SimpleNamespace(words=['ご','くんっ','やばい'],postags=['接頭辞','接尾辞','形容詞'])
        units,issues=make_units(dict(start=0.,segments=[seg]),tag)
        self.assertFalse(issues)
        self.assertEqual([(u['kind'],u['text']) for u in units],[('sound','ごくんっ'),('dialogue','やばい')])

    def test_long_phonetic_prefix_does_not_backtrack_or_become_a_word(self):
        self.assertIsNone(NONVERBAL.fullmatch('あ'*10000+'日本語'))
        for text in ['すぅーすぅー','ぢゅるるるっ','んふぅっ','ちゅぱっちゅぱっ']:
            self.assertIsNotNone(NONVERBAL.fullmatch(text),text)

    def test_native_fallback_requires_independent_text_and_same_window_speech(self):
        unit={'text':'そう?','start':10.,'end':11.,'words':[{'text':'そう?','start':10.,'end':11.}]}
        check=dict(native_text='そう?',unit_start=10.,unit_end=11.,clip_start=9.65,clip_end=11.35,
                   hit_token_limit=False,independent={'requested_language':'Japanese','transcription':'そう'})
        self.assertIsNone(verified_native_interval(unit,{},[{'start':10.,'end':11.}]))
        self.assertIsNone(verified_native_interval(unit,check,[]))
        self.assertIsNone(verified_native_interval(unit,check,[{'start':1.,'end':2.}]))
        bad=dict(check,clip_start=0.,clip_end=30.)
        self.assertIsNone(verified_native_interval(unit,bad,[{'start':10.,'end':11.}]))
        result=verified_native_interval(unit,check,[{'start':10.3,'end':10.8}])
        self.assertEqual(result[0]['text'],'そう?')
        self.assertAlmostEqual(result[0]['start'],10.2)

    def test_failed_quality_preserves_previous_srt_and_every_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);job=root/'job';job.mkdir();audio=job/'full_audio.wav';audio.write_bytes(b'audio')
            (root/'model.bin').touch();(root/'config.json').touch()
            output=root/'movie.srt';output.write_text('old subtitle',encoding='utf-8')
            candidate=job/'full_audio.candidate.ja.srt'
            candidate.write_text('1\n00:00:01,000 --> 00:00:02,000\nはい\n',encoding='utf-8')
            for name,data in [('alignment.json',{'issues':[{'type':'unverified_content'}]}),
                              ('event_captioning.json',{'issues':[]}),('zero_event_repair.json',{'unresolved':[]})]:
                (job/name).write_text(json.dumps(data),encoding='utf-8')
            with patch.object(extractor,'ASR_MODEL',root),patch.object(extractor,'ALIGN_MODEL',root),\
                 patch.object(extractor,'ALIGN_PYTHON',audio),patch.object(extractor,'ALIGN_LIBS',root),\
                 patch.object(extractor,'probe_duration',return_value=10.),\
                 patch.object(extractor.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='',stderr='')):
                with self.assertRaises(ValueError):extractor.extract(audio,output,job)
            self.assertEqual(output.read_text('utf-8'),'old subtitle')
            self.assertFalse(json.loads(output.with_suffix('.alignment.json').read_text('utf-8'))['quality_passed'])
            self.assertTrue(candidate.exists())


if __name__=='__main__':unittest.main()
