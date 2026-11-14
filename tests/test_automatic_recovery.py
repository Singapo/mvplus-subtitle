from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from aligned_asr.recovery import matching_words,consistent_recovery,vocal_event_supported,nonvocal_background_supported,native_clock_supported,very_low_content_confidence


class RecoveryTests(unittest.TestCase):
    def test_recovered_clock_preserves_source_kana_and_punctuation(self):
        from aligned_asr.utterance import map_aligned_words
        original='こいつのちんぽも、大きくしちゃったよ'
        unit=dict(text=original,words=[dict(text=original,start=1.,end=2.)])
        recovered=[dict(text='こいつのチンポも大きくしちゃったよ。',start_time=3.,end_time=4.)]
        mapped,error=map_aligned_words(recovered,unit,0.,10.)
        self.assertIsNone(error)
        self.assertEqual(mapped[0]['text'],original)
        self.assertEqual((mapped[0]['start'],mapped[0]['end']),(3.,4.))

    def test_legacy_repair_does_not_modify_source_or_verification(self):
        import json,subprocess,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            for name in ('asr.json','content_verification.json'):
                (dest/name).write_bytes(b'{"source":"keep exactly"}')
            before={p.name:p.read_bytes() for p in dest.iterdir()}
            subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'/'aligned_asr/repair_recognition.py'),str(dest)],check=True)
            self.assertTrue(all((dest/name).read_bytes()==value for name,value in before.items()))
            self.assertEqual(json.loads((dest/'recognition_repair.json').read_text())['repairs'],[])

    def segments(self,a=10.,b=11.,text='すごい'):
        return [dict(words=[dict(text=text,start=a,end=b)])]

    def test_two_models_recover_only_unique_same_text_near_native_anchor(self):
        unit=dict(text='すごい',start=9.,end=11.)
        self.assertEqual(consistent_recovery(unit,self.segments(),self.segments(10.1,11.1))[0]['start'],10.)
        for second in (self.segments(text='やばい'),self.segments(12.,13.),self.segments(10.,10.)):
            self.assertIsNone(consistent_recovery(unit,self.segments(),second))
        self.assertIsNone(consistent_recovery(dict(unit,start=20.,end=22.),self.segments(),self.segments()))

    def test_complete_narrower_clock_can_resolve_a_stretched_first_syllable(self):
        unit=dict(text='やばい',start=19.,end=22.)
        result=consistent_recovery(unit,self.segments(19.,22.,'やばい'),self.segments(21.3,22.1,'やばい'))
        self.assertAlmostEqual(result[0]['start'],21.3)

    def test_light_dialogue_keeps_native_clock_but_invalid_or_low_confidence_does_not(self):
        unit=dict(text='そう',start=10.,end=11.,confidence=.8,no_speech_prob=.1,
            words=self.segments()[0]['words'])
        self.assertTrue(native_clock_supported(unit,20.))
        self.assertFalse(native_clock_supported(dict(unit,confidence=.1),20.))
        self.assertFalse(native_clock_supported(dict(unit,end=10.),20.))
        self.assertTrue(very_low_content_confidence(unit,'low_confidence_speech','別の言葉'))
        self.assertFalse(very_low_content_confidence(unit,'low_confidence_speech','そう'))

    def test_partial_word_duplicate_and_bad_clock_are_not_content_evidence(self):
        self.assertIsNone(matching_words(self.segments(text='すごいですね'),'すごい'))
        self.assertIsNone(matching_words(self.segments()+self.segments(11.,12.),'すごい'))
        for a,b in [(10.,9.),(10.,15.),(float('nan'),11.)]:
            self.assertIsNone(matching_words(self.segments(a,b),'すごい'))

    def test_repeated_phrase_is_usable_only_at_its_own_audio_anchor(self):
        segments=self.segments(10.,11.)+self.segments(13.,14.)
        self.assertEqual(matching_words(segments,'すごい',(13.,14.))[0]['start'],13.)
        self.assertIsNone(matching_words(segments,'すごい',(10.,14.)))

    def test_energy_or_music_cannot_verify_a_vocal_event(self):
        self.assertFalse(vocal_event_supported({}))
        self.assertFalse(vocal_event_supported(dict(scores={'Music':.5,'Gasp':.2})))
        self.assertFalse(vocal_event_supported(dict(scores={'Speech':.9,'Music':0})))
        self.assertTrue(vocal_event_supported(dict(scores={'Music':.03,'Gasp':.35})))

    def test_background_omission_requires_strong_negative_voice_evidence(self):
        scores=dict(Music=.95,Speech=.01,Singing=.01,Breathing=.01)
        self.assertTrue(nonvocal_background_supported(dict(scores=scores)))
        self.assertFalse(nonvocal_background_supported(dict(scores={'Music':.95})))
        for label in ('Speech','Singing','Breathing','Gasp','Whispering'):
            self.assertFalse(nonvocal_background_supported(dict(scores=dict(scores,**{label:.1}))))


if __name__=='__main__':unittest.main()
