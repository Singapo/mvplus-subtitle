import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from audio_archive import archive_audio,restore_audio,digest


class AudioArchiveTests(unittest.TestCase):
    def wav(self,path,sample=b'\x00\x00'):
        with wave.open(str(path),'wb') as stream:
            stream.setparams((1,2,16000,0,'NONE','not compressed'));stream.writeframes(sample*160)

    def test_archive_survives_job_cleanup_and_restores_exact_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source.wav';self.wav(source)
            sha=digest(source);saved,meta=archive_audio(source,'Film','video1',base/'originals')
            source.unlink();self.assertEqual(digest(saved),sha)
            restored=base/'job/full_audio.wav'
            self.assertTrue(restore_audio(restored,'Film','video1',base/'originals'))
            self.assertEqual(digest(restored),sha)
            self.assertFalse(restore_audio(base/'wrong.wav','Film','video2',base/'originals'))

    def test_different_audio_keeps_existing_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source.wav';self.wav(source)
            saved,_=archive_audio(source,'Film','video1',base/'originals');first=digest(saved)
            source.unlink();self.wav(source,b'\x01\x00')
            second,_=archive_audio(source,'Film','video1',base/'originals')
            self.assertNotEqual(saved,second);self.assertEqual(digest(saved),first)

    def test_invalid_archive_never_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source.wav';self.wav(source)
            saved,_=archive_audio(source,'Film','video1',base/'originals')
            saved.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):restore_audio(base/'target.wav','Film','video1',base/'originals')
            self.assertFalse((base/'target.wav').exists())

    def test_truncated_wav_not_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);source=base/'source.wav';self.wav(source)
            source.write_bytes(source.read_bytes()[:-2])
            with self.assertRaises(ValueError):archive_audio(source,'Film','video1',base/'originals')
            self.assertFalse((base/'originals').exists())

