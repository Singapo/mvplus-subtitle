"""A compressed container clock must not produce stretched extraction samples."""
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
import wave
import http.server
import functools
import threading
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from cloud_common import Cloud115

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class AudioClockTests(unittest.TestCase):
    def test_chunk_extraction_respects_container_clock(self):
        self.exercise(False)

    def test_rejected_authorization_bundle_is_refreshed(self):
        self.exercise(True)

    def exercise(self,reject_stale):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);source=directory/'clock.mp4'
            result=subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',
                'sine=frequency=440:sample_rate=44100:duration=10',
                '-af','asetpts=PTS*44100/48000','-c:a','aac','-movflags','+faststart',str(source)],capture_output=True)
            self.assertEqual(result.returncode,0)
            probe_allowed=[False];auth_lock=threading.Lock()
            class Handler(http.server.SimpleHTTPRequestHandler):
                def log_message(self,*args):pass
                def do_GET(self):
                    if self.path!='/clock.mp4':self.send_error(404);return
                    if reject_stale and self.headers.get('X-Auth')=='stale':
                        with auth_lock:
                            if probe_allowed[0]:self.send_error(403);return
                            probe_allowed[0]=True
                    data=source.read_bytes();range_header=self.headers.get('Range')
                    start,end=0,len(data)-1
                    if range_header:
                        limits=range_header.removeprefix('bytes=').split('-')
                        start=int(limits[0]);end=int(limits[1]) if limits[1] else end
                    self.send_response(206 if range_header else 200)
                    self.send_header('Content-Type','audio/mp4')
                    self.send_header('Accept-Ranges','bytes')
                    self.send_header('Content-Length',str(end-start+1))
                    if range_header:self.send_header('Content-Range',f'bytes {start}-{end}/{len(data)}')
                    self.end_headers()
                    try:self.wfile.write(data[start:end+1])
                    except (BrokenPipeError,ConnectionResetError):pass
            server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(directory)))
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            class Local(Cloud115):
                def __init__(self):self.log=lambda value:None;self.calls=0
                def download_url(self,pc):
                    self.calls+=1
                    return f'http://127.0.0.1:{server.server_port}/clock.mp4',{'X-Auth':'stale' if self.calls==1 and reject_stale else 'fresh'}
            target=directory/'sync.wav'
            client=Local()
            try:duration=client.extract_audio('local',target,chunk_seconds=4)
            finally:server.shutdown();server.server_close();worker.join()
            with wave.open(str(target),'rb') as handle:actual=handle.getnframes()/handle.getframerate()
            self.assertAlmostEqual(actual,duration,delta=.2)
            self.assertLess(actual,9.5)
            if reject_stale:self.assertGreater(client.calls,1)
            else:self.assertEqual(client.calls,1)
