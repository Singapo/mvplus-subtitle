import sys
from pathlib import Path
from datetime import datetime,timezone
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'mvsub' / 'pipeline'))
from deepseek_offpeak import seconds_until_allowed

class OffPeakTests(unittest.TestCase):
    def delay(self,h,m=0,day=12,budget=360):
        return seconds_until_allowed(datetime(2026,10,day,h,m,tzinfo=timezone.utc),budget)
    def test_weekday_peak_windows(self):
        self.assertEqual(self.delay(1),3*3600)
        self.assertEqual(self.delay(6),4*3600)
    def test_boundary_end_is_allowed(self):
        self.assertEqual(self.delay(4),0)
        self.assertEqual(self.delay(10),0)
    def test_timeout_cannot_cross_peak_start(self):
        self.assertEqual(self.delay(0,54),3*3600+360)
        self.assertEqual(self.delay(0,53),0)
        self.assertEqual(self.delay(5,59),4*3600+60)
    def test_weekend_all_day(self):
        self.assertEqual(self.delay(1,day=17),0)
        self.assertEqual(self.delay(6,day=18),0)
    def test_beijing_conversion(self):
        from datetime import timedelta
        cn=timezone(timedelta(hours=8))
        self.assertEqual(seconds_until_allowed(datetime(2026,10,12,14,tzinfo=cn)),4*3600)
