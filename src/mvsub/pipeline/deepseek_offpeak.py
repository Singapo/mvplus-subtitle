"""Conservative DeepSeek official off-peak request gate (checked per request)."""
from datetime import datetime, timedelta, timezone
import time

UTC = timezone.utc


def seconds_until_allowed(now=None, request_budget=360):
    """Allow weekends and weekday UTC hours outside 01-04 / 06-10.

    Leave time for the request timeout before the next peak boundary. Holidays
    deliberately use the ordinary weekday rule, which is conservative.
    """
    now = (now or datetime.now(UTC)).astimezone(UTC)
    if now.weekday() >= 5:
        return 0
    for start, end in ((1, 4), (6, 10)):
        left = now.replace(hour=start, minute=0, second=0, microsecond=0)
        right = now.replace(hour=end, minute=0, second=0, microsecond=0)
        if left - timedelta(seconds=request_budget) <= now < right:
            return (right - now).total_seconds()
    return 0


def wait_offpeak(timeout=300):
    announced = False
    while True:
        delay = seconds_until_allowed(request_budget=timeout + 60)
        if not delay:
            return
        if not announced:
            print(f'[off-peak] 等待优惠时段，约 {delay / 60:.0f} 分钟；不发送请求', flush=True)
            announced = True
        time.sleep(min(delay, 60))
