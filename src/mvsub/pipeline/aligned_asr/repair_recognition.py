"""Compatibility entry point: preserve source ASR; never rewrite dialogue as sounds.

Content verification and alignment now record doubts or explicit omissions.
This former repair stage is intentionally absent from the production pipeline.
"""
import json
from pathlib import Path
import sys

if __name__ == '__main__':
    dest=Path(sys.argv[1])
    (dest/'recognition_repair.json').write_text(json.dumps(dict(repairs=[],attempts=[],
        disabled_reason='preserve_source_text_report_or_omit_uncertain_content'),indent=2),encoding='utf-8')
