"""mvsub — 115 云盘影片补字幕流水线。

导入本包即把 ``src/mvsub/pipeline`` 放进 ``sys.path``。流水线内部沿用上游脚本的
平铺导入风格（``import cloud_common``、``from asr_config import ...``），所以这里
只做一次路径引导，不改写任何模块的导入。把它硬改成包内相对导入不会带来收益，
只会让 500 KB 已验证代码同时面对两套模块身份（``cloud_common`` 与
``mvsub.pipeline.cloud_common`` 会各自加载一份），那是真实的 bug 来源。
"""
from __future__ import annotations

import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

__version__ = "0.1.5"
__all__ = ["PIPELINE_DIR", "__version__"]
