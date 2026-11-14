"""Resolved locations of the local ASR and forced-aligner deployments.

Models are never downloaded automatically: point ``model_home`` (or the
``MVSUB_MODEL_HOME`` variable) at a directory you prepared, and both the whisper
and aligner trees must exist before any extraction starts. See
docs/configuration.md.
"""
from config import (
    align_libs as _align_libs,
    align_model as _align_model,
    align_python as _align_python,
    asr_model as _asr_model,
    model_home as _model_home,
)

MODEL_HOME = _model_home()
ASR_MODEL = _asr_model()
ALIGN_MODEL = _align_model()
ALIGN_PYTHON = _align_python()
# The aligner imports this package's own aligned_asr, so the live tree is used
# instead of a stale copy under align-libs.
ALIGN_LIBS = _align_libs()
