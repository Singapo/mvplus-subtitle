"""Held cues: `context` + empty must park, not kill the batch.

A film whose opening is mostly non-verbal produces a batch of pure sounds. The
system prompt tells the model to keep those IDs and mark them ``context``; the
parser used to reject exactly that, which failed the batch and — through the
budget guard — cancelled every remaining batch of the film. All the work still
to be done was lost because one batch was untranslatable by design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "mvsub" / "pipeline"))

import grounded_translation as grounded  # noqa: E402

TARGET = [("1", 0.0, 1.0, "ぴちゅ"), ("2", 1.0, 2.0, "ん、")]


def _reply(items: list[dict]) -> str:
    return json.dumps({"items": items}, ensure_ascii=False)


def test_empty_context_is_held_not_fatal():
    items, missing = grounded.parse_reply(
        _reply([{"id": "1", "zh": "", "review": "context"},
                {"id": "2", "zh": "嗯、", "review": "none"}]),
        TARGET,
    )

    assert items["1"] == {"id": "1", "zh": "", "review": "context"}
    assert items["2"]["zh"] == "嗯、"
    assert missing == []


def test_empty_without_a_review_is_still_rejected():
    """An empty line with nothing to explain it is a broken translation."""
    with pytest.raises(ValueError, match="Empty accepted translation"):
        grounded.parse_reply(_reply([{"id": "1", "zh": "   ", "review": "none"}]), TARGET)


def test_source_review_still_requires_a_span():
    with pytest.raises(ValueError, match="exact source span"):
        grounded.parse_reply(_reply([{"id": "1", "zh": "", "review": "source"}]), TARGET)


def test_held_cues_block_publication_but_keep_the_rest(tmp_path):
    """The film must not publish, yet everything else must still be paid for and cached."""
    output = tmp_path / "movie.srt"
    asked = []

    def ask(prompt: str, record: Path) -> str:
        asked.append(prompt)
        # First batch: the model holds the non-dialogue cue and translates the rest.
        payload = _reply([{"id": "1", "zh": "", "review": "context"},
                          {"id": "2", "zh": "嗯、", "review": "none"}])
        record.write_text(json.dumps({
            "prompt_sha256": grounded.sha(prompt.encode("utf-8")),
            "content_sha256": grounded.sha(payload.encode("utf-8")),
            "content": payload,
            "model": "deepseek-flash",
            "reasoning_effort": "high",
            "max_output_tokens": 16384,
            "response_format": "json",
            "finish_reason": "stop",
        }), encoding="utf-8")
        return payload

    report = grounded.translate(
        TARGET, output, "system prompt", ask,
        dict(model="deepseek-flash", reasoning_effort="high", max_output_tokens=16384),
        batch_size=16, max_workers=1,
    )

    assert report["held_ids"] == ["1"]
    assert report["review_ids"] == ["1"]
    assert report["quality_passed"] is False
    assert report["output_updated"] is False
    assert not output.exists(), "a held cue must never reach the published SRT"
    assert asked, "the batch must still be requested"
