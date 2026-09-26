"""字幕解析 / 序列化的往返测试。不需要 Qt 运行时。"""
from __future__ import annotations

import pytest

from app.core.subtitle_io import (
    Cue,
    SubtitleFormatError,
    format_timestamp,
    parse_srt,
    parse_timestamp,
    parse_vtt,
    to_srt,
    to_vtt,
)

SAMPLE_SRT = """1
00:00:01,000 --> 00:00:04,500
Hello world.

2
00:00:05,000 --> 00:00:07,250
Second line
continues here.
"""


def test_parse_srt_reads_index_timing_and_multiline_text():
    cues = parse_srt(SAMPLE_SRT)
    assert [cue.index for cue in cues] == [1, 2]
    assert cues[0].start == pytest.approx(1.0)
    assert cues[0].end == pytest.approx(4.5)
    assert cues[1].text == "Second line\ncontinues here."
    assert cues[0].duration == pytest.approx(3.5)


def test_timestamps_round_trip():
    assert format_timestamp(parse_timestamp("01:02:03,456")) == "01:02:03,456"
    assert format_timestamp(parse_timestamp("00:00:00,007")) == "00:00:00,007"
    # 单位数时/分/秒与不足三位的毫秒都要能解析
    assert format_timestamp(parse_timestamp("1:2:3.4")) == "01:02:03,400"


def test_translation_wins_over_original_text():
    cue = Cue(index=1, start=0.0, end=1.0, text="hello", translation="你好")
    assert cue.render_text() == "你好"
    cue.translation = "   "
    assert cue.render_text() == "hello"


def test_to_srt_renumbers_and_writes_translation():
    cues = parse_srt(SAMPLE_SRT)
    for cue in cues:
        cue.translation = f"译文{cue.index}"
    out = to_srt(cues)
    assert out.startswith("1\n00:00:01,000 --> 00:00:04,500\n译文1")
    assert "2\n00:00:05,000 --> 00:00:07,250\n译文2" in out


def test_parse_vtt_skips_header_and_cue_identifier():
    content = "WEBVTT\n\ncue-1\n00:00:01.000 --> 00:00:02.000 align:start\nHi\n"
    cues = parse_vtt(content)
    assert len(cues) == 1
    assert cues[0].text == "Hi"
    assert to_vtt(cues).startswith("WEBVTT\n\n00:00:01.000 --> 00:00:02.000")


def test_crlf_and_bom_are_tolerated():
    content = "\ufeff" + SAMPLE_SRT.replace("\n", "\r\n")
    cues = parse_srt(content)
    assert len(cues) == 2
    assert cues[0].text == "Hello world."


def test_bad_timecode_raises():
    with pytest.raises(SubtitleFormatError):
        parse_srt("1\nnot-a-timecode\nHello\n")


def test_empty_input_raises():
    with pytest.raises(SubtitleFormatError):
        parse_srt("\n\n")
