"""翻译基类与引擎注册表的通用测试。不访问网络。"""
from __future__ import annotations

import pytest

from app.core.subtitle_io import parse_srt
from app.core.translator import (
    ENGINES,
    EchoTranslator,
    TranslationError,
    Translator,
    create_engine,
    create_engine_for,
    register,
)

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
hello

2
00:00:03,000 --> 00:00:04,000
world
"""


def test_builtin_engines_are_registered():
    assert "echo" in ENGINES
    assert "openai" in ENGINES
    assert ENGINES["echo"] is EchoTranslator


def test_create_engine_unknown_name_raises():
    with pytest.raises(TranslationError) as info:
        create_engine("no-such-engine")
    assert "no-such-engine" in str(info.value)
    assert "echo" in str(info.value)


def test_echo_prefixes_every_cue():
    cues = parse_srt(SAMPLE)
    EchoTranslator().translate_cues(cues, source_lang="en", target_lang="zh-CN")
    assert [c.translation for c in cues] == ["[zh-CN] hello", "[zh-CN] world"]


def test_progress_callback_reports_cumulative_counts():
    cues = parse_srt(SAMPLE)
    seen: list[tuple[int, int]] = []
    EchoTranslator().translate_cues(
        cues, source_lang="en", target_lang="zh-CN",
        batch_size=1, progress=lambda d, t: seen.append((d, t)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError):
        EchoTranslator().translate_cues(
            parse_srt(SAMPLE), source_lang="en", target_lang="zh-CN", batch_size=0
        )
    with pytest.raises(ValueError):
        EchoTranslator().translate_cues(
            parse_srt(SAMPLE), source_lang="en", target_lang="zh-CN", batch_size=-3
        )


def test_batch_size_none_uses_engine_default():
    cues = parse_srt(SAMPLE)
    seen: list[tuple[int, int]] = []
    EchoTranslator().translate_cues(
        cues, source_lang="en", target_lang="zh-CN",
        batch_size=None, progress=lambda d, t: seen.append((d, t)),
    )
    # EchoTranslator.batch_size 默认 10，两条一次吃完
    assert seen == [(2, 2)]


class _WrongLength(Translator):
    name = "wrong-length"

    def translate_batch(self, requests):
        return ["only-one"]


def test_engine_returning_wrong_length_is_rejected():
    engine = _WrongLength()
    with pytest.raises(TranslationError) as info:
        engine.translate_cues(
            parse_srt(SAMPLE), source_lang="en", target_lang="zh-CN"
        )
    assert "期望 2 条" in str(info.value)


def test_register_rejects_missing_or_base_name():
    class Nameless(Translator):
        def translate_batch(self, requests):
            return []

    with pytest.raises(ValueError):
        register(Nameless)

    class BaseNamed(Translator):
        name = "base"

        def translate_batch(self, requests):
            return []

    with pytest.raises(ValueError):
        register(BaseNamed)


def test_create_engine_for_without_config_uses_no_arg_constructor():
    engine = create_engine_for("echo", None)
    assert isinstance(engine, EchoTranslator)
