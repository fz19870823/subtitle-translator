"""翻译基类与引擎注册表的通用测试。不访问网络。"""
from __future__ import annotations

import pytest

from app.core.subtitle_io import parse_srt
from app.core.translator import (
    ENGINES,
    EchoTranslator,
    TranslationCancelled,
    TranslationError,
    Translator,
    create_engine,
    create_engine_for,
    detect_script,
    is_translatable,
    locate_untranslated,
    looks_like_verbatim_echo,
    register,
    should_check_echo,
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


def test_should_stop_aborts_between_batches():
    """取消在批与批之间生效：不再发新请求，已经翻好的部分要留着。

    上千条字幕要发几十次请求、跑好几分钟，用户点取消后不该干等下去。
    """
    cues = parse_srt(SAMPLE)
    batches: list[int] = []

    class Counting(Translator):
        name = "counting"

        def translate_batch(self, requests):
            batches.append(len(requests))
            return [f"[zh-CN] {r.text}" for r in requests]

    asked: list[int] = []

    def should_stop() -> bool:
        asked.append(1)
        return len(asked) > 1  # 第一批放行，第二批之前喊停

    with pytest.raises(TranslationCancelled):
        Counting().translate_cues(
            cues,
            source_lang="en",
            target_lang="zh-CN",
            batch_size=1,
            should_stop=should_stop,
        )

    assert batches == [1], "第二批请求不该再发出去"
    assert cues[0].translation == "[zh-CN] hello", "已翻好的部分必须留下"
    assert cues[1].translation == "", "没轮到的条目保持未翻译"


def test_should_stop_before_first_batch_translates_nothing():
    cues = parse_srt(SAMPLE)
    with pytest.raises(TranslationCancelled):
        EchoTranslator().translate_cues(
            cues, source_lang="en", target_lang="zh-CN", should_stop=lambda: True
        )
    assert [cue.translation for cue in cues] == ["", ""]


def test_cancelled_is_not_a_translation_error():
    """取消不是故障：界面靠这个区分「取消」和「翻译失败」两种收尾。"""
    assert not issubclass(TranslationCancelled, TranslationError)


# ------------------------------------------------------------------ 断点续传


def test_skip_translated_only_translates_the_rest():
    """续传跳过已有译文 —— 这是断点能省下几分钟的直接原因。

    待翻条目**未必连成一片**：中途取消时，可能是第 1、2、4 条翻好了、第 3 条没有，
    所以要按下标挑而不是只挪起点。
    """
    cues = parse_srt(SAMPLE)
    cues[0].translation = "[zh-CN] 已翻"
    sent: list[str] = []

    class Recorder(Translator):
        name = "recorder"
        batch_size = 10

        def translate_batch(self, requests):
            sent.extend(r.text for r in requests)
            return [f"[zh-CN] {r.text}" for r in requests]

    Recorder().translate_cues(
        cues, source_lang="en", target_lang="zh-CN", skip_translated=True
    )

    assert sent == ["world"], "翻过的条目不该再发一次（白烧额度）"
    assert cues[0].translation == "[zh-CN] 已翻", "已有译文必须原样保留"
    assert cues[1].translation == "[zh-CN] world"


def test_skip_translated_spreads_the_rest_over_batches():
    """跳过的条目散落在中间时，批次要按剩下的条目重新切。"""
    long_sample = SAMPLE + """
3
00:00:05,000 --> 00:00:06,000
foo

4
00:00:07,000 --> 00:00:08,000
bar
"""
    cues = parse_srt(long_sample)
    for index in (0, 2):
        cues[index].translation = "[zh-CN] 已翻"
    batches: list[list[str]] = []

    class Recorder(Translator):
        name = "recorder"

        def translate_batch(self, requests):
            batches.append([r.text for r in requests])
            return [f"[zh-CN] {r.text}" for r in requests]

    Recorder().translate_cues(
        cues, source_lang="en", target_lang="zh-CN", batch_size=1, skip_translated=True
    )
    assert batches == [["world"], ["bar"]], "只剩 2 条要翻，就该只发这两条"


def test_skip_translated_reports_progress_from_the_checkpoint():
    cues = parse_srt(SAMPLE)
    cues[0].translation = "[zh-CN] 已翻"
    seen: list[tuple[int, int]] = []
    EchoTranslator().translate_cues(
        cues,
        source_lang="en",
        target_lang="zh-CN",
        batch_size=1,
        progress=lambda done, total: seen.append((done, total)),
        skip_translated=True,
    )
    assert seen == [(1, 2), (2, 2)], "进度要接着断点算，不能从头重数"


def test_skip_translated_with_nothing_left_makes_no_request():
    """全部翻完时不该再发任何请求（用户续传时又点了一次翻译）。"""
    cues = parse_srt(SAMPLE)
    for cue in cues:
        cue.translation = f"[zh-CN] {cue.text}"

    class Boom(Translator):
        name = "boom"

        def translate_batch(self, requests):
            raise AssertionError("没有待翻条目时不该发请求")

    Boom().translate_cues(
        cues, source_lang="en", target_lang="zh-CN", skip_translated=True
    )
    assert all(cue.is_translated for cue in cues)


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


# ------------------------------------------------------------ 整批原样回抄检测
#
# 背景：实测某些中继会间歇性把整个数组原样返回（12 次请求里中 2 次，两个方向都中）。
# 这是最危险的失败形态 —— 用户拿到一份没翻译的字幕却看不出来。


@pytest.mark.parametrize(
    "text,expected",
    [
        ("hello", True),
        ("こんにちは", True),
        ("你好", True),
        ("안녕하세요", True),
        ("Привет", True),
        ("", False),
        ("   ", False),
        ("123", False),
        ("---", False),
        ("♪♪", False),
        ("\u23ce", False),
        ("<i></i>", False),
        ("<i>hi</i>", True),  # 剥掉标签后还剩字母
        ("你好\u23ce再见", True),
    ],
)
def test_is_translatable(text, expected):
    assert is_translatable(text) is expected


def test_should_check_echo_skips_same_or_unknown_language():
    assert should_check_echo("en", "zh-CN") is True
    assert should_check_echo("ja", "ko") is True
    assert should_check_echo("auto", "zh-CN") is False
    assert should_check_echo("ja", "ja") is False
    assert should_check_echo("zh-CN", "zh-CN") is False
    # 判不了就问不出结论：宁可不查，也别把正常任务判成失败
    assert should_check_echo("", "ja") is False
    # 同一门语言的地区/字形变体：大量条目本就该原样，回抄比例天然偏高
    assert should_check_echo("en-GB", "en-US") is False
    assert should_check_echo("zh-CN", "zh-TW") is False


def test_echo_detection_requires_min_items_and_majority():
    src = ["hello", "world", "good"]
    assert looks_like_verbatim_echo(src, list(src), "en", "zh-CN") is True
    # 只有 1/3 回抄，属于正常现象（专有名词等），不判故障
    assert looks_like_verbatim_echo(src, ["你好", "world", "好的"], "en", "zh-CN") is False
    # 可比条目不足 3 条时不判定
    assert looks_like_verbatim_echo(["a", "b"], ["a", "b"], "en", "zh-CN") is False
    # 源语言=目标语言，本就不该翻译
    assert looks_like_verbatim_echo(src, list(src), "en", "en") is False


def test_echo_detection_ignores_untranslatable_lines():
    # 符号行、数字行原样保留是正常的，不能算进"回抄"
    src = ["123", "---", "♪♪", "OK!", "hello"]
    out = ["123", "---", "♪♪", "OK!", "hello"]
    # 可比条目只有 "OK!" "hello" 两条，达不到 min_items=3
    assert looks_like_verbatim_echo(src, out, "en", "zh-CN") is False


# ------------------------------------------------------------------ auto 源语言
#
# 实测背景（2026-09-26）：界面默认源语言就是 auto，而旧的 should_check_echo
# 见到 auto 直接返回 False。于是 grok-chat-fast + 日文字幕 8 批里那 2 批
# 「整批 20 条原样退回」全被静默交付 —— 用户拿到一份没翻译的字幕却看不出异常。


def test_should_check_echo_infers_source_from_text_when_auto():
    """源语言是 auto 时必须靠文本把真实语言认出来，不能直接放弃检查。"""
    ja = ["何度も言ったはずだ。", "そんなの無理だよ。", "分かった。任せるよ。"]
    assert should_check_echo("auto", "zh-CN", samples=ja) is True
    assert should_check_echo("auto", "zh-CN", samples=["hello", "world"]) is True

    # 目标就是中文、原文也是中文：整批原样返回是**正确**行为，不能误报
    zh = ["我已经说过很多次了。", "那种事做不到。"]
    assert should_check_echo("auto", "zh-CN", samples=zh) is False
    # 英文原文译英文：同理不该查
    assert should_check_echo("auto", "en", samples=["hello", "world"]) is False

    # 没给样本还是判不了 —— 退化成原来的保守行为
    assert should_check_echo("auto", "zh-CN") is False


def test_detect_script_picks_kana_before_han():
    # 日语句子同时含汉字与假名，必须先认成日语，否则会被当成中文原文
    assert detect_script("東京駅で待っている。") == "ja"
    assert detect_script("何度も言ったはずだ。") == "ja"
    assert detect_script("我已经说过很多次了。") == "han"
    assert detect_script("hello world") == "latin"
    assert detect_script("123 ---") is None


def test_guess_text_family_survives_a_few_pure_kanji_lines():
    """整批投票：夹着几条纯汉字的日文句子也不该把整批带偏。"""
    from app.core.translator import guess_text_family

    mixed = ["何度も言ったはずだ。", "東京駅", "そんなの無理だよ。", "学校"]
    assert guess_text_family(mixed) == "ja"


def test_locate_untranslated_spares_legitimately_identical_items():
    """同形汉字词原样保留是合法的，只有"书写族根本不对"才能定罪。"""
    src = ["何度も言ったはずだ。", "学校", "hello"]
    out = list(src)
    # 0 = 假名句没翻，1 = 「学校」对中文本就该原样（放过），2 = 英文没翻
    assert locate_untranslated(src, out, "zh-CN") == [0, 2]

    # 已经翻过的条目不参与
    out2 = ["我已经说过很多次了。", "学校", "你好"]
    assert locate_untranslated(src, out2, "zh-CN") == []


def test_locate_untranslated_is_empty_when_target_unknown():
    src = ["hello", "world"]
    assert locate_untranslated(src, list(src), "auto") == []
    assert locate_untranslated(src, list(src), "") == []


# ------------------------------------------------------------------ 质量插曲


def test_quality_notes_can_skip_the_untranslated_line():
    """界面自己会从字幕里再数一遍未翻译条数，引擎不该再报一个数。

    两边都报会变成状态栏里「仍有 3 条未翻译　|　仍有 3 条疑似未翻译」，
    而且引擎只知道这一跑里撞上几条、界面知道整份还剩几条 —— 两个数还可能不一样。
    """

    class Noisy(Translator):
        name = "noisy"
        untranslated_count = 3

        def translate_batch(self, requests):
            return [r.text for r in requests]

    engine = Noisy()
    assert any("3 条疑似未翻译" in note for note in engine.quality_notes())
    notes = engine.quality_notes(include_untranslated=False)
    assert not any("疑似未翻译" in note for note in notes)
