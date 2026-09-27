"""未翻译条目的处置链路：置空 → 打标记 → 边翻边落盘 → 队列收尾后重试。

这条链路是「一批里个别条目翻不出来」的完整出路。任何一环断了都会**静默**劣化：
成品看上去完好，实际少了几句，而界面上没有任何地方能看出来。

- **置空** —— 留着原文的话，导出时它和「本来就该是原文」的条目分不出来；
- **打标记** —— 观众至少该看出「这儿有一句没翻」，而不是以为本来就没字幕；
- **边翻边落盘** —— 翻到哪磁盘上就是哪，关窗/掉电不必前功尽弃；
- **重试轮** —— 整条队列跑完后换个前提再试一次，且**不打断**其他文件。

这一层（subtitle_io / translator / queue）刻意与 Qt 解耦，所以不需要图形环境；
界面那一端另见 ``test_ui_smoke.py``。
"""
from __future__ import annotations

from app.core import queue as q
from app.core.subtitle_io import (
    UNTRANSLATED_MARK,
    Cue,
    OutputSink,
    parse_srt,
    to_srt,
    write_file,
)
from app.core.translator import Translator, _apply_results

SAMPLE = """1
00:00:01,000 --> 00:00:02,000
hello

2
00:00:03,000 --> 00:00:04,000
world
"""

#: 第 1 条是纯符号行（本来就不需要翻译），第 2 条是日文。
SYMBOLS = """1
00:00:01,000 --> 00:00:02,000
♪♪

2
00:00:03,000 --> 00:00:04,000
こんにちは
"""


class _EchoBack(Translator):
    """把原文原样退回 —— 真实故障的形态（中继把请求路由到了不干活的通道）。"""

    name = "echoback"

    def translate_batch(self, requests):
        return [request.text for request in requests]


class _Clock:
    """手控时钟：验证节流而不必真的 sleep（真睡会让测试慢且不稳）。"""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------ 导出形态


def test_untranslated_entry_renders_as_a_marker_not_the_original():
    """没翻出来的条目要留标记，**不能**悄悄回落到原文。

    回落原文的后果：用户拿到一份看起来完整的字幕，实际那条根本没翻，
    而文件里完全看不出来 —— 这正是「回抄」这个故障最坏的地方。
    """
    cue = Cue(index=1, start=0.0, end=1.0, text="こんにちは", failed=True)
    assert cue.render_text() == UNTRANSLATED_MARK


def test_untranslatable_line_keeps_its_own_text():
    """纯符号行不送模型，也不该被当成「没翻出来」而标成 [未翻译]。"""
    cue = Cue(index=1, start=0.0, end=1.0, text="♪♪")
    assert cue.failed is False
    assert cue.render_text() == "♪♪"


def test_blanked_entry_still_counts_as_untranslated():
    """置空的条目要继续算「未翻译」，续传与手动重试才能重新挑出它。"""
    cue = Cue(index=1, start=0.0, end=1.0, text="hello", failed=True)
    assert cue.is_translated is False


def test_export_keeps_the_index_and_timeline_of_a_failed_entry():
    """序号与时间轴必须留住 —— 观众至少知道「这里本来有一句话」。"""
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="hello", translation="你好"),
        Cue(index=2, start=1.0, end=2.0, text="world", failed=True),
    ]
    out = to_srt(cues)
    assert f"2\n00:00:01,000 --> 00:00:02,000\n{UNTRANSLATED_MARK}" in out


# ------------------------------------------------------------------ 原子写


def test_write_file_is_atomic_and_leaves_no_tmp_behind(tmp_path):
    """写盘要原子：用户随时可能打开那个文件，半截字幕比没写更糟。"""
    target = tmp_path / "out" / "a.srt"
    written = write_file(target, [Cue(index=1, start=0.0, end=1.0, text="hi")])

    assert written == target
    assert target.exists()
    assert list(target.parent.glob("*.tmp")) == [], "临时文件不许留在原地"


# ------------------------------------------------------------------ 边翻边写


def test_output_sink_writes_the_first_batch_then_throttles(tmp_path):
    """第一批必写，之后按间隔节流。

    第一批必写是有理由的：进程刚起来时 clock() 也可能很小，若用 clock() 当起点，
    「前两秒什么都没落盘」；那两秒内崩掉，用户手里就一个字节都没有。
    """
    clock = _Clock()
    sink = OutputSink(tmp_path / "b.srt", interval=2.0, clock=clock)
    cues = [Cue(index=1, start=0.0, end=1.0, text="hi", translation="你好")]

    assert sink.maybe(cues) is True, "第一批必须落盘"
    stamp = sink.path.stat().st_mtime_ns

    clock.now += 1.0
    assert sink.maybe(cues) is False, "没到间隔不该写"
    assert sink.path.stat().st_mtime_ns == stamp

    clock.now += 1.5
    assert sink.maybe(cues) is True, "过了间隔就该写"


def test_output_sink_flush_ignores_the_interval(tmp_path):
    """收尾（取消 / 失败 / 成功）必须能强制写一次，不受节流限制。"""
    clock = _Clock()
    sink = OutputSink(tmp_path / "c.srt", interval=999.0, clock=clock)
    cues = [Cue(index=1, start=0.0, end=1.0, text="hi", translation="你好")]

    sink.maybe(cues)
    clock.now += 1.0
    assert sink.flush(cues) is True
    assert sink.error == ""


def test_output_sink_records_a_write_failure_instead_of_raising(tmp_path):
    """写不进去要说出来，而不是把翻译整个掀掉。

    用户以为成品正一路落盘、实际一个字节都没写进去，属于「不特意说一声就永远
    不会知道」的故障 —— 界面会在状态栏把它报出来（见 MainWindow._sink_note）。
    """
    blocker = tmp_path / "blocked"
    blocker.write_text("我是文件，不是目录", encoding="utf-8")
    sink = OutputSink(blocker / "x.srt")

    assert sink.maybe([Cue(index=1, start=0.0, end=1.0, text="hi")]) is False
    assert sink.error, "失败原因必须被记下来"


def test_output_sink_writes_what_the_exporter_would(tmp_path):
    """落盘内容必须与「导出」一致 —— 不然用户看到的成品和界面里核对的不一样。"""
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="hello", translation="你好"),
        Cue(index=2, start=1.0, end=2.0, text="world", failed=True),
    ]
    sink = OutputSink(tmp_path / "d.srt")
    assert sink.flush(cues) is True
    assert sink.path.read_text(encoding="utf-8") == to_srt(cues)


# ------------------------------------------------------------------ 写回判据


def test_apply_results_blanks_and_flags_echoed_items():
    """引擎交回来的回抄值不许当译文收下：置空 + 打标记。"""
    cues = [
        Cue(index=1, start=0.0, end=1.0, text="hello"),
        Cue(index=2, start=1.0, end=2.0, text="world"),
    ]
    flagged = _apply_results(cues, ["你好", "world"], "zh-CN")

    assert flagged == 1
    assert cues[0].translation == "你好" and cues[0].failed is False
    assert cues[1].translation == "" and cues[1].failed is True


def test_apply_results_spares_same_script_words():
    """日文「学校」译成中文仍是「学校」—— 同形词不是失败，不能误标。"""
    cues = [Cue(index=1, start=0.0, end=1.0, text="学校")]
    assert _apply_results(cues, ["学校"], "zh-CN") == 0
    assert cues[0].failed is False
    assert cues[0].translation == "学校"


def test_apply_results_clears_a_stale_failure_flag():
    """上一轮标过失败、这一轮翻好了，标记必须撤掉。

    留着的话，这条会在导出时被写成 [未翻译]，把刚救回来的成果又盖掉。
    """
    cues = [Cue(index=1, start=0.0, end=1.0, text="world", failed=True)]
    assert _apply_results(cues, ["世界"], "zh-CN") == 0
    assert cues[0].failed is False
    assert cues[0].render_text() == "世界"


def test_translate_cues_blank_and_flag_through_the_whole_pipeline():
    """端到端：整批都翻不出来时，每一条都被置空并标上 failed。"""
    cues = parse_srt(SAMPLE)
    _EchoBack().translate_cues(cues, source_lang="en", target_lang="zh-CN")

    assert [cue.translation for cue in cues] == ["", ""]
    assert [cue.failed for cue in cues] == [True, True]


def test_translate_cues_leaves_symbol_lines_alone():
    """符号行原样保留，只有真正该翻却没翻出来的那条才被标记。"""
    cues = parse_srt(SYMBOLS)
    _EchoBack().translate_cues(cues, source_lang="auto", target_lang="zh-CN")

    assert cues[0].failed is False and cues[0].render_text() == "♪♪"
    assert cues[1].failed is True and cues[1].render_text() == UNTRANSLATED_MARK


# ------------------------------------------------------------------ 队列重试轮


def _queue(tmp_path, *names: str) -> q.TranslationQueue:
    source = tmp_path / "src"
    source.mkdir(parents=True, exist_ok=True)
    for name in names:
        (source / name).write_text("subtitle", encoding="utf-8")
    batch = q.TranslationQueue(tmp_path / "out")
    batch.add([source / name for name in names])
    return batch


def test_mark_retry_round_picks_only_files_with_untranslated_items(tmp_path):
    """只挑「翻完了、但还剩没翻出来的」那几份，其余一动不动。"""
    batch = _queue(tmp_path, "01.srt", "02.srt")
    batch.mark_done(0)
    batch.mark_done(1)
    batch[0].untranslated = 3

    assert batch.mark_retry_round() == 1
    assert batch[0].status == q.PENDING and batch[0].retry_only is True
    assert batch[1].status == q.DONE and batch[1].retry_only is False


def test_mark_retry_round_returns_zero_when_nothing_is_left(tmp_path):
    """没有未翻译条目时返回 0 —— 调用方据此拒绝启动一次空跑。"""
    batch = _queue(tmp_path, "01.srt")
    batch.mark_done(0)

    assert batch.mark_retry_round() == 0
    assert batch[0].status == q.DONE


def test_reset_clears_the_retry_only_flag(tmp_path):
    """``retry_only`` 是「这一轮」的属性，重置时必须一起清掉。

    不然后面某次普通的「继续队列」会莫名其妙地只翻那几条，
    而用户以为自己点的是整份重来。
    """
    batch = _queue(tmp_path, "01.srt")
    batch.mark_done(0)
    batch[0].untranslated = 1
    batch.mark_retry_round()

    batch.reset(0)
    assert batch[0].retry_only is False


def test_summary_counts_files_that_still_have_untranslated_items(tmp_path):
    batch = _queue(tmp_path, "01.srt", "02.srt", "03.srt")
    for index in range(3):
        batch.mark_done(index)
    batch[0].untranslated = 2
    batch[2].untranslated = 5

    counts = batch.summary()
    assert counts["untranslated"] == 7
    assert counts["stuck_files"] == 2, "「重试未翻译」按钮要显示涉及几个文件"


# ------------------------------------------------------------------ 不中断


def test_a_failed_file_does_not_block_the_rest(tmp_path):
    """一份失败不该拖住后面 —— 队列是给人不在场时用的。"""
    batch = _queue(tmp_path, "01.srt", "02.srt")
    batch.fail(0, "boom")

    assert batch.next_pending() == 1


def test_a_file_with_untranslated_items_does_not_block_the_rest(tmp_path):
    """还剩几条没翻出来，也算这一份跑完了：不阻塞，留到收尾后一起重试。"""
    batch = _queue(tmp_path, "01.srt", "02.srt")
    batch.mark_done(0)
    batch[0].untranslated = 3

    assert batch.next_pending() == 1
