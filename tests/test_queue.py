"""批量队列的状态机测试。

这一层刻意不含 Qt 依赖，所以能在只装了 pytest 的那套解释器里跑（本机 PySide6
与 pytest 装在两套 Python 里）。界面那一端的行为另见 ``test_ui_smoke.py``。
"""
from __future__ import annotations

from pathlib import Path

from app.core import queue as q


def make_file(directory: Path, name: str, content: str = "subtitle") -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build(tmp_path: Path, *names: str) -> q.TranslationQueue:
    """建一个队列，顺带在磁盘上真造出这几个文件（``add`` 会检查文件存在）。"""
    source = tmp_path / "src"
    for name in names:
        make_file(source, name)
    return q.TranslationQueue(tmp_path / "out")


def add(batch: q.TranslationQueue, tmp_path: Path, *names: str):
    return batch.add([tmp_path / "src" / name for name in names])


# ------------------------------------------------------------------ 入队


def test_add_puts_the_output_next_to_the_configured_directory(tmp_path):
    batch = build(tmp_path, "01.srt")
    added, notes = add(batch, tmp_path, "01.srt")

    assert notes == []
    assert len(added) == 1
    assert added[0].output_path == tmp_path / "out" / "01.translated.srt"
    assert added[0].status == q.PENDING
    assert batch.summary()["total"] == 1


def test_add_keeps_the_lowercase_extension_of_the_source(tmp_path):
    """扩展名决定导出格式（``subtitle_io.write_file`` 按后缀选 SRT/VTT）。"""
    batch = build(tmp_path, "01.SRT")
    added, _ = add(batch, tmp_path, "01.SRT")
    assert added[0].output_path.name == "01.translated.srt"


def test_add_skips_duplicates_and_says_why(tmp_path):
    """同一个文件加两次不该翻两遍 —— 白烧额度，而且第二遍会覆盖第一遍的成果。"""
    batch = build(tmp_path, "01.srt")
    add(batch, tmp_path, "01.srt")
    added, notes = add(batch, tmp_path, "01.srt")

    assert added == []
    assert len(batch) == 1
    assert "已在队列中" in notes[0]


def test_add_skips_unsupported_and_missing_files_with_reasons(tmp_path):
    """跳过的原因必须说出来：用户选了 20 个文件只进来 18 个却毫无提示，
    他会以为全在里面。"""
    batch = build(tmp_path, "01.srt")
    (tmp_path / "src" / "02.ass").write_text("x", encoding="utf-8")

    added, notes = batch.add(
        [
            tmp_path / "src" / "01.srt",
            tmp_path / "src" / "02.ass",
            tmp_path / "src" / "03.srt",  # 不存在
        ]
    )

    assert [item.path.name for item in added] == ["01.srt"]
    assert len(notes) == 2
    assert any("不支持" in note for note in notes)
    assert any("不是可读的文件" in note for note in notes)


def test_same_name_from_different_directories_gets_its_own_output(tmp_path):
    """同名不同目录的两份字幕必须落到两个译文文件上。

    都写成 ``01.translated.srt`` 的话，后翻完的会把先翻完的覆盖掉，
    而队列列表上两份都显示「完成」—— 用户拿到手的只有一份，还看不出少了什么。
    """
    batch = build(tmp_path, "a/01.srt", "b/01.srt")
    added, _ = batch.add([tmp_path / "src" / "a" / "01.srt", tmp_path / "src" / "b" / "01.srt"])

    assert len(added) == 2
    assert added[0].output_path.name == "01.translated.srt"
    assert added[1].output_path.name == "01.translated-2.srt"


def test_existing_output_on_disk_is_overwritten_not_renamed(tmp_path):
    """磁盘上已有同名译文不加序号：重跑一遍本来就是要覆盖上一轮的结果。

    加序号只会让输出目录堆起一堆 ``-2`` ``-3``，谁是最新的反而看不出来。
    """
    batch = build(tmp_path, "01.srt")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "01.translated.srt").write_text("上一轮的结果", encoding="utf-8")

    added, _ = add(batch, tmp_path, "01.srt")
    assert added[0].output_path.name == "01.translated.srt"


# ------------------------------------------------------------------ 顺序


def test_translation_order_follows_the_list_and_move(tmp_path):
    batch = build(tmp_path, "01.srt", "02.srt", "03.srt")
    add(batch, tmp_path, "01.srt", "02.srt", "03.srt")

    assert batch.next_pending() == 0
    batch.mark_done(0)
    assert batch.next_pending() == 1

    assert batch.move(2, -1) == 1
    assert [item.name for item in batch] == ["01.srt", "03.srt", "02.srt"]


def test_move_at_the_edges_stays_put(tmp_path):
    batch = build(tmp_path, "01.srt", "02.srt")
    add(batch, tmp_path, "01.srt", "02.srt")

    assert batch.move(0, -1) == 0
    assert batch.move(1, 1) == 1
    assert [item.name for item in batch] == ["01.srt", "02.srt"]


# ------------------------------------------------------------------ 状态


def test_failed_item_is_not_retried_in_the_same_round(tmp_path):
    """失败的项不能当场自动重来。

    否则一个注定失败的文件（整条链路都挂了）会被反复重发，队列永远跑不完 ——
    而用户以为自己只是点了「依次翻译」一次。
    """
    batch = build(tmp_path, "01.srt", "02.srt", "03.srt")
    add(batch, tmp_path, "01.srt", "02.srt", "03.srt")

    batch.mark_running(0)
    batch.fail(0, "HTTP 502")

    assert batch.next_pending() == 1, "失败的要跳过，而不是重来"
    assert batch[0].status == q.FAILED
    assert batch[0].error == "HTTP 502"


def test_reset_unfinished_keeps_what_already_succeeded(tmp_path):
    """再点「依次翻译」：失败和没轮到的重来，已经翻好的不白跑第二遍。"""
    batch = build(tmp_path, "01.srt", "02.srt", "03.srt")
    add(batch, tmp_path, "01.srt", "02.srt", "03.srt")
    batch.mark_running(0)
    batch.mark_done(0)
    batch.mark_running(1)
    batch.fail(1, "boom")

    assert batch.reset_unfinished() == 2
    assert batch[0].status == q.DONE, "成功的原样留着"
    assert batch[1].status == q.PENDING
    assert batch[1].error == "", "重置要清掉上一轮的报错，否则会一直挂着吓人"
    assert batch[2].status == q.PENDING
    assert batch.next_pending() == 1


def test_reset_all_backs_every_file_to_pending(tmp_path):
    batch = build(tmp_path, "01.srt", "02.srt")
    add(batch, tmp_path, "01.srt", "02.srt")
    batch.mark_running(0)
    batch.mark_done(0)

    assert batch.reset_all() == 2
    assert all(item.status == q.PENDING for item in batch)
    assert batch.next_pending() == 0


def test_running_item_cannot_be_removed_or_cleared(tmp_path):
    """正在翻的那一项不能动：后台线程还在往它的 cues 里写译文。"""
    batch = build(tmp_path, "01.srt", "02.srt")
    add(batch, tmp_path, "01.srt", "02.srt")
    batch.mark_running(0)

    assert batch.remove(batch[0]) is False
    batch.clear()
    assert len(batch) == 2, "有文件在跑时清空队列会让状态机对不上号"

    batch.mark_done(0)
    assert batch.remove(batch[0]) is True
    assert [item.name for item in batch] == ["02.srt"]


def test_summary_counts_every_state(tmp_path):
    batch = build(tmp_path, "01.srt", "02.srt", "03.srt", "04.srt")
    add(batch, tmp_path, "01.srt", "02.srt", "03.srt", "04.srt")
    for item, count in zip(batch, (10, 20, 30, 40)):
        item.cue_count = count

    batch.mark_running(0)
    batch.mark_done(0)
    batch[0].untranslated = 2
    batch.mark_running(1)
    batch.fail(1, "boom")

    counts = batch.summary()
    assert counts["total"] == 4
    assert counts["done"] == 1
    assert counts["failed"] == 1
    assert counts["pending"] == 2
    assert counts["unfinished"] == 3, "待翻译 + 失败 = 按钮上该显示的数字"
    assert counts["untranslated"] == 2
    assert counts["cue_count"] == 100


def test_resumed_count_is_reported(tmp_path):
    batch = build(tmp_path, "01.srt")
    add(batch, tmp_path, "01.srt")
    batch[0].resumed = 12
    batch.mark_running(0)
    batch.mark_done(0)

    assert batch.summary()["resumed"] == 12


def test_mark_done_clears_a_previous_failure_reason(tmp_path):
    """重跑成功后不该还挂着上一轮的报错 —— 列表上会写成「完成：HTTP 502」。"""
    batch = build(tmp_path, "01.srt")
    add(batch, tmp_path, "01.srt")
    batch.mark_running(0)
    batch.fail(0, "HTTP 502")
    batch.mark_running(0)
    batch.mark_done(0)

    assert batch[0].status == q.DONE
    assert batch[0].error == ""


def test_one_line_error_flattens_and_truncates():
    item = q.QueueItem(
        path=Path("a.srt"),
        output_path=Path("a.translated.srt"),
        error="第一行\n\n第二行\n" + "长" * 200,
    )
    text = item.one_line_error(limit=40)

    assert "\n" not in text, "换行会把列表行高撑开"
    assert len(text) == 40
    assert text.endswith("…")


def test_contains_matches_the_same_file_through_another_spelling(tmp_path):
    """``./src/01.srt`` 和 ``src/01.srt`` 是同一个文件，不该被当成两个。"""
    batch = build(tmp_path, "01.srt")
    add(batch, tmp_path, "01.srt")
    assert batch.contains(tmp_path / "src" / ".." / "src" / "01.srt") is True
    assert batch.contains(tmp_path / "src" / "02.srt") is False
